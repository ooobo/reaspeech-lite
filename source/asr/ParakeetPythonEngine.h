#pragma once

#include <memory>
#include <string>
#include <vector>

#include <juce_core/juce_core.h>
#include <juce_audio_formats/juce_audio_formats.h>

#include "../utils/SafeUTF8.h"
#include "ASROptions.h"
#include "ASRSegment.h"

class ParakeetPythonEngine
{
public:
    ParakeetPythonEngine (const std::string& modelsDirIn) : modelsDir (modelsDirIn) {}

    ~ParakeetPythonEngine()
    {
        DBG ("ParakeetPythonEngine destructor");
    }

    // Download the model - for Python version, we rely on onnx-asr to download models
    bool downloadModel (const std::string& /*modelName*/, std::function<bool ()> /*isAborted*/)
    {
        // Check if Python and onnx-asr are available
        if (! checkPythonAvailable())
        {
            DBG ("Python or onnx-asr not available");
            return false;
        }

        progress.store (100);
        return true;
    }

    // Load the model - for Python version, this is a no-op (model loads on each transcription)
    bool loadModel (const std::string& modelName)
    {
        DBG ("ParakeetPythonEngine::loadModel: " + modelName);

        if (! checkPythonAvailable())
        {
            DBG ("Python or onnx-asr not available");
            return false;
        }

        lastModelName = modelName;
        return true;
    }

    // Transcribe the audio data using Python subprocess
    bool transcribe (
        const std::vector<float>& audioData,
        ASROptions& /*options*/,
        std::vector<ASRSegment>& segments,
        std::function<bool ()> isAborted)
    {
        DBG ("ParakeetPythonEngine::transcribe");

        auto startTime = juce::Time::getMillisecondCounterHiRes();
        progress.store (0);

        try
        {
            // Create temporary WAV file
            auto tempFile = juce::File::getSpecialLocation (juce::File::tempDirectory)
                .getChildFile ("reaspeech_temp_" + juce::String (juce::Random::getSystemRandom().nextInt()) + ".wav");

            if (! writeWavFile (tempFile, audioData, 16000))
            {
                DBG ("Failed to write temporary WAV file");
                return false;
            }

            progress.store (20);

            if (isAborted())
            {
                tempFile.deleteFile();
                return false;
            }

            // Create Python script to run transcription
            auto transcriptionResult = runPythonTranscription (tempFile.getFullPathName(), isAborted);

            // Clean up temp file
            tempFile.deleteFile();

            if (transcriptionResult.isEmpty())
            {
                DBG ("Python transcription returned empty result");
                return false;
            }

            progress.store (90);

            // Parse the result - split by lines (one sentence per line)
            juce::StringArray lines;
            lines.addLines (transcriptionResult);

            // Create segments from sentences
            // For now, distribute them evenly across the audio duration
            // (Parakeet doesn't provide timestamps, so this is an approximation)
            float totalDuration = static_cast<float> (audioData.size()) / 16000.0f;
            int numLines = 0;

            // Count non-empty lines
            for (const auto& line : lines)
                if (line.trim().isNotEmpty())
                    numLines++;

            if (numLines > 0)
            {
                float segmentDuration = totalDuration / numLines;
                float currentTime = 0.0f;

                for (const auto& line : lines)
                {
                    auto trimmedLine = line.trim();
                    if (trimmedLine.isEmpty())
                        continue;

                    ASRSegment segment;
                    segment.text = trimmedLine;
                    segment.start = currentTime;
                    segment.end = currentTime + segmentDuration;
                    segments.push_back (segment);

                    currentTime += segmentDuration;
                }
            }
            else
            {
                // Fallback: single segment with all text
                ASRSegment segment;
                segment.text = transcriptionResult;
                segment.start = 0.0f;
                segment.end = totalDuration;
                segments.push_back (segment);
            }

            auto endTime = juce::Time::getMillisecondCounterHiRes();
            processingTimeSeconds.store ((endTime - startTime) / 1000.0);

            progress.store (100);
            return true;
        }
        catch (const std::exception& /*ex*/)
        {
            DBG ("Error during Python transcription");
            return false;
        }
    }

    int getProgress() const
    {
        return progress.load();
    }

    double getProcessingTime() const
    {
        return processingTimeSeconds.load();
    }

private:
    juce::File findParakeetExecutable()
    {
        // Try several locations for the parakeet transcription executable
        juce::StringArray executableNames;

        #ifdef _WIN32
        executableNames.add ("parakeet-transcribe-windows.exe");
        #elif __APPLE__
        executableNames.add ("parakeet-transcribe-macos");
        #else
        executableNames.add ("parakeet-transcribe-linux");
        #endif

        // Also try generic name as fallback
        executableNames.add ("parakeet-transcribe");

        // Search locations
        juce::Array<juce::File> searchPaths;

        // 1. Plugin's Resources directory (macOS bundle structure)
        auto pluginFile = juce::File::getSpecialLocation (juce::File::currentExecutableFile);
        #ifdef __APPLE__
        searchPaths.add (pluginFile.getParentDirectory().getParentDirectory().getChildFile ("Resources"));
        #endif

        // 2. Same directory as plugin
        searchPaths.add (pluginFile.getParentDirectory());

        // 3. Application data directory
        searchPaths.add (juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                        .getChildFile ("ReaSpeechLite"));

        // Try each location with each executable name
        for (const auto& searchPath : searchPaths)
        {
            for (const auto& exeName : executableNames)
            {
                auto exeFile = searchPath.getChildFile (exeName);
                if (exeFile.existsAsFile())
                {
                    DBG ("Found Parakeet executable: " + exeFile.getFullPathName());
                    return exeFile;
                }
            }
        }

        return {};
    }

    bool checkPythonAvailable()
    {
        // First, try to find the bundled executable
        auto executable = findParakeetExecutable();
        if (executable.existsAsFile())
        {
            parakeetExecutablePath = executable.getFullPathName();
            DBG ("Using bundled Parakeet executable: " + parakeetExecutablePath);
            return true;
        }

        // Fallback: Try to find Python executable (for development)
        juce::StringArray pythonCommands = { "python3", "python" };

        for (const auto& cmd : pythonCommands)
        {
            juce::ChildProcess process;
            if (process.start (cmd + " --version"))
            {
                process.waitForProcessToFinish (2000);
                pythonCommand = cmd;
                DBG ("Found Python: " + cmd);
                return true;
            }
        }

        DBG ("Neither bundled executable nor Python found");
        return false;
    }

    bool writeWavFile (const juce::File& file, const std::vector<float>& audioData, int sampleRate)
    {
        try
        {
            // Create an AudioBuffer from the float data
            juce::AudioBuffer<float> buffer (1, static_cast<int> (audioData.size()));
            buffer.copyFrom (0, 0, audioData.data(), static_cast<int> (audioData.size()));

            // Use JUCE's WavAudioFormat to write the file
            juce::WavAudioFormat wavFormat;
            auto outputStream = file.createOutputStream();

            if (outputStream == nullptr)
                return false;

            std::unique_ptr<juce::AudioFormatWriter> writer (
                wavFormat.createWriterFor (outputStream.release(),
                                          sampleRate,
                                          1, // num channels
                                          16, // bits per sample
                                          {}, // metadata
                                          0)); // quality option

            if (writer == nullptr)
                return false;

            return writer->writeFromAudioSampleBuffer (buffer, 0, buffer.getNumSamples());
        }
        catch (...)
        {
            return false;
        }
    }

    juce::String runPythonTranscription (const juce::String& audioFilePath, std::function<bool ()> isAborted)
    {
        juce::StringArray args;

        // Use bundled executable if available, otherwise fall back to Python
        if (parakeetExecutablePath.isNotEmpty())
        {
            args.add (parakeetExecutablePath);
            args.add (audioFilePath);
        }
        else
        {
            // Fallback: Use Python with inline script (for development)
            juce::String pythonScript = R"(
import sys
try:
    from onnx_asr import OnnxASR

    audio_file = sys.argv[1]

    # Initialize Parakeet TDT ASR
    asr = OnnxASR(model='parakeet_tdt_0.6b')

    # Transcribe
    result = asr.transcribe(audio_file)

    # Print result (just the text)
    if result and 'text' in result:
        print(result['text'])
    elif isinstance(result, str):
        print(result)
    else:
        print('')
except ImportError:
    print('ERROR: onnx-asr not installed. Install with: pip install onnx-asr', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'ERROR: {str(e)}', file=sys.stderr)
    sys.exit(1)
)";

            // Save script to temp file
            auto scriptFile = juce::File::getSpecialLocation (juce::File::tempDirectory)
                .getChildFile ("reaspeech_transcribe.py");

            if (! scriptFile.replaceWithText (pythonScript))
            {
                DBG ("Failed to write Python script");
                return {};
            }

            args.add (pythonCommand);
            args.add (scriptFile.getFullPathName());
            args.add (audioFilePath);
        }

        DBG ("Running: " + args.joinIntoString (" "));

        // Run the command
        juce::ChildProcess process;
        if (! process.start (args))
        {
            DBG ("Failed to start process");
            return {};
        }

        progress.store (50);

        // Wait for process to complete (with periodic abort checks)
        while (process.isRunning())
        {
            if (isAborted())
            {
                process.kill();
                return {};
            }
            juce::Thread::sleep (100);
        }

        progress.store (80);

        auto output = process.readAllProcessOutput();

        DBG ("Process output: " + output);

        // Check for errors
        if (output.startsWith ("ERROR:"))
        {
            DBG ("Process error: " + output);
            return {};
        }

        return output.trim();
    }

    std::string modelsDir;
    std::string lastModelName;
    juce::String pythonCommand = "python3";
    juce::String parakeetExecutablePath;
    std::atomic<int> progress { 0 };
    std::atomic<double> processingTimeSeconds { 0.0 };
};
