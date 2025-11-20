#pragma once

#include <memory>
#include <string>
#include <vector>

#include <juce_core/juce_core.h>
#include <juce_audio_formats/juce_audio_formats.h>

#include "../utils/SafeUTF8.h"
#include "ASROptions.h"
#include "ASRSegment.h"

class OnnxPythonEngine
{
public:
    OnnxPythonEngine (const std::string& modelsDirIn) : modelsDir (modelsDirIn) {}

    ~OnnxPythonEngine() = default;

    bool downloadModel (const std::string& /*modelName*/, std::function<bool ()> /*isAborted*/)
    {
        if (! checkPythonAvailable())
            return false;

        progress.store (100);
        return true;
    }

    bool loadModel (const std::string& modelName)
    {
        if (! checkPythonAvailable())
            return false;

        lastModelName = modelName;
        return true;
    }

    bool transcribe (
        const std::vector<float>& audioData,
        ASROptions& /*options*/,
        std::vector<ASRSegment>& segments,
        std::function<bool ()> isAborted)
    {
        auto startTime = juce::Time::getMillisecondCounterHiRes();
        progress.store (0);

        auto updateProcessingTime = [&]() {
            auto endTime = juce::Time::getMillisecondCounterHiRes();
            processingTimeSeconds.store ((endTime - startTime) / 1000.0);
        };

        // Clear any existing segments from previous transcription
        segments.clear();

        try
        {
            float audioDuration = static_cast<float> (audioData.size()) / 16000.0f;
            juce::Logger::writeToLog ("Parakeet: Starting transcription for " +
                                     juce::String (audioDuration, 1) + "s audio");

            auto tempFile = juce::File::getSpecialLocation (juce::File::tempDirectory)
                .getChildFile ("reaspeech_temp_" + juce::String (juce::Random::getSystemRandom().nextInt()) + ".wav");

            if (! writeWavFile (tempFile, audioData, 16000))
            {
                juce::Logger::writeToLog ("Parakeet: Failed to write WAV file");
                updateProcessingTime();
                return false;
            }

            progress.store (20);

            if (isAborted())
            {
                tempFile.deleteFile();
                updateProcessingTime();
                return false;
            }

            juce::Logger::writeToLog ("Parakeet: Running transcription process...");
            auto transcriptionResult = runPythonTranscription (tempFile.getFullPathName(), isAborted);
            tempFile.deleteFile();

            if (transcriptionResult.isEmpty())
            {
                juce::Logger::writeToLog ("Parakeet: Transcription returned empty result");
                updateProcessingTime();
                return false;
            }

            juce::Logger::writeToLog ("Parakeet: Received " + juce::String (transcriptionResult.length()) +
                                     " bytes of output");
            progress.store (90);
            juce::StringArray lines;
            lines.addLines (transcriptionResult);

            for (const auto& line : lines)
            {
                auto trimmedLine = line.trim();
                if (trimmedLine.isEmpty())
                    continue;

                // Try to parse as JSON
                auto json = juce::JSON::parse (trimmedLine);
                if (! json.isObject())
                {
                    // Not JSON - this is a progress/debug message from stderr
                    // Use Logger instead of DBG so it works in Release builds
                    juce::Logger::writeToLog ("Parakeet: " + trimmedLine);
                    continue;
                }

                auto jsonObj = json.getDynamicObject();
                if (jsonObj == nullptr)
                {
                    juce::Logger::writeToLog ("Parakeet: Failed to parse JSON object: " + trimmedLine);
                    continue;
                }

                ASRSegment segment;
                segment.text = jsonObj->getProperty ("text").toString();
                segment.start = static_cast<float> (static_cast<double> (jsonObj->getProperty ("start")));
                segment.end = static_cast<float> (static_cast<double> (jsonObj->getProperty ("end")));

                if (! segment.text.isEmpty())
                    segments.push_back (segment);
            }

            if (segments.empty())
            {
                juce::Logger::writeToLog ("Parakeet: No segments parsed, using raw output as single segment");
                float totalDuration = static_cast<float> (audioData.size()) / 16000.0f;
                ASRSegment segment;
                segment.text = transcriptionResult;
                segment.start = 0.0f;
                segment.end = totalDuration;
                segments.push_back (segment);
            }

            juce::Logger::writeToLog ("Parakeet: Successfully parsed " + juce::String (segments.size()) + " segments");
            updateProcessingTime();
            progress.store (100);
            return true;
        }
        catch (...)
        {
            updateProcessingTime();
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
    juce::File findOnnxExecutable()
    {
        juce::StringArray executableNames;

        #ifdef _WIN32
        executableNames.add ("parakeet-transcribe-windows.exe");
        #elif __APPLE__
        executableNames.add ("parakeet-transcribe-macos");
        #else
        executableNames.add ("parakeet-transcribe-linux");
        #endif

        executableNames.add ("parakeet-transcribe");

        juce::Array<juce::File> searchPaths;
        auto pluginFile = juce::File::getSpecialLocation (juce::File::currentExecutableFile);

        #ifdef __APPLE__
        searchPaths.add (pluginFile.getParentDirectory().getParentDirectory().getChildFile ("Resources"));
        #endif

        searchPaths.add (pluginFile.getParentDirectory());
        searchPaths.add (juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                        .getChildFile ("ReaSpeechLite"));

        for (const auto& searchPath : searchPaths)
        {
            for (const auto& exeName : executableNames)
            {
                auto exeFile = searchPath.getChildFile (exeName);
                if (exeFile.existsAsFile())
                    return exeFile;
            }
        }

        return {};
    }

    bool checkPythonAvailable()
    {
        auto executable = findOnnxExecutable();
        if (executable.existsAsFile())
        {
            onnxExecutablePath = executable.getFullPathName();
            return true;
        }

        juce::StringArray pythonCommands = { "python3", "python" };
        for (const auto& cmd : pythonCommands)
        {
            juce::ChildProcess process;
            if (process.start (cmd + " --version"))
            {
                process.waitForProcessToFinish (2000);
                pythonCommand = cmd;
                return true;
            }
        }

        return false;
    }

    bool writeWavFile (const juce::File& file, const std::vector<float>& audioData, int sampleRate)
    {
        try
        {
            juce::AudioBuffer<float> buffer (1, static_cast<int> (audioData.size()));
            buffer.copyFrom (0, 0, audioData.data(), static_cast<int> (audioData.size()));

            juce::WavAudioFormat wavFormat;
            auto outputStream = file.createOutputStream();

            if (outputStream == nullptr)
                return false;

            std::unique_ptr<juce::AudioFormatWriter> writer (
                wavFormat.createWriterFor (outputStream.release(), sampleRate, 1, 16, {}, 0));

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

        juce::String modelForPython = lastModelName;
        if (modelForPython.startsWith ("onnx-"))
            modelForPython = modelForPython.substring (5);

        if (onnxExecutablePath.isNotEmpty())
        {
            args.add (onnxExecutablePath);
            args.add (audioFilePath);
            args.add ("--model");
            args.add (modelForPython);
        }
        else
        {
            juce::String pythonScript = R"(
import sys
try:
    from onnx_asr import OnnxASR
    audio_file = sys.argv[1]
    asr = OnnxASR(model='parakeet_tdt_0.6b')
    result = asr.transcribe(audio_file)
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

            // Save script to temp file with random name to avoid conflicts
            auto scriptFile = juce::File::getSpecialLocation (juce::File::tempDirectory)
                .getChildFile ("reaspeech_transcribe_" + juce::String (juce::Random::getSystemRandom().nextInt()) + ".py");

            if (! scriptFile.replaceWithText (pythonScript))
                return {};

            args.add (pythonCommand);
            args.add (scriptFile.getFullPathName());
            args.add (audioFilePath);
        }

        juce::ChildProcess process;
        if (! process.start (args))
            return {};

        progress.store (50);

        // Read output incrementally while process runs to avoid buffer overflow
        juce::String output;
        while (process.isRunning())
        {
            if (isAborted())
            {
                process.kill();
                return {};
            }

            // Read any available output to prevent buffer from filling up
            auto chunk = process.readAllProcessOutput();
            if (chunk.isNotEmpty())
                output += chunk;

            juce::Thread::sleep (100);
        }

        // Read any remaining output
        auto remaining = process.readAllProcessOutput();
        if (remaining.isNotEmpty())
            output += remaining;

        progress.store (80);

        // Check exit code
        auto exitCode = process.getExitCode();
        if (exitCode != 0)
        {
            juce::Logger::writeToLog ("Parakeet process exited with code: " + juce::String (exitCode));
            juce::Logger::writeToLog ("Output: " + output);
            return {};
        }

        // Check for errors
        if (output.startsWith ("ERROR:") || output.contains ("ERROR:"))
        {
            juce::Logger::writeToLog ("Parakeet error: " + output);
            return {};
        }

        return output.trim();
    }

    std::string modelsDir;
    std::string lastModelName;
    juce::String pythonCommand = "python3";
    juce::String onnxExecutablePath;
    std::atomic<int> progress { 0 };
    std::atomic<double> processingTimeSeconds { 0.0 };
};
