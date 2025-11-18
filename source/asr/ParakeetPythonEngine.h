#pragma once

#include <memory>
#include <string>
#include <vector>

#include <juce_core/juce_core.h>

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
    bool downloadModel (const std::string& modelName, std::function<bool ()> isAborted)
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

            // Parse the result
            ASRSegment segment;
            segment.text = transcriptionResult;
            segment.start = 0.0f;
            segment.end = static_cast<float> (audioData.size()) / 16000.0f;
            segments.push_back (segment);

            auto endTime = juce::Time::getMillisecondCounterHiRes();
            processingTimeSeconds.store ((endTime - startTime) / 1000.0);

            progress.store (100);
            return true;
        }
        catch (const std::exception& ex)
        {
            DBG ("Error during Python transcription: " + juce::String (ex.what()));
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
    bool checkPythonAvailable()
    {
        // Try to find Python executable
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

        DBG ("Python not found");
        return false;
    }

    bool writeWavFile (const juce::File& file, const std::vector<float>& audioData, int sampleRate)
    {
        try
        {
            auto outputStream = file.createOutputStream();
            if (outputStream == nullptr)
                return false;

            // Write WAV header
            const int numChannels = 1;
            const int bitsPerSample = 16;
            const int dataSize = static_cast<int> (audioData.size() * sizeof (int16_t));
            const int fileSize = 36 + dataSize;

            // RIFF header
            outputStream->write ("RIFF", 4);
            outputStream->writeIntLittleEndian (fileSize);
            outputStream->write ("WAVE", 4);

            // fmt chunk
            outputStream->write ("fmt ", 4);
            outputStream->writeIntLittleEndian (16); // chunk size
            outputStream->writeShortLittleEndian (1); // PCM format
            outputStream->writeShortLittleEndian (static_cast<short> (numChannels));
            outputStream->writeIntLittleEndian (sampleRate);
            outputStream->writeIntLittleEndian (sampleRate * numChannels * bitsPerSample / 8); // byte rate
            outputStream->writeShortLittleEndian (static_cast<short> (numChannels * bitsPerSample / 8)); // block align
            outputStream->writeShortLittleEndian (static_cast<short> (bitsPerSample));

            // data chunk
            outputStream->write ("data", 4);
            outputStream->writeIntLittleEndian (dataSize);

            // Write audio samples as 16-bit PCM
            for (float sample : audioData)
            {
                int16_t intSample = static_cast<int16_t> (juce::jlimit (-32768.0f, 32767.0f, sample * 32767.0f));
                outputStream->writeShortLittleEndian (intSample);
            }

            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    juce::String runPythonTranscription (const juce::String& audioFilePath, std::function<bool ()> isAborted)
    {
        // Create Python script inline
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

        // Run Python script
        juce::ChildProcess process;
        juce::String command = pythonCommand + " \"" + scriptFile.getFullPathName() + "\" \"" + audioFilePath + "\"";

        DBG ("Running: " + command);

        if (! process.start (command))
        {
            DBG ("Failed to start Python process");
            scriptFile.deleteFile();
            return {};
        }

        progress.store (50);

        // Wait for process to complete (with periodic abort checks)
        while (process.isRunning())
        {
            if (isAborted())
            {
                process.kill();
                scriptFile.deleteFile();
                return {};
            }
            juce::Thread::sleep (100);
        }

        progress.store (80);

        auto output = process.readAllProcessOutput();
        scriptFile.deleteFile();

        DBG ("Python output: " + output);

        // Check for errors
        if (output.startsWith ("ERROR:"))
        {
            DBG ("Python error: " + output);
            return {};
        }

        return output.trim();
    }

    std::string modelsDir;
    std::string lastModelName;
    juce::String pythonCommand = "python3";
    std::atomic<int> progress { 0 };
    std::atomic<double> processingTimeSeconds { 0.0 };
};
