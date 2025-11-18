#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>

#include <juce_core/juce_core.h>
#include <onnxruntime_cxx_api.h>

#include "../utils/SafeUTF8.h"
#include "ASROptions.h"
#include "ASRSegment.h"

class ParakeetEngine
{
public:
    ParakeetEngine (const std::string& modelsDirIn) : modelsDir (modelsDirIn)
    {
        memoryInfo = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeCPU);
    }

    ~ParakeetEngine()
    {
        DBG ("ParakeetEngine destructor");
        preprocessor.reset();
        encoder.reset();
        decoderJoint.reset();
        downloadTask.reset();
    }

    // Download the model if needed. Returns true if successful or already downloaded.
    bool downloadModel (const std::string& modelName, std::function<bool ()> isAborted)
    {
        std::string modelDir = getModelDir (modelName);
        juce::File modelDirFile (modelDir);

        // Check if all required model files exist
        std::vector<juce::String> requiredFiles = {
            "nemo128.onnx",
            "encoder-model.onnx",
            "decoder_joint-model.onnx",
            "vocab.txt"
        };

        bool allFilesExist = true;
        for (const auto& filename : requiredFiles)
        {
            juce::String filePath = juce::String (modelDir) + "/" + filename;
            if (! juce::File (filePath).exists())
            {
                allFilesExist = false;
                break;
            }
        }

        if (allFilesExist)
        {
            DBG (juce::String ("Parakeet model already downloaded: ") + juce::String (modelDir));
            progress.store (100);
            return true;
        }

        modelDirFile.createDirectory();
        progress.store (0);

        DBG ("Downloading Parakeet model files");

        // Download each file
        std::map<juce::String, juce::String> fileUrls = {
            { "nemo128.onnx", "https://raw.githubusercontent.com/ZuoFuhong/subtitle/refs/heads/master/resources/model/parakeet-tdt-0.6b-v2/nemo128.onnx" },
            { "encoder-model.onnx", "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/encoder-model.onnx" },
            { "decoder_joint-model.onnx", "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/decoder_joint-model.onnx" },
            { "vocab.txt", "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/vocab.txt" }
        };

        int fileIndex = 0;
        for (const auto& [filename, url] : fileUrls)
        {
            if (isAborted())
                return false;

            juce::String filePath = juce::String (modelDir) + "/" + filename;
            if (juce::File (filePath).exists())
            {
                fileIndex++;
                progress.store (static_cast<int> (fileIndex * 100 / static_cast<int> (fileUrls.size())));
                continue;
            }

            DBG (juce::String ("Downloading: ") + filename);
            auto file = juce::File (filePath);

            downloadTask = juce::URL (url).downloadToFile (file, juce::URL::DownloadTaskOptions());

            while (downloadTask != nullptr && ! downloadTask->isFinished())
            {
                if (isAborted())
                {
                    DBG ("Download aborted");
                    downloadTask.reset();
                    progress.store (0);
                    file.deleteFile();
                    return false;
                }

                auto totalLength = downloadTask->getTotalLength();
                if (totalLength > 0)
                {
                    auto downloadedLength = downloadTask->getLengthDownloaded();
                    auto fileProgress = static_cast<int> ((downloadedLength * 100) / totalLength);
                    auto totalProgress = (fileIndex * 100 + fileProgress) / static_cast<int> (fileUrls.size());
                    progress.store (totalProgress);
                }

                juce::Thread::sleep (100);
            }

            if (downloadTask == nullptr || downloadTask->hadError())
            {
                DBG ("Failed to download: " + filename);
                downloadTask.reset();
                progress.store (0);
                file.deleteFile();
                return false;
            }

            downloadTask.reset();
            fileIndex++;
            DBG (juce::String ("Successfully downloaded: ") + filename);
        }

        progress.store (100);
        return true;
    }

    // Load the model by name. Returns true if successful.
    bool loadModel (const std::string& modelName)
    {
        DBG ("ParakeetEngine::loadModel: " + modelName);

        if (modelName == lastModelName && preprocessor != nullptr && encoder != nullptr && decoderJoint != nullptr)
        {
            DBG ("Model already loaded");
            return true;
        }

        std::string modelDir = getModelDir (modelName);
        DBG ("Loading model from: " + modelDir);

        // Check all required files exist
        std::vector<std::string> requiredFiles = {
            modelDir + "/nemo128.onnx",
            modelDir + "/encoder-model.onnx",
            modelDir + "/decoder_joint-model.onnx",
            modelDir + "/vocab.txt"
        };

        for (const auto& path : requiredFiles)
        {
            if (! juce::File (path).exists())
            {
                DBG ("Model file not found: " + path);
                return false;
            }
        }

        try
        {
            // Initialize ONNX Runtime environment and session options
            if (env == nullptr)
            {
                env = std::make_unique<Ort::Env> (ORT_LOGGING_LEVEL_WARNING, "ParakeetEngine");
            }

            Ort::SessionOptions sessionOptions;
            sessionOptions.SetIntraOpNumThreads (1);
            sessionOptions.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);

            // Load the three model files
            preprocessor = std::make_shared<Ort::Session> (*env, juce::String (modelDir + "/nemo128.onnx").toStdString().c_str(), sessionOptions);
            encoder = std::make_shared<Ort::Session> (*env, juce::String (modelDir + "/encoder-model.onnx").toStdString().c_str(), sessionOptions);
            decoderJoint = std::make_shared<Ort::Session> (*env, juce::String (modelDir + "/decoder_joint-model.onnx").toStdString().c_str(), sessionOptions);

            // Load vocabulary
            vocab = loadVocab (modelDir + "/vocab.txt");
            blankIdx = findBlankIdx ("<blk>", vocab);

            if (blankIdx < 0)
            {
                DBG ("Failed to find blank token in vocabulary");
                return false;
            }

            DBG ("Parakeet model loaded successfully");
            lastModelName = modelName;
            return true;
        }
        catch (const Ort::Exception& ex)
        {
            DBG ("ONNX Runtime error: " + juce::String (ex.what()));
            preprocessor.reset();
            encoder.reset();
            decoderJoint.reset();
            return false;
        }
    }

    // Transcribe the audio data. Returns true if successful.
    bool transcribe (
        const std::vector<float>& audioData,
        ASROptions& /*options*/,
        std::vector<ASRSegment>& segments,
        std::function<bool ()> isAborted)
    {
        DBG ("ParakeetEngine::transcribe");

        if (preprocessor == nullptr || encoder == nullptr || decoderJoint == nullptr)
        {
            DBG ("No model loaded");
            return false;
        }

        auto startTime = juce::Time::getMillisecondCounterHiRes();
        progress.store (0);

        try
        {
            if (isAborted())
                return false;

            // 1. Preprocessing
            auto [features, featuresLens] = preprocess (audioData);
            progress.store (20);

            if (isAborted())
                return false;

            // 2. Encoding
            auto [encoderOut, encoderOutLens] = encode (std::move (features), std::move (featuresLens));
            progress.store (40);

            if (isAborted())
                return false;

            // 3. Decoding
            auto tokens = decode (encoderOut, encoderOutLens, isAborted);
            progress.store (90);

            if (isAborted())
                return false;

            // 4. Convert tokens to text
            if (! tokens.empty())
            {
                ASRSegment segment;
                segment.text = tokensToText (tokens);
                segment.start = 0.0f;
                segment.end = static_cast<float> (audioData.size()) / 16000.0f; // Assuming 16kHz sample rate
                segments.push_back (segment);
            }

            auto endTime = juce::Time::getMillisecondCounterHiRes();
            processingTimeSeconds.store ((endTime - startTime) / 1000.0);

            progress.store (100);
            return true;
        }
        catch (const Ort::Exception& ex)
        {
            DBG ("ONNX Runtime error during transcription: " + juce::String (ex.what()));
            return false;
        }
    }

    // Get the full path to a model directory based on its name
    std::string getModelDir (const std::string& modelName) const
    {
        return modelsDir + "parakeet-" + modelName;
    }

    // Get current progress (0-100) of download or transcription
    int getProgress() const
    {
        return progress.load();
    }

    // Get processing time in seconds from last transcription
    double getProcessingTime() const
    {
        return processingTimeSeconds.load();
    }

private:
    // Load vocabulary from file
    std::map<int, std::string> loadVocab (const std::string& vocabPath)
    {
        std::map<int, std::string> result;
        std::ifstream infile (vocabPath);
        std::string line;

        while (std::getline (infile, line))
        {
            std::istringstream iss (line);
            std::string token;
            int id;

            if (iss >> token >> id)
            {
                // Replace Unicode character U+2581 (lower one eighth block) with space
                // Using hex escape sequence to avoid encoding issues
                const char unicodeChar[] = "\xe2\x96\x81"; // UTF-8 encoding of U+2581
                size_t pos;
                while ((pos = token.find (unicodeChar)) != std::string::npos)
                {
                    token.replace (pos, 3, " ");
                }
                result[id] = token;
            }
        }

        return result;
    }

    // Find blank token index in vocabulary
    int findBlankIdx (const std::string& token, const std::map<int, std::string>& vocabMap)
    {
        auto it = std::find_if (vocabMap.begin(), vocabMap.end(), [&token] (const auto& pair)
        {
            return pair.second == token;
        });

        if (it != vocabMap.end())
            return it->first;

        return -1;
    }

    // Preprocessing: Convert audio to features
    std::pair<Ort::Value, Ort::Value> preprocess (const std::vector<float>& audioWav)
    {
        auto waveformsSize = static_cast<int64_t> (audioWav.size());
        std::vector<int64_t> waveformsShape = { 1, waveformsSize };
        std::vector<int64_t> waveformsLensShape = { 1 };
        std::vector<int64_t> waveformsLens = { waveformsSize };

        Ort::Value inputWaveforms = Ort::Value::CreateTensor<float> (
            memoryInfo,
            const_cast<float*> (audioWav.data()),
            audioWav.size(),
            waveformsShape.data(),
            waveformsShape.size());

        Ort::Value inputWaveformsLens = Ort::Value::CreateTensor<int64_t> (
            memoryInfo,
            waveformsLens.data(),
            waveformsLens.size(),
            waveformsLensShape.data(),
            waveformsLensShape.size());

        std::vector<const char*> inputNames = { "waveforms", "waveforms_lens" };
        std::vector<const char*> outputNames = { "features", "features_lens" };
        std::array<Ort::Value, 2> inputTensors = { std::move (inputWaveforms), std::move (inputWaveformsLens) };

        auto outputTensors = preprocessor->Run (
            Ort::RunOptions { nullptr },
            inputNames.data(),
            inputTensors.data(),
            inputNames.size(),
            outputNames.data(),
            outputNames.size());

        return { std::move (outputTensors[0]), std::move (outputTensors[1]) };
    }

    // Encoder inference: convert features to high-dimensional representations
    std::pair<Ort::Value, Ort::Value> encode (Ort::Value features, Ort::Value featuresLens)
    {
        std::vector<const char*> inputNames = { "audio_signal", "length" };
        std::vector<const char*> outputNames = { "outputs", "encoded_lengths" };
        std::array<Ort::Value, 2> inputTensors = { std::move (features), std::move (featuresLens) };

        auto outputTensors = encoder->Run (
            Ort::RunOptions { nullptr },
            inputNames.data(),
            inputTensors.data(),
            inputNames.size(),
            outputNames.data(),
            outputNames.size());

        return { std::move (outputTensors[0]), std::move (outputTensors[1]) };
    }

    // Create initial decoder state
    std::pair<Ort::Value, Ort::Value> createState()
    {
        std::vector<int64_t> stateShape = { 2, 1, 640 };
        std::vector<float> stateData (1280, 0.0f);

        Ort::Value state1 = Ort::Value::CreateTensor<float> (
            memoryInfo,
            stateData.data(),
            1280,
            stateShape.data(),
            stateShape.size());

        Ort::Value state2 = Ort::Value::CreateTensor<float> (
            memoryInfo,
            stateData.data(),
            1280,
            stateShape.data(),
            stateShape.size());

        return { std::move (state1), std::move (state2) };
    }

    // Clone decoder state
    std::pair<Ort::Value, Ort::Value> cloneState (Ort::Value& state1, Ort::Value& state2)
    {
        std::vector<int64_t> stateShape = { 2, 1, 640 };

        auto* state1Data = state1.GetTensorMutableData<float>();
        auto* state2Data = state2.GetTensorMutableData<float>();

        std::vector<float> state1Copy (state1Data, state1Data + 1280);
        std::vector<float> state2Copy (state2Data, state2Data + 1280);

        Ort::Value newState1 = Ort::Value::CreateTensor<float> (
            memoryInfo,
            state1Copy.data(),
            1280,
            stateShape.data(),
            stateShape.size());

        Ort::Value newState2 = Ort::Value::CreateTensor<float> (
            memoryInfo,
            state2Copy.data(),
            1280,
            stateShape.data(),
            stateShape.size());

        return { std::move (newState1), std::move (newState2) };
    }

    // Find argmax of vector
    int argmax (const std::vector<float>& v)
    {
        return static_cast<int> (std::distance (v.begin(), std::max_element (v.begin(), v.end())));
    }

    // Decoder step
    std::tuple<std::vector<float>, int, Ort::Value, Ort::Value> decodeStep (
        const std::vector<int>& prevTokens,
        int64_t vocabSize,
        Ort::Value prevState1,
        Ort::Value prevState2,
        const std::vector<float>& encoderOut)
    {
        std::vector<int64_t> encoderOutShape = { 1, static_cast<int64_t> (encoderOut.size()), 1 };

        Ort::Value inputEncoderOutputs = Ort::Value::CreateTensor<float> (
            memoryInfo,
            const_cast<float*> (encoderOut.data()),
            encoderOut.size(),
            encoderOutShape.data(),
            encoderOutShape.size());

        int target = blankIdx;
        if (! prevTokens.empty())
            target = prevTokens.back();

        std::vector<int> targets = { target };
        std::vector<int64_t> targetShape = { 1, 1 };

        Ort::Value inputTargets = Ort::Value::CreateTensor<int> (
            memoryInfo,
            targets.data(),
            targets.size(),
            targetShape.data(),
            targetShape.size());

        std::vector<int32_t> targetLength = { 1 };
        std::vector<int64_t> targetLengthShape = { 1 };

        Ort::Value inputTargetLength = Ort::Value::CreateTensor<int32_t> (
            memoryInfo,
            targetLength.data(),
            targetLength.size(),
            targetLengthShape.data(),
            targetLengthShape.size());

        std::vector<const char*> inputNames = { "encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2" };
        std::vector<const char*> outputNames = { "outputs", "output_states_1", "output_states_2" };
        std::array<Ort::Value, 5> inputTensors = { std::move (inputEncoderOutputs), std::move (inputTargets), std::move (inputTargetLength), std::move (prevState1), std::move (prevState2) };

        auto outputTensors = decoderJoint->Run (
            Ort::RunOptions { nullptr },
            inputNames.data(),
            inputTensors.data(),
            inputNames.size(),
            outputNames.data(),
            outputNames.size());

        auto decoderOutShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        auto* decoderOutPtr = outputTensors[0].GetTensorMutableData<float>();

        int64_t decoderNum = 1;
        for (auto dim : decoderOutShape)
            decoderNum *= dim;

        std::vector<float> decoderOut (decoderOutPtr, decoderOutPtr + decoderNum);
        std::vector<float> output1 (decoderOut.begin(), decoderOut.begin() + vocabSize);
        std::vector<float> output2 (decoderOut.begin() + vocabSize, decoderOut.end());

        return { output1, argmax (output2), std::move (outputTensors[1]), std::move (outputTensors[2]) };
    }

    // Full decoding process
    std::vector<int> decode (Ort::Value& encoderOut, Ort::Value& encoderOutLens, std::function<bool ()> isAborted)
    {
        auto* encoderOutPtr = encoderOut.GetTensorMutableData<float>();
        auto* encoderOutLensPtr = encoderOutLens.GetTensorMutableData<int64_t>();

        int64_t encodingsLen = encoderOutLensPtr[0];
        int64_t vocabSize = static_cast<int64_t> (vocab.size());

        auto [prevState1, prevState2] = createState();
        std::vector<int> tokens;

        int maxTokensPerStep = 10;
        int t = 0;
        int emittedTokens = 0;

        while (t < encodingsLen)
        {
            if (isAborted())
                return tokens;

            // Extract encoding at time t
            std::vector<float> encodingT (1024);
            for (int i = 0; i < 1024; i++)
            {
                encodingT[i] = encoderOutPtr[i * encodingsLen + t];
            }

            auto [clonedState1, clonedState2] = cloneState (prevState1, prevState2);
            auto [probs, step, newState1, newState2] = decodeStep (tokens, vocabSize, std::move (clonedState1), std::move (clonedState2), encodingT);

            int token = argmax (probs);

            if (token != blankIdx)
            {
                prevState1 = std::move (newState1);
                prevState2 = std::move (newState2);
                tokens.push_back (token);
                emittedTokens += 1;
            }

            if (step >= 0)
            {
                t += step;
                emittedTokens = 0;
            }
            else if (token == blankIdx || emittedTokens == maxTokensPerStep)
            {
                t += 1;
                emittedTokens = 0;
            }

            // Update progress (40% to 90% is decoding)
            int decodeProgress = 40 + static_cast<int> ((50.0 * t) / encodingsLen);
            progress.store (decodeProgress);
        }

        return tokens;
    }

    // Decode space pattern (post-processing)
    std::string decodeSpacePattern (const std::string& tokensStr)
    {
        static const std::regex re1 (R"(^\s)");
        static const std::regex re2 (R"(\s(?!\b))");
        static const std::regex re3 (R"((\s)\b)");

        std::string result = tokensStr;
        result = std::regex_replace (result, re1, "");
        result = std::regex_replace (result, re2, "");
        result = std::regex_replace (result, re3, " ");

        return result;
    }

    // Convert tokens to text
    std::string tokensToText (const std::vector<int>& tokens)
    {
        std::string joined;
        for (int id : tokens)
        {
            auto it = vocab.find (id);
            if (it != vocab.end())
            {
                joined += it->second;
            }
        }

        return SafeUTF8::encode (decodeSpacePattern (joined).c_str()).toStdString();
    }

    std::string modelsDir;
    std::string lastModelName;
    std::unique_ptr<Ort::Env> env;
    std::shared_ptr<Ort::Session> preprocessor;
    std::shared_ptr<Ort::Session> encoder;
    std::shared_ptr<Ort::Session> decoderJoint;
    std::unique_ptr<juce::URL::DownloadTask> downloadTask;
    std::atomic<int> progress;
    std::atomic<double> processingTimeSeconds { 0.0 };
    Ort::MemoryInfo memoryInfo { nullptr };
    std::map<int, std::string> vocab;
    int blankIdx = -1;
};
