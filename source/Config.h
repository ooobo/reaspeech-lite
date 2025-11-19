#pragma once

#include <string>
#include <vector>

#include <juce_core/juce_core.h>

struct Config
{
    static inline const std::vector<std::pair<std::string, std::string>> models = {
        { "tdt-0.6b-v2", "Parakeet TDT" },
        { "small", "Whisper Small" },
        { "medium", "Whisper Medium" },
        { "large-v3", "Whisper Large" },
        { "large-v3-turbo", "Whisper Turbo" }
    };

    static const juce::URL getModelURL (std::string modelNameIn)
    {
        // Check if this is a Parakeet model
        if (isParakeetModel (modelNameIn))
        {
            // Parakeet models are handled differently (multiple files)
            return juce::URL ("");
        }

        return juce::URL ("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-" + modelNameIn + ".bin");
    }

    static const std::string getModelsDir()
    {
        const auto tempDir = juce::File::getSpecialLocation (juce::File::SpecialLocationType::tempDirectory);
        return tempDir.getFullPathName().toStdString() + "/models/";
    }

    static bool isParakeetModel (const std::string& modelName)
    {
        return modelName.find ("tdt-") != std::string::npos;
    }
};
