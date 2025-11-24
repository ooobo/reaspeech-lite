#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_core/juce_core.h>

// Forward declaration to avoid circular dependency
class ReaSpeechLiteAudioSource;

struct ResamplingExporter
{
    static constexpr int blockSize = 4096;

    /**
     * Reads audio data from the given audio source, resamples it to the
     * specified destination sample rate, and stores the resampled audio data
     * in the provided buffer.
     *
     * @param audioSource The audio source to read audio data from.
     * @param destSampleRate The sample rate to which the audio data should be resampled.
     * @param channel The channel index to read from the audio source.
     * @param buffer A vector to store the resampled audio data.
     * @param isAborted Optional callback that returns true if the operation should be aborted.
     * @param logger Optional callback for logging messages.
     */
    static void exportAudio (juce::ARAAudioSource* audioSource,
        ARA::ARASampleRate destSampleRate,
        int channel,
        std::vector<float>& buffer,
        std::function<bool()> /* isAborted */ = nullptr,
        std::function<void(const juce::String&)> logger = nullptr)
    {
        const auto sourceChannelCount = audioSource->getChannelCount();
        jassert (channel >= 0 && channel < sourceChannelCount);
        juce::ignoreUnused (channel); // channel parameter kept for API compatibility

        const auto sourceSampleRate = audioSource->getSampleRate();
        const auto sourceSampleCount = audioSource->getSampleCount();

        auto log = [&logger] (const juce::String& msg) {
            if (logger)
                logger (msg);
            DBG (msg);  // Also log to debug console
        };

        // WORKAROUND FOR REAPER ARA BUG:
        // ARAAudioSourceReader::read() returns silence for successive reads after the first one.
        // This appears to be a REAPER ARA implementation bug.
        // Solution: Read all source audio into memory first, then resample manually.

        // NOTE: This loads entire audio file into memory before resampling.
        // Memory usage: ~1.3GB for 1-hour stereo file at 44.1kHz
        // Files are processed sequentially, so only one file is in memory at a time.
        juce::AudioBuffer<float> sourceBuffer (sourceChannelCount, static_cast<int>(sourceSampleCount));
        sourceBuffer.clear();

        // Try ARA read first (REAPER ARA bug workaround - may return silence)
        bool gotAudio = false;

        juce::ARAAudioSourceReader reader (audioSource);
        reader.read (&sourceBuffer, 0, static_cast<int>(sourceSampleCount), 0, true, true);

        // Check if we got any audio (quick check on first 1000 samples to avoid scanning huge files)
        float maxSample = 0.0f;
        int samplesToCheck = juce::jmin (1000, static_cast<int>(sourceSampleCount));
        for (int ch = 0; ch < sourceChannelCount && maxSample < 0.00001f; ++ch)
        {
            auto range = sourceBuffer.findMinMax (ch, 0, samplesToCheck);
            maxSample = juce::jmax (maxSample, juce::jmax (std::abs (range.getStart()), std::abs (range.getEnd())));
        }

        if (maxSample > 0.00001f) // Found audio
            gotAudio = true;

        // If ARA read failed, try reading directly from file (REAPER ARA bug workaround)
        if (!gotAudio)
        {
            log ("ResamplingExporter: WARNING - ARA read returned silence, trying direct file access...");

            // Try to get file path from ReaSpeechLiteAudioSource
            juce::String filePath;
            if (auto* rsAudioSource = dynamic_cast<ReaSpeechLiteAudioSource*>(audioSource))
                filePath = rsAudioSource->getFilePath();

            if (filePath.isNotEmpty())
            {
                juce::File audioFile (filePath);
                if (audioFile.existsAsFile())
                {
                    log ("ResamplingExporter: Reading from file: " + filePath);

                    juce::AudioFormatManager formatManager;
                    formatManager.registerBasicFormats();

                    std::unique_ptr<juce::AudioFormatReader> fileReader (formatManager.createReaderFor (audioFile));
                    if (fileReader != nullptr)
                    {
                        // Read directly from file
                        int samplesToRead = juce::jmin (static_cast<int>(sourceSampleCount),
                                                       static_cast<int>(fileReader->lengthInSamples));
                        fileReader->read (&sourceBuffer, 0, samplesToRead, 0, true, true);

                        // Check if file read succeeded (quick check on first 1000 samples)
                        float maxSample = 0.0f;
                        int samplesToCheck = juce::jmin (1000, samplesToRead);
                        for (int ch = 0; ch < sourceChannelCount && maxSample < 0.00001f; ++ch)
                        {
                            auto range = sourceBuffer.findMinMax (ch, 0, samplesToCheck);
                            maxSample = juce::jmax (maxSample, juce::jmax (std::abs (range.getStart()), std::abs (range.getEnd())));
                        }

                        if (maxSample > 0.00001f)
                            gotAudio = true;
                        else
                            log ("ResamplingExporter: File read also returned silence");
                    }
                    else
                    {
                        log ("ResamplingExporter: Failed to create reader for file");
                    }
                }
                else
                {
                    log ("ResamplingExporter: File does not exist: " + filePath);
                }
            }
            else
            {
                log ("ResamplingExporter: No file path available for direct read");
            }
        }

        // Calculate destination buffer size
        const double speedRatio = sourceSampleRate / destSampleRate; // Source samples per dest sample
        const auto destSampleCount = juce::roundToInt (sourceSampleCount * destSampleRate / sourceSampleRate);
        buffer.resize (static_cast<size_t>(destSampleCount));

        // Mix down to mono if needed and resample
        juce::LagrangeInterpolator interpolator;
        interpolator.reset();

        if (sourceChannelCount == 1)
        {
            // Mono source - direct resample
            interpolator.process (speedRatio,
                                 sourceBuffer.getReadPointer(0),
                                 buffer.data(),
                                 destSampleCount);
        }
        else
        {
            // Multi-channel - mix to mono first, then resample
            std::vector<float> monoSource (sourceSampleCount);

            for (int i = 0; i < static_cast<int>(sourceSampleCount); ++i)
            {
                float sample = 0.0f;
                for (int ch = 0; ch < sourceChannelCount; ++ch)
                    sample += sourceBuffer.getSample(ch, i);
                monoSource[static_cast<size_t>(i)] = sample / static_cast<float>(sourceChannelCount);
            }

            interpolator.process (speedRatio,
                                 monoSource.data(),
                                 buffer.data(),
                                 destSampleCount);
        }

        // Calculate max amplitude for diagnostics
        float maxAmplitude = 0.0f;
        for (const auto& sample : buffer)
            maxAmplitude = juce::jmax (maxAmplitude, std::abs (sample));

        log ("ResamplingExporter: Resampled to " + juce::String (destSampleCount) +
             " samples at " + juce::String (destSampleRate) + " Hz, max amplitude: " + juce::String (maxAmplitude, 6));
    }
};
