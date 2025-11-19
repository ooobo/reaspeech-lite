#!/usr/bin/env python3
"""
ReaSpeech Parakeet TDT Transcription Service
Standalone executable for ASR transcription using onnx-asr
"""
import sys
import os

# Check if this is a multiprocessing spawn invocation
# On macOS, multiprocessing uses spawn mode which re-executes the script
# We need to detect this and allow it to proceed without our main() logic
if '-c' in sys.argv:
    # This is a multiprocessing spawn - let it execute as-is
    # Don't import our modules or run main()
    code_index = sys.argv.index('-c') + 1
    if code_index < len(sys.argv):
        exec(sys.argv[code_index])
    sys.exit(0)

import argparse
import re
import json
from pathlib import Path
import numpy as np

# Disable progress bars before importing huggingface_hub/onnx_asr
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# Import at module level so PyInstaller can detect it
from onnx_asr import load_model

def load_audio(audio_path):
    """Load audio file and return as numpy array at 16kHz."""
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            # Simple linear interpolation for resampling
            # (audio from C++ side is always 16kHz, but handle other sources)
            ratio = 16000 / sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        return audio.astype(np.float32)
    except Exception as e:
        print(f"ERROR: Failed to load audio: {str(e)}", file=sys.stderr)
        sys.exit(1)

def chunk_audio(audio, chunk_duration=120.0, overlap_duration=15.0, sample_rate=16000):
    """
    Split audio into overlapping chunks.

    Args:
        audio: Audio data as numpy array
        chunk_duration: Duration of each chunk in seconds (default 120s)
        overlap_duration: Overlap between chunks in seconds (default 15s)
        sample_rate: Sample rate of audio (default 16000)

    Returns:
        List of (chunk_audio, start_time, end_time) tuples
    """
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    stride = chunk_samples - overlap_samples

    chunks = []
    for start in range(0, len(audio), stride):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]

        # Calculate time offsets
        start_time = start / sample_rate
        end_time = end / sample_rate

        chunks.append((chunk, start_time, end_time))

        # Break if we've reached the end
        if end >= len(audio):
            break

    return chunks

def tokens_to_sentences(tokens, timestamps):
    """
    Group tokens into sentences based on punctuation.

    Args:
        tokens: List of token strings
        timestamps: List of timestamps (one per token, representing token start time)

    Returns:
        List of dicts with {text, start, end}
    """
    if not tokens or not timestamps:
        return []

    sentences = []
    current_tokens = []
    current_start = timestamps[0] if timestamps else 0.0

    # Sentence-ending punctuation
    sentence_end_puncts = {'.', '!', '?'}

    for i, (token, ts) in enumerate(zip(tokens, timestamps)):
        current_tokens.append(token)

        # Check if this token ends a sentence
        # Look for sentence-ending punctuation at the end of the token
        is_sentence_end = any(token.strip().endswith(p) for p in sentence_end_puncts)

        # Also end sentence at the last token
        is_last_token = (i == len(tokens) - 1)

        if is_sentence_end or is_last_token:
            # Calculate end time: use next token's timestamp, or estimate from current
            if i + 1 < len(timestamps):
                current_end = timestamps[i + 1]
            else:
                # For last token, estimate duration (average 0.16s per token)
                current_end = ts + 0.16

            text = ''.join(current_tokens).strip()
            if text:  # Only add non-empty sentences
                sentences.append({
                    'text': text,
                    'start': current_start,
                    'end': current_end
                })

            # Reset for next sentence
            current_tokens = []
            if i + 1 < len(timestamps):
                current_start = timestamps[i + 1]

    return sentences

def transcribe_with_chunking(asr, audio_path, chunk_duration=120.0):
    """
    Transcribe audio file with chunking for long files, preserving timestamps.

    Args:
        asr: ASR model instance (with timestamps)
        audio_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds

    Returns:
        List of sentence dicts with {text, start, end}
    """
    # Load audio
    audio = load_audio(audio_path)
    duration = len(audio) / 16000.0

    # If audio is short, process directly without chunking
    if duration <= chunk_duration:
        result = asr.recognize(audio_path)
        if hasattr(result, 'tokens') and hasattr(result, 'timestamps'):
            return tokens_to_sentences(result.tokens, result.timestamps)
        else:
            # Fallback: no timestamps available
            text = result.text if hasattr(result, 'text') else str(result)
            return [{'text': text, 'start': 0.0, 'end': duration}]

    # Process in chunks
    print(f"Processing {duration:.1f}s audio in chunks of {chunk_duration}s...", file=sys.stderr)
    chunks = chunk_audio(audio, chunk_duration=chunk_duration, overlap_duration=15.0)

    all_tokens = []
    all_timestamps = []

    for i, (chunk, chunk_start, chunk_end) in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({chunk_start:.1f}s - {chunk_end:.1f}s)...", file=sys.stderr)

        # Transcribe chunk
        result = asr.recognize(chunk, sample_rate=16000)

        if hasattr(result, 'tokens') and hasattr(result, 'timestamps'):
            # Adjust timestamps by chunk offset
            adjusted_timestamps = [ts + chunk_start for ts in result.timestamps]

            # For overlapping chunks, handle the overlap region
            # Keep tokens from overlap only if this is the last chunk or if tokens don't overlap
            overlap_time = 15.0  # seconds

            if i > 0 and all_timestamps:
                # Remove tokens from current chunk that fall in the overlap region
                # (keep tokens from the previous chunk for the overlap)
                last_prev_time = all_timestamps[-1]
                overlap_start = chunk_start
                overlap_end = chunk_start + overlap_time

                # Filter out tokens in the overlap region from current chunk
                filtered_tokens = []
                filtered_timestamps = []
                for token, ts in zip(result.tokens, adjusted_timestamps):
                    # Keep token if it's after the overlap or if there's no overlap
                    if ts >= overlap_end or ts > last_prev_time:
                        filtered_tokens.append(token)
                        filtered_timestamps.append(ts)

                all_tokens.extend(filtered_tokens)
                all_timestamps.extend(filtered_timestamps)
            else:
                # First chunk: keep all tokens
                all_tokens.extend(result.tokens)
                all_timestamps.extend(adjusted_timestamps)

    # Convert tokens to sentences
    return tokens_to_sentences(all_tokens, all_timestamps)

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio using Parakeet TDT')
    parser.add_argument('audio_file', type=str, help='Path to audio file (WAV, 16kHz)')
    parser.add_argument('--model', type=str, default='nemo-parakeet-tdt-0.6b-v2',
                       help='Model name (default: nemo-parakeet-tdt-0.6b-v2)')
    parser.add_argument('--chunk-duration', type=float, default=120.0,
                       help='Chunk duration in seconds for long files (default: 120.0)')
    parser.add_argument('--quantization', type=str, default='int8',
                       help='Model quantization (default: int8, options: int8, None)')
    args = parser.parse_args()

    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        print(f"ERROR: Audio file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load ASR model (with progress bars disabled)
        # Use CPU provider only - CoreML has compatibility issues with Parakeet TDT models
        # Enable timestamps for accurate segment timing
        # Handle "None" string input for quantization
        quantization = None if args.quantization.lower() == 'none' else args.quantization
        asr = load_model(args.model, quantization=quantization, providers=['CPUExecutionProvider']).with_timestamps()

        # Transcribe with chunking support
        sentences = transcribe_with_chunking(asr, str(audio_file), chunk_duration=args.chunk_duration)

        # Output as JSON (one object per line for each segment)
        for segment in sentences:
            print(json.dumps(segment))

    except Exception as e:
        print(f"ERROR: Transcription failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
