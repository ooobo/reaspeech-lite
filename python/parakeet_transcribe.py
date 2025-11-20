#!/usr/bin/env python3
"""
ReaSpeech Parakeet TDT Transcription Service
Standalone executable for ASR transcription using onnx-asr
"""
import sys
import os
import time

# Handle multiprocessing spawn on macOS
if '-c' in sys.argv:
    code_index = sys.argv.index('-c') + 1
    if code_index < len(sys.argv):
        exec(sys.argv[code_index])
    sys.exit(0)

import argparse
import re
import json
from pathlib import Path
import numpy as np

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from onnx_asr import load_model

def load_audio(audio_path):
    """Load audio file and return as numpy array at 16kHz."""
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if sr != 16000:
            ratio = 16000 / sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        return audio.astype(np.float32)
    except Exception as e:
        print(f"ERROR: Failed to load audio: {str(e)}", file=sys.stderr)
        sys.exit(1)

def chunk_audio_generator(audio_path, chunk_duration=120.0, overlap_duration=15.0, sample_rate=16000):
    """
    Generator that yields audio chunks on-demand to avoid loading entire file into memory.

    Args:
        audio_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds (default 120s)
        overlap_duration: Overlap between chunks in seconds (default 15s)
        sample_rate: Sample rate of audio (default 16000)

    Yields:
        (chunk_audio, start_time, end_time, chunk_index, total_chunks) tuples
    """
    import soundfile as sf

    # Get file info without loading the entire file
    info = sf.info(audio_path)
    total_samples = info.frames
    file_sr = info.samplerate
    total_duration = total_samples / file_sr

    # Calculate chunk parameters
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    stride = chunk_samples - overlap_samples

    # Calculate total number of chunks
    total_chunks = 0
    for start in range(0, total_samples, stride):
        total_chunks += 1
        if start + chunk_samples >= total_samples:
            break

    # Generate chunks on-demand
    chunk_index = 0
    for start in range(0, total_samples, stride):
        end = min(start + chunk_samples, total_samples)

        # Read only the chunk we need from the file
        with sf.SoundFile(audio_path) as f:
            f.seek(start)
            chunk = f.read(end - start)

            # Convert to mono if stereo
            if len(chunk.shape) > 1:
                chunk = chunk.mean(axis=1)

            # Resample if needed
            if file_sr != sample_rate:
                ratio = sample_rate / file_sr
                new_length = int(len(chunk) * ratio)
                indices = np.linspace(0, len(chunk) - 1, new_length)
                chunk = np.interp(indices, np.arange(len(chunk)), chunk)

            chunk = chunk.astype(np.float32)

        start_time = start / file_sr
        end_time = end / file_sr

        yield (chunk, start_time, end_time, chunk_index, total_chunks)

        chunk_index += 1
        if end >= total_samples:
            break

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
    sentence_end_puncts = {'.', '!', '?'}

    for i, (token, ts) in enumerate(zip(tokens, timestamps)):
        current_tokens.append(token)
        is_sentence_end = any(token.strip().endswith(p) for p in sentence_end_puncts)
        is_last_token = (i == len(tokens) - 1)

        if is_sentence_end or is_last_token:
            if i + 1 < len(timestamps):
                current_end = timestamps[i + 1]
            else:
                current_end = ts + 0.16

            text = ''.join(current_tokens).strip()
            if text:
                sentences.append({
                    'text': text,
                    'start': current_start,
                    'end': current_end
                })

            current_tokens = []
            if i + 1 < len(timestamps):
                current_start = timestamps[i + 1]

    return sentences

def transcribe_with_chunking(asr, audio_path, chunk_duration=120.0):
    """
    Transcribe audio file with chunking for long files, preserving timestamps.
    Memory-efficient: processes chunks on-demand without loading entire file.

    Args:
        asr: ASR model instance (with timestamps)
        audio_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds

    Returns:
        List of sentence dicts with {text, start, end}
    """
    import soundfile as sf

    # Get file duration without loading entire file
    info = sf.info(audio_path)
    duration = info.frames / info.samplerate

    if duration <= chunk_duration:
        result = asr.recognize(audio_path)
        if hasattr(result, 'tokens') and hasattr(result, 'timestamps'):
            return tokens_to_sentences(result.tokens, result.timestamps)
        else:
            text = result.text if hasattr(result, 'text') else str(result)
            return [{'text': text, 'start': 0.0, 'end': duration}]

    # Progress messages go to stderr - C++ code will filter and show in Reaper console
    print(f"Processing {duration:.1f}s audio in chunks of {chunk_duration}s...", file=sys.stderr)

    all_tokens = []
    all_timestamps = []

    # Process chunks on-demand using generator (memory-efficient)
    for chunk, chunk_start, chunk_end, chunk_idx, total_chunks in chunk_audio_generator(
        audio_path, chunk_duration=chunk_duration, overlap_duration=15.0
    ):
        # Progress messages go to stderr - C++ code will filter and show in Reaper console
        print(f"Processing chunk {chunk_idx+1}/{total_chunks} ({chunk_start:.1f}s - {chunk_end:.1f}s)...", file=sys.stderr)
        result = asr.recognize(chunk, sample_rate=16000)

        if hasattr(result, 'tokens') and hasattr(result, 'timestamps'):
            adjusted_timestamps = [ts + chunk_start for ts in result.timestamps]
            overlap_time = 15.0

            if chunk_idx > 0 and all_timestamps:
                last_prev_time = all_timestamps[-1]
                overlap_end = chunk_start + overlap_time

                filtered_tokens = []
                filtered_timestamps = []
                for token, ts in zip(result.tokens, adjusted_timestamps):
                    if ts >= overlap_end or ts > last_prev_time:
                        filtered_tokens.append(token)
                        filtered_timestamps.append(ts)

                all_tokens.extend(filtered_tokens)
                all_timestamps.extend(filtered_timestamps)
            else:
                all_tokens.extend(result.tokens)
                all_timestamps.extend(adjusted_timestamps)

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
        start_time = time.time()

        quantization = None if args.quantization.lower() == 'none' else args.quantization
        asr = load_model(args.model, quantization=quantization, providers=['CPUExecutionProvider']).with_timestamps()
        sentences = transcribe_with_chunking(asr, str(audio_file), chunk_duration=args.chunk_duration)

        for segment in sentences:
            print(json.dumps(segment))

        elapsed = time.time() - start_time
        print(f"Processing time: {elapsed:.2f}s", file=sys.stderr)

    except Exception as e:
        print(f"ERROR: Transcription failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
