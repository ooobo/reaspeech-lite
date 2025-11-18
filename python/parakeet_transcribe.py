#!/usr/bin/env python3
"""
ReaSpeech Parakeet TDT Transcription Service
Standalone executable for ASR transcription using onnx-asr
"""
import sys
import os

# Filter out Python interpreter flags that might be incorrectly passed as arguments
# This fixes the "unrecognized arguments: -B -S -I -c" error on macOS
python_flags = {'-B', '-S', '-I', '-c', '-u', '-O', '-OO', '-d', '-E', '-s', '-v', '-W', '-X'}
sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg not in python_flags]

import argparse
import re
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

def split_into_sentences(text):
    """Split text into sentences based on punctuation."""
    if not text:
        return []

    # Split on sentence-ending punctuation followed by space or end of string
    # Keep the punctuation with the sentence
    sentences = re.split(r'([.!?]+(?:\s+|$))', text)

    # Recombine sentences with their punctuation
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = (sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')).strip()
        if sentence:
            result.append(sentence)

    # Handle any remaining text without punctuation
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())

    return result if result else [text.strip()]

def transcribe_with_chunking(asr, audio_path, chunk_duration=120.0):
    """
    Transcribe audio file with chunking for long files.

    Args:
        asr: ASR model instance
        audio_path: Path to audio file
        chunk_duration: Duration of each chunk in seconds

    Returns:
        List of sentence strings
    """
    # Load audio
    audio = load_audio(audio_path)
    duration = len(audio) / 16000.0

    # If audio is short, process directly without chunking
    if duration <= chunk_duration:
        result = asr.recognize(audio_path)
        text = extract_text(result)
        return split_into_sentences(text)

    # Process in chunks
    print(f"Processing {duration:.1f}s audio in chunks of {chunk_duration}s...", file=sys.stderr)
    chunks = chunk_audio(audio, chunk_duration=chunk_duration, overlap_duration=15.0)

    all_sentences = []
    for i, (chunk, start_time, end_time) in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({start_time:.1f}s - {end_time:.1f}s)...", file=sys.stderr)

        # Transcribe chunk
        result = asr.recognize(chunk, sample_rate=16000)
        text = extract_text(result)

        if text:
            sentences = split_into_sentences(text)

            # For overlapping regions, trim the last sentence of previous chunks
            # to avoid duplication (except for the last chunk)
            if i < len(chunks) - 1 and sentences:
                # Keep all but the last sentence to avoid overlap duplication
                all_sentences.extend(sentences[:-1])
            else:
                # Last chunk: keep all sentences
                all_sentences.extend(sentences)

    return all_sentences

def extract_text(result):
    """Extract text from recognition result."""
    if not result:
        return ""

    if isinstance(result, dict) and 'text' in result:
        return result['text']
    elif isinstance(result, str):
        return result
    elif isinstance(result, list):
        return ' '.join(str(r) if isinstance(r, str) else r.get('text', '') for r in result)
    else:
        return str(result)

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio using Parakeet TDT')
    parser.add_argument('audio_file', type=str, help='Path to audio file (WAV, 16kHz)')
    parser.add_argument('--model', type=str, default='nemo-parakeet-tdt-0.6b-v2',
                       help='Model name (default: nemo-parakeet-tdt-0.6b-v2)')
    parser.add_argument('--chunk-duration', type=float, default=120.0,
                       help='Chunk duration in seconds for long files (default: 120.0)')
    args = parser.parse_args()

    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        print(f"ERROR: Audio file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load ASR model (with progress bars disabled)
        asr = load_model(args.model)

        # Transcribe with chunking support
        sentences = transcribe_with_chunking(asr, str(audio_file), chunk_duration=args.chunk_duration)

        # Output one sentence per line
        for sentence in sentences:
            print(sentence)

    except Exception as e:
        print(f"ERROR: Transcription failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
