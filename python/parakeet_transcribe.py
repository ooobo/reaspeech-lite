#!/usr/bin/env python3
"""
ReaSpeech Parakeet TDT Transcription Service
Standalone executable for ASR transcription using onnx-asr
"""
import sys
import os
import argparse
import re
from pathlib import Path

# Disable progress bars before importing huggingface_hub/onnx_asr
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# Import at module level so PyInstaller can detect it
from onnx_asr import load_model

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

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio using Parakeet TDT')
    parser.add_argument('audio_file', type=str, help='Path to audio file (WAV, 16kHz)')
    parser.add_argument('--model', type=str, default='nemo-parakeet-tdt-0.6b-v2',
                       help='Model name (default: nemo-parakeet-tdt-0.6b-v2)')
    args = parser.parse_args()

    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        print(f"ERROR: Audio file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load ASR model (with progress bars disabled)
        asr = load_model(args.model)

        # Transcribe using recognize method
        result = asr.recognize(str(audio_file))

        # Extract text from result
        text = ""
        if result:
            if isinstance(result, dict) and 'text' in result:
                text = result['text']
            elif isinstance(result, str):
                text = result
            elif isinstance(result, list):
                # Handle list of results
                text = ' '.join(str(r) if isinstance(r, str) else r.get('text', '') for r in result)
            else:
                text = str(result)

        # Split into sentences and output one per line
        sentences = split_into_sentences(text)
        for sentence in sentences:
            print(sentence)

    except Exception as e:
        print(f"ERROR: Transcription failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
