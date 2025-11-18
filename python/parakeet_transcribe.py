#!/usr/bin/env python3
"""
ReaSpeech Parakeet TDT Transcription Service
Standalone executable for ASR transcription using onnx-asr
"""
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio using Parakeet TDT')
    parser.add_argument('audio_file', type=str, help='Path to audio file (WAV, 16kHz)')
    parser.add_argument('--model', type=str, default='parakeet_tdt_0.6b',
                       help='Model name (default: parakeet_tdt_0.6b)')
    args = parser.parse_args()

    try:
        from onnx_asr import OnnxASR
    except ImportError:
        print("ERROR: onnx-asr not installed", file=sys.stderr)
        print("Install with: pip install onnx-asr", file=sys.stderr)
        sys.exit(1)

    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        print(f"ERROR: Audio file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize ASR
        asr = OnnxASR(model=args.model)

        # Transcribe
        result = asr.transcribe(str(audio_file))

        # Output result
        if result:
            if isinstance(result, dict) and 'text' in result:
                print(result['text'])
            elif isinstance(result, str):
                print(result)
            else:
                print("")
        else:
            print("")

    except Exception as e:
        print(f"ERROR: Transcription failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
