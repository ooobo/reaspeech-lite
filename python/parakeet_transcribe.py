#!/usr/bin/env python3
"""
ReaSpeech Parakeet TDT Transcription Service
Standalone executable for ASR transcription using onnx-asr
"""
import sys
import argparse
from pathlib import Path

# Import at module level so PyInstaller can detect it
from onnx_asr import load_model

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
        # Load ASR model
        asr = load_model(args.model)

        # Transcribe
        result = asr(str(audio_file))

        # Output result
        if result:
            if isinstance(result, dict) and 'text' in result:
                print(result['text'])
            elif isinstance(result, str):
                print(result)
            elif isinstance(result, list):
                # Handle list of results
                text = ' '.join(str(r) if isinstance(r, str) else r.get('text', '') for r in result)
                print(text)
            else:
                print(str(result))
        else:
            print("")

    except Exception as e:
        print(f"ERROR: Transcription failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
