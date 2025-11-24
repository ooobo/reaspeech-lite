# Parakeet Transcription Executable

This directory contains the Python script and build tools for creating standalone transcription executables.

## Building

### Requirements
- Python 3.8 or newer
- pip

### All Platforms

The build script will automatically install PyInstaller and dependencies:

**Linux/macOS:**
```bash
cd python
chmod +x build_executable.sh
./build_executable.sh
```

**Windows:**
```cmd
cd python
build_executable.bat
```

### Output

Executables are created in `dist/`:
- Linux: `parakeet-transcribe-linux`
- macOS: `parakeet-transcribe-macos`
- Windows: `parakeet-transcribe-windows.exe`

## Usage

```bash
# Basic usage
./parakeet-transcribe-linux /path/to/audio.wav

# Output is printed to stdout
# Errors are printed to stderr
Note, both are picked up by plugin because of juce quirk.
```

## Integration with VST3

The C++ plugin automatically:
1. Looks for the executable in the plugin's Resources directory
2. Falls back to looking in the same directory as the plugin
3. Exports audio to temporary WAV file
4. Calls the executable with the WAV file path
5. Reads transcription from stdout

## Distribution

If building seperately, copy the platform-specific executable to:
- **Windows**: Place `parakeet-transcribe-windows.exe` next to the VST3 plugin
- **macOS**: Place inside `ReaSpeechLite.vst3/Contents/Resources/`
- **Linux**: Place next to the VST3 plugin

The VST3 plugin will find and use it automatically.

