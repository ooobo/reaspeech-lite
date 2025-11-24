# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for parakeet-transcribe executable

import sys
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Collect all data files from onnx_asr (includes .onnx files, vocab files, etc.)
onnx_asr_datas = collect_data_files('onnx_asr')

# Determine platform name
if sys.platform.startswith('linux'):
    platform_name = 'linux'
elif sys.platform == 'darwin':
    platform_name = 'macos'
elif sys.platform == 'win32':
    platform_name = 'windows'
else:
    platform_name = 'unknown'

a = Analysis(
    ['parakeet_transcribe.py'],
    pathex=[],
    binaries=[],
    datas=onnx_asr_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=f'parakeet-transcribe-{platform_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
