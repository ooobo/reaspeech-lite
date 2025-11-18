# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for parakeet-transcribe executable

import sys
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

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
    datas=[],
    hiddenimports=[
        'onnx_asr',
        'onnxruntime',
        'numpy',
        'librosa',
        'soundfile',
        'scipy',
        'sklearn',
        'numba',
        'resampy',
        'audioread',
        'decorator',
        'joblib',
        'lazy_loader',
        'msgpack',
        'pooch',
        'soxr',
        'typing_extensions',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Collect all from onnx-asr
a.datas += collect_all('onnx_asr')[0]
a.binaries += collect_all('onnx_asr')[1]
a.hiddenimports += collect_all('onnx_asr')[2]

# Collect all from onnxruntime
a.datas += collect_all('onnxruntime')[0]
a.binaries += collect_all('onnxruntime')[1]
a.hiddenimports += collect_all('onnxruntime')[2]

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
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
