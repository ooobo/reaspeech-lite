# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for parakeet-transcribe executable

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Collect all data files from onnx_asr (includes .onnx files, vocab files, etc.)
onnx_asr_datas = collect_data_files('onnx_asr')

# Collect numpy dynamic libraries to fix circular import issues
numpy_binaries = collect_dynamic_libs('numpy')

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
    binaries=numpy_binaries,
    datas=onnx_asr_datas,
    hiddenimports=[
        '_posixsubprocess',  # Required for subprocess on Unix
        'pyexpat',  # Required for XML parsing
        'xml.parsers.expat',  # Required for plistlib on macOS
        'numpy.core._multiarray_umath',  # Fix numpy circular import
        'numpy.linalg._umath_linalg',  # Fix numpy linalg circular import
        'numpy.random._common',
        'numpy.random._bounded_integers',
        'numpy.random._mt19937',
        'numpy.random._philox',
        'numpy.random._pcg64',
        'numpy.random._sfc64',
        'numpy.random._generator',
    ],
    hookspath=[],
    hooksconfig={
        'gi': {
            'module-versions': {},
        }
    },
    runtime_hooks=[],
    excludes=[
        'setuptools',
        'distutils',
        'pkg_resources',
        'wheel',
        'pip',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out problematic runtime hooks that cause module loading issues
filtered_scripts = []
excluded_rthooks = ['pyi_rth_pkgres', 'pyi_rth_setuptools', 'pyi_rth_pkgutil', 'pyi_rth_multiprocessing']
for script in a.scripts:
    script_name = script[0] if isinstance(script, tuple) else str(script)
    if not any(rth in script_name for rth in excluded_rthooks):
        filtered_scripts.append(script)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    filtered_scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=f'parakeet-transcribe-{platform_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
