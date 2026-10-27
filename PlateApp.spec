# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('models', 'models')]
binaries = [('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-console-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-console-l1-2-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-datetime-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-debug-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-errorhandling-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-fibers-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-file-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-file-l1-2-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-file-l2-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-handle-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-heap-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-interlocked-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-libraryloader-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-localization-l1-2-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-memory-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-namedpipe-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-processenvironment-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-processthreads-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-processthreads-l1-1-1.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-profile-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-rtlsupport-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-string-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-synch-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-synch-l1-2-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-sysinfo-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-timezone-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-core-util-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-conio-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-convert-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-environment-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-filesystem-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-heap-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-locale-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-math-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-multibyte-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-private-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-process-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-runtime-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-stdio-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-string-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-time-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/api-ms-win-crt-utility-l1-1-0.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/bzip2.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/concrt140.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/expat.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/ffi-7.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/ffi-8.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/ffi.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/libbz2.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/libcrypto-3-x64.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/libexpat.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/liblzma.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/libssl-3-x64.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/msvcp140.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/msvcp140_1.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/msvcp140_2.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/msvcp140_atomic_wait.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/msvcp140_codecvt_ids.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/sqlite3.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/tcl86t.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/tk86t.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/ucrtbase.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/vcamp140.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/vccorlib140.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/vcruntime140.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/vcruntime140_1.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/vcruntime140_threads.dll', '.'), ('C:/Users/19500/.conda/envs/plate/Library/bin/zlib.dll', '.')]
hiddenimports = ['imghdr', 'imgaug', 'lmdb', 'scipy', 'six', 'requests', 'tqdm', 'visualdl', 'openpyxl', 'pdf2docx', 'premailer', 'attrdict', 'fire', 'flask', 'babel', 'fontTools', 'matplotlib', 'imageio', 'shapely', 'pyclipper', 'rapidfuzz', 'skimage', 'yaml', 'lxml', 'bs4', 'cssutils', 'docx', 'fitz', 'contourpy', 'cycler', 'kiwisolver', 'dateutil', 'pytz', 'pandas', 'networkx', 'opt_einsum', 'decorator', 'psutil', 'termcolor', 'tifffile', 'lazy_loader', 'more_itertools', 'importlib_metadata', 'zipp', 'Crypto', 'crc32c', 'rarfile', 'PIL._tkinter_finder']
datas += copy_metadata('imageio')
datas += copy_metadata('imgaug')
datas += copy_metadata('paddleocr')
datas += copy_metadata('scipy')
hiddenimports += collect_submodules('skimage')
tmp_ret = collect_all('paddleocr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('paddle')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('imgaug')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('imageio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('lmdb')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyclipper')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('shapely')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['plate_recognition.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'torchtext', 'sympy', 'mpmath', 'fsspec', 'filelock', 'ultralytics', 'tqdm.notebook'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PlateApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PlateApp',
)
