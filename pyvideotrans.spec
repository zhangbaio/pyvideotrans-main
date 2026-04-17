# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for pyVideoTrans (Windows).

构建 (在 Windows 执行):
    uv sync
    uv run pyinstaller pyvideotrans.spec --clean --noconfirm

产物: dist/pyVideoTrans/pyVideoTrans.exe (+ 同级资源目录)

设计 (Linus):
- onedir + contents-directory=. → 资源和 exe 同级, ROOT_DIR 直接可用
- 模型目录留空, 用户自己下载 (见 models/下载说明.txt)
- 用 collect_all 处理懒加载的大包, 不手动列 hiddenimports
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules
from pathlib import Path
import os

# ---------- 项目根 ----------
ROOT = Path(os.getcwd())

# ---------- 需要整包收集的"难搞"依赖 ----------
# 这些包大量用懒加载 / 动态 import / 非 .py 资源, 必须 collect_all
COLLECT_PKGS = [
    'torch', 'torchaudio',
    'transformers', 'tokenizers', 'safetensors', 'accelerate', 'peft',
    'faster_whisper', 'ctranslate2',
    'funasr', 'modelscope',
    'pyannote', 'pyannote.audio', 'pyannote.core', 'pyannote.database',
    'pyannote.metrics', 'pyannote.pipeline',
    'sherpa_onnx',
    'librosa', 'resampy', 'soxr', 'soundfile', 'audioread', 'pooch',
    'edge_tts', 'gtts',
    'piper', 'piper_tts',
    'qwen_tts',
    'qdarkstyle',
    'dashscope', 'openai', 'google', 'anthropic',
    'whisper',
    'diffusers',
    'hdbscan', 'pynndescent', 'numba', 'llvmlite',
    'hydra', 'omegaconf',
    'onnxruntime',
    'speechbrain',
    'srt', 'zhconv', 'jieba', 'sentencepiece',
    'elevenlabs', 'deepgram',
]

datas = []
binaries = []
hiddenimports = []
for pkg in COLLECT_PKGS:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f'[spec] skip collect_all({pkg}): {e}')

# ---------- 业务资源 (必须在 exe 同级) ----------
# 目标路径相对 contents-directory(='.'), 即 exe 所在目录
VT = 'videotrans'
datas += [
    (f'{VT}/cfg.json',      VT),
    (f'{VT}/codec.json',    VT),
    (f'{VT}/params.json',   VT),
    (f'{VT}/styles',        f'{VT}/styles'),
    (f'{VT}/language',      f'{VT}/language'),
    (f'{VT}/presets',       f'{VT}/presets'),
    (f'{VT}/voicejson',     f'{VT}/voicejson'),
    (f'{VT}/codes',         f'{VT}/codes'),
    (f'{VT}/prompts',       f'{VT}/prompts'),
    # ffmpeg 二进制
    ('ffmpeg',              'ffmpeg'),
]

# ---------- hiddenimports: 显式补几个 pyinstaller 偶尔漏的 ----------
hiddenimports += [
    'videotrans',
    'videotrans.configure.config',
    'videotrans.mainwin._main_win',
    'videotrans.mainwin._presets',
    'PySide6.QtSvg',
    'PySide6.QtNetwork',
    'PySide6.QtMultimedia',
    'PySide6.QtMultimediaWidgets',
    'sounddevice',
    'ten_vad',
    'pytsmod',
    'pyrubberband',
]

# ---------- 瘦身排除 ----------
excludes = [
    'tensorflow', 'tensorflow_gpu', 'keras',
    'jax', 'jaxlib',
    'PyQt5', 'PyQt6',
    'tkinter',
    'matplotlib.tests', 'numpy.tests', 'scipy.tests', 'pandas.tests',
    'notebook', 'jupyter', 'ipykernel', 'IPython.tests',
]

block_cipher = None

a = Analysis(
    ['sp.py'],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pyVideoTrans',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                # upx 压缩 torch DLL 风险大, 关闭
    console=False,            # GUI 模式, 不弹黑框
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='videotrans/styles/icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='pyVideoTrans',
)
