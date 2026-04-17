"""
Qwen3-TTS 本地 1.7B-Base 克隆烟雾测试
用法:  .venv/bin/python tools/probe_qwen3_local.py [cpu|mps]
默认 cpu (与 pyVideoTrans 线上行为一致); 传 mps 测 Apple Silicon 加速。
"""
import sys
import time
from pathlib import Path

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
REF_DIR = ROOT / "output" / "马年逆袭：发小哭疯了-第2集-mp4"
OUT_DIR = ROOT / "tools" / "_qwen3_local_test"
OUT_DIR.mkdir(exist_ok=True)

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
dtype = torch.float16 if device != "cpu" else torch.float32
print(f"[probe] device={device} dtype={dtype}")

t0 = time.time()
model = Qwen3TTSModel.from_pretrained(
    str(MODEL_DIR),
    device_map=device,
    dtype=dtype,
    attn_implementation=None,
)
print(f"[probe] model loaded in {time.time()-t0:.1f}s")

cases = [
    ("spk0", "这里是第一位说话人,测试声音克隆效果。"),
    ("spk1", "第二位,换个人说话。"),
]

for spk, text in cases:
    ref_wav = REF_DIR / f"{spk}_ref.wav"
    assert ref_wav.exists(), ref_wav
    t0 = time.time()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="Chinese",
        ref_audio=str(ref_wav),
        x_vector_only_mode=True,  # 无 ref_text 模式
    )
    elapsed = time.time() - t0
    out_path = OUT_DIR / f"{spk}_{device}.wav"
    sf.write(str(out_path), wavs[0], sr)
    audio_sec = len(wavs[0]) / sr
    print(f"[probe] {spk}: 合成 {audio_sec:.1f}s 音频 耗时 {elapsed:.1f}s "
          f"→ RTF={elapsed/audio_sec:.2f}x → {out_path.name}")

print("\n[probe] 完成")
