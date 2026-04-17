"""
验证 speaker embedding 缓存优化: 同 speaker 连续 4 句, 对比 baseline vs cache.
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
OUT_DIR = ROOT / "tools" / "_qwen3_cache_test"
OUT_DIR.mkdir(exist_ok=True)

device = "mps"
print(f"[probe] device={device}")
t0 = time.time()
model = Qwen3TTSModel.from_pretrained(
    str(MODEL_DIR), device_map=device, dtype=torch.float16, attn_implementation=None,
)
print(f"[probe] model loaded in {time.time()-t0:.1f}s")

# 4 句同一说话人 spk3 (短剧里出现最多的那位)
ref_wav = str(REF_DIR / "spk3_ref.wav")
texts = [
    "老张啊,你这小子真是越来越不像话了。",
    "昨天的事情我还没跟你算账呢。",
    "赶紧把钱还给我,不然别怪我翻脸。",
    "我跟你说,再不还就报警了啊。",
]

def run_round(tag, use_cache):
    print(f"\n=== {tag} ===")
    total_audio, total_elapsed = 0.0, 0.0
    cached_item = None
    for i, text in enumerate(texts):
        t0 = time.time()
        if use_cache and cached_item is not None:
            wavs, sr = model.generate_voice_clone(
                text=text, language="Chinese",
                voice_clone_prompt=[cached_item],
            )
        else:
            if use_cache:
                items = model.create_voice_clone_prompt(
                    ref_audio=ref_wav, x_vector_only_mode=True,
                )
                cached_item = items[0]
                wavs, sr = model.generate_voice_clone(
                    text=text, language="Chinese",
                    voice_clone_prompt=[cached_item],
                )
            else:
                wavs, sr = model.generate_voice_clone(
                    text=text, language="Chinese",
                    ref_audio=ref_wav, x_vector_only_mode=True,
                )
        elapsed = time.time() - t0
        audio_sec = len(wavs[0]) / sr
        total_audio += audio_sec
        total_elapsed += elapsed
        sf.write(str(OUT_DIR / f"{tag}_{i}.wav"), wavs[0], sr)
        print(f"  [{i}] {elapsed:.2f}s 推理 / {audio_sec:.2f}s 音频 (RTF={elapsed/audio_sec:.2f}x)")
    print(f"  小计: {total_elapsed:.2f}s / {total_audio:.2f}s 平均 RTF={total_elapsed/total_audio:.2f}x")
    return total_elapsed, total_audio

# 预热: 先跑一次丢掉, 避免冷启动偏差影响 baseline
print("\n[warmup]")
_ = model.generate_voice_clone(text="热身句.", language="Chinese",
                               ref_audio=ref_wav, x_vector_only_mode=True)

b_elapsed, b_audio = run_round("baseline", use_cache=False)
c_elapsed, c_audio = run_round("cached",   use_cache=True)

print(f"\n=== 结论 ===")
print(f"baseline: {b_elapsed:.2f}s")
print(f"cached:   {c_elapsed:.2f}s")
saved = b_elapsed - c_elapsed
print(f"省掉:     {saved:.2f}s ({saved/b_elapsed*100:.1f}%)")
