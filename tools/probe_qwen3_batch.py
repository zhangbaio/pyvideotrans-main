"""
验证同 speaker batch 合成 vs 逐条合成 的加速比。
"""
import time
from pathlib import Path

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
REF_DIR = ROOT / "output" / "马年逆袭：发小哭疯了-第2集-mp4"
OUT_DIR = ROOT / "tools" / "_qwen3_batch_test"
OUT_DIR.mkdir(exist_ok=True)

print("[probe] loading model on MPS...")
model = Qwen3TTSModel.from_pretrained(
    str(MODEL_DIR), device_map="mps", dtype=torch.float16, attn_implementation=None,
)

ref_wav = str(REF_DIR / "spk3_ref.wav")
texts = [
    "老张啊,你这小子真是越来越不像话了。",
    "昨天的事情我还没跟你算账呢。",
    "赶紧把钱还给我,不然别怪我翻脸。",
    "我跟你说,再不还就报警了啊。",
]

# 预热: speaker embedding + model 首次前向
print("[warmup] 冷启动一次")
items = model.create_voice_clone_prompt(ref_audio=ref_wav, x_vector_only_mode=True)
prompt_item = items[0]
_ = model.generate_voice_clone(text="热身.", language="Chinese",
                               voice_clone_prompt=[prompt_item])

# A) 逐条
print("\n=== A: 逐条合成 4 句 ===")
t0 = time.time()
total_audio = 0.0
for i, text in enumerate(texts):
    wavs, sr = model.generate_voice_clone(
        text=text, language="Chinese", voice_clone_prompt=[prompt_item],
    )
    total_audio += len(wavs[0]) / sr
    sf.write(str(OUT_DIR / f"seq_{i}.wav"), wavs[0], sr)
elapsed_seq = time.time() - t0
print(f"  总耗时 {elapsed_seq:.2f}s, 音频总长 {total_audio:.2f}s, RTF={elapsed_seq/total_audio:.2f}x")

# B) batch=4
print("\n=== B: 一次 batch 4 句 ===")
t0 = time.time()
wavs, sr = model.generate_voice_clone(
    text=texts,
    language="Chinese",
    voice_clone_prompt=[prompt_item] * len(texts),
)
elapsed_batch = time.time() - t0
total_audio_b = sum(len(w) / sr for w in wavs)
for i, w in enumerate(wavs):
    sf.write(str(OUT_DIR / f"batch_{i}.wav"), w, sr)
print(f"  总耗时 {elapsed_batch:.2f}s, 音频总长 {total_audio_b:.2f}s, RTF={elapsed_batch/total_audio_b:.2f}x")

print("\n=== 结论 ===")
speedup = elapsed_seq / elapsed_batch
print(f"逐条: {elapsed_seq:.1f}s | batch4: {elapsed_batch:.1f}s | 加速 {speedup:.2f}x")
if speedup > 1.3:
    print("✅ batch 有收益, 值得实施")
else:
    print("❌ batch 无明显收益, 不值得改动")
