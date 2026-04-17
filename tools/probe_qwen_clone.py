"""
QwenttsCloneTTS 端到端烟雾测试
用法: .venv/bin/python tools/probe_qwen_clone.py

直接跑 3 条不同 speaker 的合成：
  - 上传 spkN_ref.wav 到 OSS
  - 调 DashScope create_voice
  - 合成一句中文, 落盘到 tools/_qwen_clone_test/
"""
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from videotrans.configure.config import params  # noqa: E402
from videotrans.tts._qwenttsclone import QwenttsCloneTTS  # noqa: E402

REF_DIR = ROOT / "output" / "马年逆袭：发小哭疯了-第2集-mp4"
OUT_DIR = ROOT / "tools" / "_qwen_clone_test"
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

# 模拟 Step 5 queue_tts: 3 条分属 spk0/spk1/spk2, 用真实 ref_wav
cases = [
    ("spk0", "这里是第一位说话人的测试。"),
    ("spk1", "第二位说话人来啦,声音克隆效果怎么样?"),
    ("spk2", "第三位,简短一句话。"),
]

queue_tts = []
for spk, text in cases:
    ref_wav = REF_DIR / f"{spk}_ref.wav"
    assert ref_wav.exists(), f"missing {ref_wav}"
    queue_tts.append({
        "text": text,
        "filename": str(OUT_DIR / f"{spk}_out.wav"),
        "ref_wav": str(ref_wav),
        "role": "clone",
    })

print(f"[probe] qwentts_key configured: {bool(params.get('qwentts_key'))}")
print(f"[probe] oss_bucket: {params.get('oss_bucket')}")
print(f"[probe] {len(queue_tts)} cases")

tts = QwenttsCloneTTS(
    queue_tts=queue_tts,
    language="zh-cn",
    uuid="probe-qwen-clone",
    play=False,
    is_test=False,
    tts_type=35,
    is_cuda=False,
)
tts.run()

print("\n[probe] 产物:")
for item in queue_tts:
    p = Path(item["filename"])
    print(f"  {p.name}: exists={p.exists()} size={p.stat().st_size if p.exists() else 0}")

print(f"\n[probe] voice_map: {tts._voice_map}")
print(f"[probe] voice_map 缓存路径: {tts._voice_map_path}")
