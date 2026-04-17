"""
端到端测 qwen3tts_fun 的批量合成分支。
- 7 条混合 queue: spk3 x4连续 → spk1 x2连续 → spk2 x1
- 断点续跑: 先跑一次, 再删一个产物, 第二次只该合成被删的那条
"""
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from videotrans.process.tts_fun import qwen3tts_fun  # noqa: E402

REF_DIR = ROOT / "output" / "马年逆袭：发小哭疯了-第2集-mp4"
OUT_DIR = ROOT / "tools" / "_qwen3_fun_test"
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

queue = []
def add(spk, text):
    queue.append({
        "text": text,
        "filename": str(OUT_DIR / f"{len(queue):02d}_{spk}.wav"),
        "ref_wav": str(REF_DIR / f"{spk}_ref.wav"),
        "role": "clone",
    })

# spk3 x4 连续 → 1 个 batch=4 调用
add("spk3", "第一句,老张你听着。")
add("spk3", "第二句,赶紧把钱还我。")
add("spk3", "第三句,再拖我就报警。")
add("spk3", "第四句,别怪我不讲情面。")
# spk1 x2 连续 → 1 个 batch=2 调用
add("spk1", "换人了,我是小李。")
add("spk1", "咱们改天再聊吧。")
# spk2 x1 → 1 个 batch=1 调用
add("spk2", "我就一句话,谢谢。")

queue_file = OUT_DIR / "_queue.json"
queue_file.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
log_file = OUT_DIR / "_logs.txt"

print(f"[probe] 第一次运行: 7 条 clone, 预期 3 次 batch 调用 (4+2+1)")
t0 = time.time()
ok, err = qwen3tts_fun(
    queue_tts_file=str(queue_file),
    language="Chinese",
    logs_file=str(log_file),
    defaulelang="zh",
    is_cuda=False,
    prompt=None,
    model_name="1.7B",
    roledict={},
)
print(f"[probe] ok={ok} 耗时={time.time()-t0:.1f}s")
if err:
    print(err)

print("\n[probe] 产物:")
for it in queue:
    p = Path(it['filename'] + "-qwen3tts.wav")
    print(f"  {p.name}: exists={p.exists()} size={p.stat().st_size if p.exists() else 0}")

print("\n[probe] 日志:")
print(log_file.read_text(encoding='utf-8'))

# 断点续跑: 删掉第 3 条, 再跑, 应该只合成那一条
to_del = Path(queue[2]['filename'] + "-qwen3tts.wav")
print(f"\n[probe] 删除 {to_del.name} 模拟失败, 再跑一次...")
to_del.unlink()

t0 = time.time()
ok, _ = qwen3tts_fun(
    queue_tts_file=str(queue_file),
    language="Chinese",
    logs_file=str(log_file),
    defaulelang="zh",
    is_cuda=False,
    prompt=None,
    model_name="1.7B",
    roledict={},
)
print(f"[probe] 续跑 ok={ok} 耗时={time.time()-t0:.1f}s (应远小于第一次)")
print(f"[probe] 被删的文件回来没: {to_del.exists()}")
