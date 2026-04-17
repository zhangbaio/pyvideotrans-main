# -*- coding: utf-8 -*-
"""
多集剧集声音克隆库 (L1 手动命名版)

目录布局:
    {ROOT_DIR}/voice_library/
    └── <drama_id>/
        ├── drama.json              # 角色索引
        └── <character_name>/
            ├── ref.wav             # 16kHz mono, 累计 <= 15s
            ├── ref_text.txt        # 参考文本 (可选, 拼接)
            └── meta.json           # 片段来源元信息

drama.json:
    {
      "drama_name": "<drama_id>",
      "characters": {
        "<name>": {
          "ref": "<name>/ref.wav",
          "total_sec": 14.3,
          "episodes": ["ep1", "ep2"]
        }
      }
    }

设计决策 (用户确认):
  1) drama_id 默认 = 视频父目录名
  2) ref.wav 累计上限 15s (Qwen3-TTS 克隆甜点区)
  3) 说话人分配 UI = 新名输入框 + 已有下拉
  4) 未命名说话人不入库 (走 round-robin 固定音色兜底)

依赖 ffmpeg (系统或 ROOT_DIR/ffmpeg 目录)。不依赖 pydub, 避免再拉一层。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from videotrans.configure.config import ROOT_DIR, logger
except Exception:  # 允许脱壳测试
    ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent)
    import logging
    logger = logging.getLogger("voice_library")

# ---------------------------------------------------------------- 常量

MAX_TOTAL_SEC = 15.0           # ref.wav 累计上限
MIN_SEG_SEC = 1.5              # 单片段最短时长 (过滤语气词)
TOP_N_SEGMENTS = 5             # 首次建库时取用片段数
SAMPLE_RATE = 16000            # Qwen3-TTS 克隆期望 16kHz

_SAFE_NAME = re.compile(r'[\\/:*?"<>|\s]+')


def _lib_root() -> Path:
    p = Path(ROOT_DIR) / "voice_library"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize(name: str) -> str:
    """把角色名 / drama 名清成安全目录名。保留中英文, 替换非法字符为下划线。"""
    name = (name or "").strip()
    if not name:
        return ""
    return _SAFE_NAME.sub("_", name)[:64]


def _ffmpeg_bin() -> str:
    local = Path(ROOT_DIR) / "ffmpeg" / ("ffmpeg.exe" if Path("/").exists() and (Path(ROOT_DIR) / "ffmpeg" / "ffmpeg.exe").exists() else "ffmpeg")
    return str(local) if local.exists() else "ffmpeg"


# ---------------------------------------------------------------- drama.json I/O

def get_drama_dir(video_path: str, override: str = "") -> Path:
    """返回 {lib}/<drama_id>/ 目录。自动创建。

    override 非空 → 直接用 (清洗后);
    否则取 Path(video_path).parent.name。
    """
    drama_id = _sanitize(override) or _sanitize(Path(video_path).parent.name) or "default"
    d = _lib_root() / drama_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _drama_json_path(drama_dir: Path) -> Path:
    return drama_dir / "drama.json"


def load_drama(drama_dir: Path) -> dict:
    p = _drama_json_path(drama_dir)
    if not p.exists():
        return {"drama_name": drama_dir.name, "characters": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[voice_library] drama.json 解析失败 {p}: {e}")
        return {"drama_name": drama_dir.name, "characters": {}}


def _save_drama(drama_dir: Path, data: dict) -> None:
    p = _drama_json_path(drama_dir)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_characters(drama_dir: Path) -> List[dict]:
    """返回 [{name, ref, total_sec, episodes}, ...] 按名字排序。"""
    data = load_drama(drama_dir)
    out = []
    for name, meta in data.get("characters", {}).items():
        out.append({
            "name": name,
            "ref": meta.get("ref", ""),
            "total_sec": float(meta.get("total_sec", 0.0)),
            "episodes": list(meta.get("episodes", [])),
        })
    out.sort(key=lambda x: x["name"])
    return out


# ---------------------------------------------------------------- ffmpeg 切/拼

def _probe_duration(wav_path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
            capture_output=True, text=True, timeout=15,
        )
        return float((r.stdout or "0").strip() or 0.0)
    except Exception:
        return 0.0


def _cut_segment(src_wav: str, start: float, end: float, dst_wav: Path) -> bool:
    """ffmpeg 切段, 重采样为 16kHz mono."""
    dur = max(0.0, end - start)
    if dur < 0.2:
        return False
    cmd = [
        _ffmpeg_bin(), "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src_wav,
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        str(dst_wav),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return dst_wav.exists() and dst_wav.stat().st_size > 0
    except Exception as e:
        logger.warning(f"[voice_library] 切段失败 {start}-{end}: {e}")
        return False


def _concat_wavs(pieces: List[Path], dst: Path) -> bool:
    if not pieces:
        return False
    if len(pieces) == 1:
        shutil.copyfile(pieces[0], dst)
        return True
    # ffmpeg concat demuxer
    list_file = dst.with_suffix(".txt")
    list_file.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in pieces),
        encoding="utf-8",
    )
    cmd = [
        _ffmpeg_bin(), "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return dst.exists()
    finally:
        try:
            list_file.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------- 对外 API

def get_ref_path(drama_dir: Path, character_name: str) -> Optional[Path]:
    """角色已入库 → 返回 ref.wav 绝对路径; 否则 None。"""
    name = _sanitize(character_name)
    if not name:
        return None
    data = load_drama(drama_dir)
    meta = data.get("characters", {}).get(name)
    if not meta:
        return None
    ref = drama_dir / meta.get("ref", "")
    return ref if ref.exists() else None


def add_or_extend_character(
    drama_dir: Path,
    character_name: str,
    source_wav: str,
    segments: List[Tuple[float, float]],
    episode_id: str,
    max_total_sec: float = MAX_TOTAL_SEC,
    ref_text: str = "",
) -> Optional[Path]:
    """把 segments 中的片段切出并拼到该角色 ref.wav 里, 不超上限。

    segments: [(start_sec, end_sec), ...]  建议外部已按 duration desc 排序。
    返回最终 ref.wav 路径; 无任何有效片段则 None。

    策略:
      - 已存在 ref.wav 且 total_sec >= max_total_sec: 直接跳过, 返回现有路径 (只追加 episode_id)
      - 否则: 按顺序切片段, 累计到达 max_total_sec 即停
      - 已有 ref.wav 时: 旧 ref 作为第一段, 再拼新片段 (保留历史克隆特征)
    """
    name = _sanitize(character_name)
    if not name:
        return None
    if not Path(source_wav).exists():
        logger.warning(f"[voice_library] 源音频不存在: {source_wav}")
        return None

    char_dir = drama_dir / name
    char_dir.mkdir(parents=True, exist_ok=True)
    ref_path = char_dir / "ref.wav"
    meta_path = char_dir / "meta.json"
    txt_path = char_dir / "ref_text.txt"

    data = load_drama(drama_dir)
    characters = data.setdefault("characters", {})
    char_meta = characters.setdefault(name, {
        "ref": f"{name}/ref.wav",
        "total_sec": 0.0,
        "episodes": [],
    })

    current_total = float(char_meta.get("total_sec", 0.0))
    if ref_path.exists() and current_total >= max_total_sec - 0.5:
        # 已满, 仅登记 episode
        if episode_id and episode_id not in char_meta["episodes"]:
            char_meta["episodes"].append(episode_id)
            _save_drama(drama_dir, data)
        return ref_path

    remaining = max(0.0, max_total_sec - current_total)
    if remaining < 0.5:
        return ref_path if ref_path.exists() else None

    tmp_dir = char_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)
    new_pieces: List[Path] = []
    used: List[Tuple[float, float]] = []
    acc = 0.0
    try:
        for i, (s, e) in enumerate(segments):
            if acc >= remaining - 0.2:
                break
            dur = max(0.0, e - s)
            if dur < MIN_SEG_SEC:
                continue
            # 截到不超 remaining
            take = min(dur, remaining - acc)
            piece = tmp_dir / f"seg_{i:03d}.wav"
            if _cut_segment(source_wav, s, s + take, piece):
                new_pieces.append(piece)
                used.append((s, s + take))
                acc += take

        if not new_pieces:
            return ref_path if ref_path.exists() else None

        # 拼接: 旧 ref + 新片段
        assembly: List[Path] = []
        if ref_path.exists():
            old_backup = tmp_dir / "_old_ref.wav"
            shutil.copyfile(ref_path, old_backup)
            assembly.append(old_backup)
        assembly.extend(new_pieces)

        if not _concat_wavs(assembly, ref_path):
            return ref_path if ref_path.exists() else None

        # 刷新时长 (以 ffprobe 为准)
        actual = _probe_duration(ref_path)
        char_meta["total_sec"] = round(actual or (current_total + acc), 2)
        if episode_id and episode_id not in char_meta["episodes"]:
            char_meta["episodes"].append(episode_id)

        # meta.json: 记录本次追加的片段
        meta_log: dict = {}
        if meta_path.exists():
            try:
                meta_log = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta_log = {}
        meta_log.setdefault("appends", []).append({
            "episode": episode_id,
            "source": source_wav,
            "segments": [{"start": s, "end": e} for s, e in used],
            "added_sec": round(acc, 2),
        })
        meta_path.write_text(json.dumps(meta_log, ensure_ascii=False, indent=2), encoding="utf-8")

        if ref_text:
            prev = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
            txt_path.write_text((prev + "\n" + ref_text).strip(), encoding="utf-8")

        _save_drama(drama_dir, data)
        logger.info(f"[voice_library] {drama_dir.name}/{name}: +{acc:.1f}s → total {char_meta['total_sec']}s")
        return ref_path
    finally:
        # 清理切片临时文件
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------- 片段挑选辅助

def set_embedding(drama_dir: Path, character_name: str, embedding: List[float]) -> None:
    """把角色声纹 (L2-normalized) 存进 drama.json。用于下集自动匹配。

    embedding 空列表 = 清空 (一般不需要)。角色不存在时自动建空条目。
    """
    name = _sanitize(character_name)
    if not name or not embedding:
        return
    data = load_drama(drama_dir)
    chars = data.setdefault("characters", {})
    meta = chars.setdefault(name, {
        "ref": "",
        "total_sec": 0.0,
        "episodes": [],
    })
    meta["embedding"] = list(embedding)
    _save_drama(drama_dir, data)


def list_embeddings(drama_dir: Path) -> List[Tuple[str, List[float]]]:
    """返回 [(name, embedding), ...], 仅含已录声纹的角色。供批量匹配用。"""
    data = load_drama(drama_dir)
    out = []
    for name, meta in data.get("characters", {}).items():
        emb = meta.get("embedding")
        if emb:
            out.append((name, list(emb)))
    return out


def match_character(
    drama_dir: Path,
    query_embedding: List[float],
    threshold: float = 0.70,
) -> Optional[Tuple[str, float]]:
    """在 drama.json 里找最相似的已录角色。命中返回 (name, score), 否则 None。"""
    try:
        from videotrans.util.speaker_embedding import match_best
    except Exception:
        return None
    candidates = list_embeddings(drama_dir)
    return match_best(query_embedding, candidates, threshold=threshold)


def get_fixed_voice(drama_dir: Path, character_name: str) -> Optional[str]:
    """读取角色绑定的固定音色 (跨集一致性)。无则 None。"""
    name = _sanitize(character_name)
    if not name:
        return None
    data = load_drama(drama_dir)
    meta = data.get("characters", {}).get(name)
    if not meta:
        return None
    voice = (meta.get("fixed_voice") or "").strip()
    return voice or None


def set_fixed_voice(drama_dir: Path, character_name: str, voice: str) -> None:
    """把角色 → 固定音色写进 drama.json。

    voice 为空/None → 清空该字段 (但保留角色条目, 不误删克隆 ref)。
    若角色在 drama.json 里不存在, 自动新建条目 (只有 fixed_voice, 无 ref)。
    """
    name = _sanitize(character_name)
    if not name:
        return
    data = load_drama(drama_dir)
    chars = data.setdefault("characters", {})
    meta = chars.setdefault(name, {
        "ref": "",
        "total_sec": 0.0,
        "episodes": [],
    })
    v = (voice or "").strip()
    if v:
        meta["fixed_voice"] = v
    else:
        meta.pop("fixed_voice", None)
    _save_drama(drama_dir, data)


def pick_top_segments(
    speaker_segments: List[Tuple[float, float]],
    top_n: int = TOP_N_SEGMENTS,
    min_sec: float = MIN_SEG_SEC,
) -> List[Tuple[float, float]]:
    """挑 top_n 个最长的合格片段, 按时长降序返回。供上游组装 add_or_extend_character。"""
    pool = [(s, e) for s, e in speaker_segments if (e - s) >= min_sec]
    pool.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    return pool[:top_n]
