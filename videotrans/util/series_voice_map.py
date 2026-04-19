# -*- coding: utf-8 -*-
"""Series-level speaker -> voice/ref persistence.

This module keeps voice assignment stable across a folder of episodes. It is a
small, file-backed layer around the existing per-episode speaker matching.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from videotrans.configure.config import logger, settings

MAP_FILENAME = "_voice_series_map.json"
DEFAULT_THRESHOLD = 0.72

_lock = threading.RLock()


def _engine_key(tts_type: int) -> str:
    try:
        return str(int(tts_type))
    except Exception:
        return str(tts_type or "")


def series_map_path(video_path: str, target_dir: str = "") -> Path:
    src_parent = Path(video_path).expanduser().resolve().parent
    primary = src_parent / MAP_FILENAME
    try:
        src_parent.mkdir(parents=True, exist_ok=True)
        return primary
    except Exception:
        pass
    if target_dir:
        return Path(target_dir).expanduser().resolve().parent / MAP_FILENAME
    return primary


def load_map(video_path: str, target_dir: str = "") -> dict:
    path = series_map_path(video_path, target_dir)
    if not path.exists():
        return {
            "version": 1,
            "source_dir": str(Path(video_path).expanduser().resolve().parent),
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "speakers": {},
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("map root is not object")
        data.setdefault("version", 1)
        data.setdefault("speakers", {})
        return data
    except Exception as e:
        logger.warning(f"[series_voice] failed to read {path}: {e}")
        return {
            "version": 1,
            "source_dir": str(Path(video_path).expanduser().resolve().parent),
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "speakers": {},
        }


def save_map(video_path: str, target_dir: str, data: dict) -> Path:
    path = series_map_path(video_path, target_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = int(time.time())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def _threshold() -> float:
    try:
        return float(settings.get("series_voice_match_threshold", DEFAULT_THRESHOLD) or DEFAULT_THRESHOLD)
    except Exception:
        return DEFAULT_THRESHOLD


def _iter_candidates(data: dict, tts_type: int):
    engine = _engine_key(tts_type)
    for gid, meta in (data.get("speakers") or {}).items():
        if not isinstance(meta, dict):
            continue
        if str(meta.get("tts_type", "")) != engine:
            continue
        emb = meta.get("embedding")
        if isinstance(emb, list) and emb:
            yield gid, meta, emb


def _compute_embedding(wav: str):
    try:
        from videotrans.util.speaker_embedding import compute_embedding
        return compute_embedding(wav)
    except Exception as e:
        logger.warning(f"[series_voice] speaker embedding unavailable: {e}")
        return None


def _best_global(data: dict, emb, tts_type: int) -> Tuple[Optional[str], float]:
    if not emb:
        return None, 0.0
    try:
        from videotrans.util.speaker_embedding import cosine_sim
    except Exception:
        return None, 0.0
    best_gid, best_score = None, -1.0
    for gid, _meta, candidate in _iter_candidates(data, tts_type):
        score = cosine_sim(emb, candidate)
        if score > best_score:
            best_gid, best_score = gid, score
    if best_gid and best_score >= _threshold():
        return best_gid, best_score
    return None, best_score


def _episode_id(video_path: str) -> str:
    return Path(video_path).name


def reuse_voice_mapping(
    *,
    video_path: str,
    target_dir: str,
    speaker_refs: Dict[str, str],
    tts_type: int,
) -> Tuple[Dict[str, str], Dict[str, dict]]:
    """Return local speaker voices already known from the series map."""
    if not speaker_refs:
        return {}, {}
    episode = _episode_id(video_path)
    with _lock:
        data = load_map(video_path, target_dir)
        mapping: Dict[str, str] = {}
        details: Dict[str, dict] = {}
        speakers = data.get("speakers") or {}
        for spk_id, wav in speaker_refs.items():
            # Exact alias from a previous run of the same episode.
            for gid, meta in speakers.items():
                aliases = ((meta or {}).get("episodes") or {}).get(episode, {})
                if isinstance(aliases, dict) and aliases.get(spk_id) and meta.get("voice"):
                    mapping[spk_id] = str(meta["voice"])
                    details[spk_id] = {
                        "voice": str(meta["voice"]),
                        "method": "series_alias",
                        "global_speaker": gid,
                        "score": 1.0,
                    }
                    break
            if spk_id in mapping:
                continue
            emb = _compute_embedding(wav)
            gid, score = _best_global(data, emb, tts_type)
            if not gid:
                continue
            meta = speakers.get(gid) or {}
            voice = str(meta.get("voice") or "")
            if not voice:
                continue
            mapping[spk_id] = voice
            details[spk_id] = {
                "voice": voice,
                "method": "series_embedding",
                "global_speaker": gid,
                "score": round(score, 4),
            }
        if mapping:
            logger.info(f"[series_voice] reused mapping for {episode}: {mapping}")
        return mapping, details


def update_voice_mapping(
    *,
    video_path: str,
    target_dir: str,
    speaker_refs: Dict[str, str],
    spk_voice_map: Dict[str, str],
    tts_type: int,
    details: Optional[Dict[str, dict]] = None,
) -> None:
    """Persist final local speaker -> voice mapping into the series map."""
    if not speaker_refs or not spk_voice_map:
        return
    episode = _episode_id(video_path)
    engine = _engine_key(tts_type)
    details = details if isinstance(details, dict) else {}
    with _lock:
        data = load_map(video_path, target_dir)
        speakers = data.setdefault("speakers", {})
        changed = False
        for spk_id, voice in spk_voice_map.items():
            if not voice or spk_id not in speaker_refs:
                continue
            wav = speaker_refs.get(spk_id)
            emb = _compute_embedding(wav) if wav else None
            gid = None
            detail = details.get(spk_id) if isinstance(details.get(spk_id), dict) else {}
            if detail.get("global_speaker") in speakers:
                gid = detail["global_speaker"]
            if gid is None and emb:
                gid, _score = _best_global(data, emb, tts_type)
            if gid is None:
                gid = f"global_spk_{len(speakers) + 1:03d}"
                while gid in speakers:
                    gid = f"global_spk_{len(speakers) + 1:03d}_{int(time.time())}"
                speakers[gid] = {
                    "voice": voice,
                    "tts_type": engine,
                    "embedding": emb or [],
                    "episodes": {},
                    "sample_count": 0,
                    "created_at": int(time.time()),
                }
            meta = speakers.setdefault(gid, {})
            meta.setdefault("tts_type", engine)
            if not meta.get("voice"):
                meta["voice"] = voice
            if emb and not meta.get("embedding"):
                meta["embedding"] = emb
            meta.setdefault("episodes", {}).setdefault(episode, {})[spk_id] = voice
            meta["sample_count"] = int(meta.get("sample_count") or 0) + 1
            meta["last_seen"] = int(time.time())
            changed = True
        if changed:
            path = save_map(video_path, target_dir, data)
            logger.info(f"[series_voice] saved {path}")


def reuse_clone_refs(
    *,
    video_path: str,
    target_dir: str,
    speaker_refs: Dict[str, dict],
    tts_type: int,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Reuse stable clone reference audio for known speakers."""
    if not speaker_refs:
        return {}, {}
    episode = _episode_id(video_path)
    out: Dict[str, dict] = {}
    details: Dict[str, dict] = {}
    with _lock:
        data = load_map(video_path, target_dir)
        speakers = data.get("speakers") or {}
        for spk_id, ref in speaker_refs.items():
            wav = (ref or {}).get("wav", "") if isinstance(ref, dict) else ""
            emb = _compute_embedding(wav) if wav else None
            gid, score = _best_global(data, emb, tts_type)
            if not gid:
                continue
            meta = speakers.get(gid) or {}
            clone_ref = meta.get("clone_ref") or {}
            ref_wav = clone_ref.get("wav") if isinstance(clone_ref, dict) else ""
            if not ref_wav or not Path(ref_wav).exists():
                continue
            reused = dict(ref or {})
            reused["wav"] = ref_wav
            if clone_ref.get("text"):
                reused["text"] = clone_ref.get("text")
            out[spk_id] = reused
            details[spk_id] = {
                "method": "series_clone_ref",
                "global_speaker": gid,
                "score": round(score, 4),
                "wav": ref_wav,
            }
        if out:
            logger.info(f"[series_voice] reused clone refs for {episode}: {list(out.keys())}")
        return out, details


def update_clone_refs(
    *,
    video_path: str,
    target_dir: str,
    speaker_refs: Dict[str, dict],
    tts_type: int,
) -> None:
    """Persist first clean clone ref for each global speaker."""
    if not speaker_refs:
        return
    episode = _episode_id(video_path)
    engine = _engine_key(tts_type)
    with _lock:
        data = load_map(video_path, target_dir)
        speakers = data.setdefault("speakers", {})
        changed = False
        for spk_id, ref in speaker_refs.items():
            if not isinstance(ref, dict):
                continue
            wav = ref.get("wav", "")
            if not wav or not Path(wav).exists():
                continue
            emb = _compute_embedding(wav)
            gid, _score = _best_global(data, emb, tts_type)
            if gid is None:
                gid = f"global_spk_{len(speakers) + 1:03d}"
                speakers[gid] = {
                    "voice": "clone",
                    "tts_type": engine,
                    "embedding": emb or [],
                    "episodes": {},
                    "sample_count": 0,
                    "created_at": int(time.time()),
                }
            meta = speakers.setdefault(gid, {})
            meta.setdefault("tts_type", engine)
            if emb and not meta.get("embedding"):
                meta["embedding"] = emb
            meta.setdefault("episodes", {}).setdefault(episode, {})[spk_id] = "clone"
            if not ((meta.get("clone_ref") or {}).get("wav")):
                meta["clone_ref"] = {
                    "wav": wav,
                    "text": ref.get("text", ""),
                    "episode": episode,
                    "speaker": spk_id,
                }
            meta["sample_count"] = int(meta.get("sample_count") or 0) + 1
            meta["last_seen"] = int(time.time())
            changed = True
        if changed:
            path = save_map(video_path, target_dir, data)
            logger.info(f"[series_voice] saved clone refs {path}")
