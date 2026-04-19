# -*- coding: utf-8 -*-
"""
说话人 -> TTS 音色自动匹配（三层降级）

优先级:
  1. 指纹库存在: 源 spk embedding vs 音色 embedding, 余弦相似度 top1
  2. 元数据/标签: F0 判源性别 + 音色 gender 标签过滤 + round-robin
  3. 纯 round-robin 兜底

对外主接口:
  - match_voices_to_speakers(): 兼容旧调用, 仅返回映射
  - match_voices_to_speakers_verbose(): 返回映射 + 匹配依据
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from videotrans.configure.config import ROOT_DIR, logger
from videotrans.util.f0_gender import detect_gender_and_f0_from_wav
from videotrans.util.voice_tagger import tag_voice

_ENGINE_KEY_MAP = {
    0: "edgetts",
    1: "qwen3local",
    5: "qwentts",
    6: "doubao2",
    7: "doubao0",
    18: "azure",
    28: "freeazure",
    29: "googletts",
}

_AUTO_SKIP_VOICES = frozenset({"No", "clone", "auto-match", ""})


def _engine_key(tts_type: int) -> str:
    return _ENGINE_KEY_MAP.get(int(tts_type), "unknown")


def _normalize_voice_name(name: str) -> str:
    return "".join(ch.lower() for ch in (name or "") if ch.isalnum())


def _resolve_fp_meta(fp: Dict[str, dict], voice_name: str) -> dict:
    if not fp or not voice_name:
        return {}
    if voice_name in fp and isinstance(fp[voice_name], dict):
        return fp[voice_name]
    normalized = _normalize_voice_name(voice_name)
    if not normalized:
        return {}
    for key, value in fp.items():
        if _normalize_voice_name(key) == normalized and isinstance(value, dict):
            return value
    return {}


def _load_fingerprint(engine_key: str) -> Dict[str, dict]:
    candidates = [Path(ROOT_DIR) / "videotrans" / "voicejson" / f"{engine_key}_emb.json"]
    if engine_key == "qwen3local":
        candidates.append(Path(ROOT_DIR) / "videotrans" / "voicejson" / "qwentts_emb.json")
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"[voice_matcher] 读取指纹库失败 {path}: {e}")
    return {}


def _candidate_voices(all_voices: List[str]) -> List[str]:
    return [v for v in all_voices if v not in _AUTO_SKIP_VOICES]


def _voice_gender(engine_key: str, voice_name: str) -> str:
    fp = _load_fingerprint(engine_key)
    meta = _resolve_fp_meta(fp, voice_name)
    gender = meta.get("gender") if isinstance(meta, dict) else None
    if gender in ("f", "m", "any"):
        return gender
    return tag_voice(voice_name)["gender"]


def _tag_summary_by_engine(voices: List[str], engine_key: str) -> Dict[str, int]:
    counter: Dict[str, int] = {"f": 0, "m": 0, "any": 0}
    for voice_name in voices:
        gender = _voice_gender(engine_key, voice_name)
        counter[gender] = counter.get(gender, 0) + 1
    return counter


def _filter_by_gender_engine(voices: List[str], target_gender: str, engine_key: str) -> List[str]:
    if not target_gender or target_gender == "any" or not voices:
        return list(voices)
    out = []
    for voice_name in voices:
        gender = _voice_gender(engine_key, voice_name)
        if gender == target_gender or gender == "any":
            out.append(voice_name)
    return out if out else list(voices)


def match_voices_to_speakers(
    *,
    speaker_refs: Dict[str, str],
    all_voices: List[str],
    tts_type: int,
    existing: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    return match_voices_to_speakers_verbose(
        speaker_refs=speaker_refs,
        all_voices=all_voices,
        tts_type=tts_type,
        existing=existing,
    )["mapping"]


def match_voices_to_speakers_verbose(
    *,
    speaker_refs: Dict[str, str],
    all_voices: List[str],
    tts_type: int,
    existing: Optional[Dict[str, str]] = None,
) -> Dict[str, dict]:
    existing = existing or {}
    pool = _candidate_voices(all_voices)
    if not pool:
        logger.warning("[voice_matcher] 音色池为空, 无法分配")
        return {"mapping": dict(existing), "details": {}}

    spk_ids = list(speaker_refs.keys())
    result: Dict[str, str] = dict(existing)
    details: Dict[str, dict] = {}
    for spk_id, voice_name in existing.items():
        details[spk_id] = {"voice": voice_name, "method": "existing", "score": None}

    if _try_embedding_match(result, details, speaker_refs, pool, tts_type):
        _fill_missing_with_gender(result, details, speaker_refs, pool, spk_ids, tts_type)
        logger.info(f"[voice_matcher] 层1 指纹匹配: {result}")
        return {"mapping": result, "details": details}

    if _try_gender_match(result, details, speaker_refs, pool, spk_ids, tts_type):
        logger.info(f"[voice_matcher] 层2 性别匹配: {result}")
        return {"mapping": result, "details": details}

    _fill_round_robin(result, details, spk_ids, pool)
    logger.info(f"[voice_matcher] 层3 round-robin: {result}")
    return {"mapping": result, "details": details}


def _try_embedding_match(
    result: Dict[str, str],
    details: Dict[str, dict],
    speaker_refs: Dict[str, str],
    pool: List[str],
    tts_type: int,
) -> bool:
    engine_key = _engine_key(tts_type)
    fp = _load_fingerprint(engine_key)
    if not fp:
        return False

    valid_fp = {}
    for voice_name in pool:
        meta = _resolve_fp_meta(fp, voice_name)
        if meta.get("embedding"):
            valid_fp[voice_name] = meta
    if not valid_fp:
        logger.info(f"[voice_matcher] 指纹库 {engine_key} 与当前音色池无 embedding 交集, 跳层1")
        return False

    try:
        from videotrans.util.speaker_embedding import compute_embedding, cosine_sim
    except Exception as e:
        logger.warning(f"[voice_matcher] embedding 模块不可用: {e}")
        return False

    used = set(result.values())
    for spk_id, wav in speaker_refs.items():
        if spk_id in result:
            continue
        query_emb = compute_embedding(wav)
        if not query_emb:
            continue
        scored = []
        for voice_name, meta in valid_fp.items():
            score = cosine_sim(query_emb, meta["embedding"])
            scored.append((score, voice_name))
        scored.sort(reverse=True)
        for score, voice_name in scored:
            if voice_name not in used:
                result[spk_id] = voice_name
                details[spk_id] = {"voice": voice_name, "method": "embedding", "score": round(score, 4)}
                used.add(voice_name)
                logger.info(f"[voice_matcher] {spk_id} -> {voice_name} (sim={score:.3f})")
                break
    return True


def _try_gender_match(
    result: Dict[str, str],
    details: Dict[str, dict],
    speaker_refs: Dict[str, str],
    pool: List[str],
    spk_ids: List[str],
    tts_type: int,
) -> bool:
    engine_key = _engine_key(tts_type)
    stats = _tag_summary_by_engine(pool, engine_key)
    if stats["f"] < 2 and stats["m"] < 2:
        logger.info(f"[voice_matcher] 音色池无明显性别区分 ({stats}), 跳层2")
        return False

    female_pool = _filter_by_gender_engine(pool, "f", engine_key)
    male_pool = _filter_by_gender_engine(pool, "m", engine_key)
    any_pool = list(pool)

    spk_gender: Dict[str, str] = {}
    spk_f0: Dict[str, Optional[float]] = {}
    for spk_id, wav in speaker_refs.items():
        g, med = detect_gender_and_f0_from_wav(wav)
        spk_gender[spk_id] = g
        spk_f0[spk_id] = med
    audit_line = (
        f'[voice_matcher] spk F0 性别判定: {spk_gender} '
        f'f0_medians={ {k: round(v,1) if v else None for k,v in spk_f0.items()} } '
        f'(pool stats={stats})'
    )
    logger.info(audit_line)
    # 把审计结果写入 details 特殊键, 让任务侧 (trans_create._signal) 能把此信息
    # 推送到 run.log —— logger.info 只进程内日志, 排错时看不到
    details.setdefault('_audit', {})
    details['_audit'].update({
        'spk_gender': dict(spk_gender),
        'spk_f0': {k: (round(v, 1) if v else None) for k, v in spk_f0.items()},
        'pool_stats': dict(stats),
    })

    # 卫生检查: 池内明显男女双全 (各 ≥ 2) 但待分配 spk 缺失某一性别 → F0 可能被 BGM 污染
    # victim 选择策略 (按 F0 数值而非标签, 对真实性别更鲁棒):
    #   - 缺 'm': 选 F0 中位数最低的 spk 翻到 m 池 (最可能是男声)
    #   - 缺 'f': 选 F0 中位数最高的 spk 翻到 f 池
    #   - 无可用 F0 数值 → 回退到 any spk / pending 末尾
    pending = [sid for sid in spk_ids if sid not in result]
    if len(pending) >= 2 and stats['f'] >= 2 and stats['m'] >= 2:
        detected = [spk_gender.get(sid, 'any') for sid in pending]
        has_m = any(g == 'm' for g in detected)
        has_f = any(g == 'f' for g in detected)
        flip_target = None
        if not has_m and (has_f or all(g == 'any' for g in detected)):
            flip_target = 'm'
        elif not has_f and has_m:
            flip_target = 'f'
        if flip_target:
            # 按 F0 中位数选 victim
            ranked = [(spk_f0.get(sid), sid) for sid in pending]
            with_f0 = [(v, sid) for v, sid in ranked if isinstance(v, (int, float))]
            victim = None
            if with_f0:
                if flip_target == 'm':
                    # 最低 F0 最可能是男声
                    with_f0.sort(key=lambda x: x[0])
                    victim = with_f0[0][1]
                else:
                    with_f0.sort(key=lambda x: x[0], reverse=True)
                    victim = with_f0[0][1]
            if victim is None:
                any_spks = [sid for sid, g in zip(pending, detected) if g == 'any']
                victim = any_spks[-1] if any_spks else pending[-1]
            flip_line = (
                f'[voice_matcher] 待分配 spk 性别分布 {dict(zip(pending, detected))} 缺 {flip_target}; '
                f'按 F0 选 victim={victim} (f0={spk_f0.get(victim)}); 强制切换到 {flip_target} 池'
            )
            logger.warning(flip_line)
            details['_audit']['flip'] = {
                'detected': dict(zip(pending, detected)),
                'missing': flip_target,
                'victim': victim,
                'victim_f0': spk_f0.get(victim),
            }
            spk_gender[victim] = flip_target

    f_cursor = 0
    m_cursor = 0
    a_cursor = 0
    used = set(result.values())
    for spk_id in spk_ids:
        if spk_id in result:
            continue
        gender = spk_gender.get(spk_id, "any")
        if gender == "f":
            current_pool = female_pool
            cursor_key = "f"
        elif gender == "m":
            current_pool = male_pool
            cursor_key = "m"
        else:
            current_pool = any_pool
            cursor_key = "a"

        chosen = None
        for offset in range(len(current_pool)):
            if cursor_key == "f":
                idx = (f_cursor + offset) % len(current_pool)
                f_cursor = idx + 1
            elif cursor_key == "m":
                idx = (m_cursor + offset) % len(current_pool)
                m_cursor = idx + 1
            else:
                idx = (a_cursor + offset) % len(current_pool)
                a_cursor = idx + 1
            candidate = current_pool[idx]
            if candidate not in used:
                chosen = candidate
                break

        if chosen is None and current_pool:
            chosen = current_pool[0]
        if chosen:
            result[spk_id] = chosen
            details[spk_id] = {
                "voice": chosen,
                "method": "gender",
                "score": None,
                "gender": gender,
                "f0": (round(spk_f0.get(spk_id), 1) if isinstance(spk_f0.get(spk_id), (int, float)) else None),
            }
            used.add(chosen)
            logger.info(f"[voice_matcher] {spk_id} gender={gender} -> {chosen}")
    return True


def _fill_missing_with_gender(
    result: Dict[str, str],
    details: Dict[str, dict],
    speaker_refs: Dict[str, str],
    pool: List[str],
    spk_ids: List[str],
    tts_type: int,
) -> None:
    missing = [spk_id for spk_id in spk_ids if spk_id not in result]
    if missing:
        _try_gender_match(result, details, {spk_id: speaker_refs[spk_id] for spk_id in missing}, pool, missing, tts_type)
        still_missing = [spk_id for spk_id in spk_ids if spk_id not in result]
        if still_missing:
            _fill_round_robin(result, details, still_missing, pool)


def _fill_round_robin(result: Dict[str, str], details: Dict[str, dict], spk_ids: List[str], pool: List[str]) -> None:
    used = set(result.values())
    cursor = 0
    for spk_id in spk_ids:
        if spk_id in result:
            continue
        chosen = None
        for offset in range(len(pool)):
            candidate = pool[(cursor + offset) % len(pool)]
            if candidate not in used:
                chosen = candidate
                cursor = (cursor + offset + 1) % len(pool)
                break
        if chosen is None:
            chosen = pool[cursor % len(pool)]
            cursor += 1
        result[spk_id] = chosen
        details[spk_id] = {"voice": chosen, "method": "round_robin", "score": None}
        used.add(chosen)
