# -*- coding: utf-8 -*-
"""
说话人 → TTS 音色的自动匹配 (三层降级)

优先级:
  1. 指纹库存在: 源 spk embedding vs 音色 embedding, 余弦相似度 top1 (最准)
  2. 名字可标签: F0 判源性别 + voice_tagger 过滤音色池, 再 round-robin
  3. 裸 round-robin (最后兜底)

对外只一个 API: match_voices_to_speakers()
调用方 (对话框 _auto_assign_speaker_voices) 不关心走哪层, 只拿结果。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from videotrans.configure.config import ROOT_DIR, logger
from videotrans.util.voice_tagger import tag_voice, filter_by_gender, tag_summary
from videotrans.util.f0_gender import detect_gender_from_wav

# tts_type (videotrans.tts.__init__) → 引擎键 (指纹库文件前缀)
# 只登记音色名带描述性标签的引擎, 其他 engine 只走 round-robin
_ENGINE_KEY_MAP = {
    0:  'edgetts',       # EDGE_TTS
    1:  'qwen3local',    # QWEN3LOCAL_TTS
    6:  'doubao2',       # DOUBAO2_TTS
    7:  'doubao0',       # DOUBAO_TTS (streaming)
    18: 'azure',         # AZURE_TTS
    28: 'freeazure',     # FreeAzure
    29: 'googletts',     # GOOGLE_TTS
}

_AUTO_SKIP_VOICES = frozenset({'No', 'clone', ''})


def _engine_key(tts_type: int) -> str:
    return _ENGINE_KEY_MAP.get(int(tts_type), 'unknown')


def _load_fingerprint(engine_key: str) -> Dict[str, dict]:
    """读 videotrans/voicejson/{engine_key}_emb.json, 没有返回 {}"""
    path = Path(ROOT_DIR) / 'videotrans' / 'voicejson' / f'{engine_key}_emb.json'
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        # 兼容两种结构:
        #   {"voice_id": {"embedding": [...], "gender": "f", ...}, ...}
        #   {"voice_name": {...}, ...}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f'[voice_matcher] 读指纹库失败 {path}: {e}')
        return {}


def _candidate_voices(all_voices: List[str]) -> List[str]:
    return [v for v in all_voices if v not in _AUTO_SKIP_VOICES]


def match_voices_to_speakers(
    *,
    speaker_refs: Dict[str, str],
    all_voices: List[str],
    tts_type: int,
    existing: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """核心 API.

    Args:
        speaker_refs: {spk_id: wav_path}, 由 extract_speaker_refs 产出的 spkN_ref.wav
        all_voices: 当前 TTS 引擎的全部可用音色 (含 No/clone, 内部会过滤)
        tts_type: videotrans.tts.__init__ 里的整数常量
        existing: 用户已手动指定的 spk→voice 映射, 这些不覆盖

    Returns:
        {spk_id: voice_name}  覆盖所有 speaker_refs 的 key
        (若全策略都不适用, 保底 round-robin, 不会漏 spk)

    失败策略: 任何一层异常都降到下一层, 最终至少返回 round-robin 结果。
    """
    existing = existing or {}
    pool = _candidate_voices(all_voices)
    if not pool:
        logger.warning('[voice_matcher] 音色池为空, 无法分配')
        return dict(existing)

    spk_ids = list(speaker_refs.keys())
    result: Dict[str, str] = dict(existing)

    # --------- 层 1: 指纹库 embedding 匹配 ---------
    if _try_embedding_match(result, speaker_refs, pool, tts_type):
        _fill_missing_with_gender(result, speaker_refs, pool, spk_ids)
        logger.info(f'[voice_matcher] 层1 指纹匹配: {result}')
        return result

    # --------- 层 2: F0 性别 + 名字标签 ---------
    if _try_gender_match(result, speaker_refs, pool, spk_ids):
        logger.info(f'[voice_matcher] 层2 性别匹配: {result}')
        return result

    # --------- 层 3: 裸 round-robin ---------
    _fill_round_robin(result, spk_ids, pool)
    logger.info(f'[voice_matcher] 层3 round-robin: {result}')
    return result


# --------------------------------------------------------- 层实现


def _try_embedding_match(
    result: Dict[str, str],
    speaker_refs: Dict[str, str],
    pool: List[str],
    tts_type: int,
) -> bool:
    """指纹库存在且源声纹可提 → 余弦相似度 top1, 写 result; 返回是否启用本层。"""
    engine_key = _engine_key(tts_type)
    fp = _load_fingerprint(engine_key)
    if not fp:
        return False

    # 指纹库里的 voice 名 ∩ 当前 pool
    # 指纹 key 可能是 voice_id (BV001_streaming) 或 voice_name (通用女声), 都允许
    valid_fp = {k: v for k, v in fp.items() if k in pool and v.get('embedding')}
    if not valid_fp:
        logger.info(f'[voice_matcher] 指纹库 {engine_key} 与当前音色池无交集, 跳层1')
        return False

    try:
        from videotrans.util.speaker_embedding import compute_embedding, cosine_sim
    except Exception as e:
        logger.warning(f'[voice_matcher] embedding 模块不可用: {e}')
        return False

    used = set(result.values())
    for spk_id, wav in speaker_refs.items():
        if spk_id in result:
            continue
        q = compute_embedding(wav)
        if not q:
            continue
        # 排序所有候选, 选分最高且未被占用
        scored = []
        for voice_name, meta in valid_fp.items():
            s = cosine_sim(q, meta['embedding'])
            scored.append((s, voice_name))
        scored.sort(reverse=True)
        for s, v in scored:
            if v not in used:
                result[spk_id] = v
                used.add(v)
                logger.info(f'[voice_matcher] {spk_id} → {v} (sim={s:.3f})')
                break

    return True  # 本层介入成功 (哪怕个别 spk 提不出 embedding 也算启用)


def _try_gender_match(
    result: Dict[str, str],
    speaker_refs: Dict[str, str],
    pool: List[str],
    spk_ids: List[str],
) -> bool:
    """F0 算每个源 spk 的性别 → voice_tagger 过滤池 → round-robin。

    池里女/男/未知三类分布需要有一定数量, 否则本层无意义, 返回 False。
    """
    stats = tag_summary(pool)
    # 池里至少要有一类性别 >=2 才值得过滤 (都是 'any' 或全同性别时层 2 无意义)
    if stats['f'] < 2 and stats['m'] < 2:
        logger.info(f'[voice_matcher] 音色池无性别区分 ({stats}), 跳层2')
        return False

    female_pool = filter_by_gender(pool, 'f')
    male_pool = filter_by_gender(pool, 'm')
    any_pool = list(pool)

    # 源 spk 性别缓存
    spk_gender: Dict[str, str] = {}
    for spk_id, wav in speaker_refs.items():
        spk_gender[spk_id] = detect_gender_from_wav(wav)

    # 三个池各自的轮询游标
    f_cursor, m_cursor, a_cursor = 0, 0, 0
    used = set(result.values())
    for spk_id in spk_ids:
        if spk_id in result:
            continue
        g = spk_gender.get(spk_id, 'any')
        if g == 'f':
            p = female_pool; cursor_ref = 'f'
        elif g == 'm':
            p = male_pool; cursor_ref = 'm'
        else:
            p = any_pool; cursor_ref = 'a'

        # 选下一个未被占用
        chosen = None
        for offset in range(len(p)):
            if cursor_ref == 'f':
                idx = (f_cursor + offset) % len(p); f_cursor = idx + 1
            elif cursor_ref == 'm':
                idx = (m_cursor + offset) % len(p); m_cursor = idx + 1
            else:
                idx = (a_cursor + offset) % len(p); a_cursor = idx + 1
            cand = p[idx]
            if cand not in used:
                chosen = cand
                break
        # 所有候选都占用了, 放弃冲突避免 (允许重复)
        if chosen is None and p:
            chosen = p[0]
        if chosen:
            result[spk_id] = chosen
            used.add(chosen)
            logger.info(f'[voice_matcher] {spk_id} gender={g} → {chosen}')

    return True


def _fill_missing_with_gender(
    result: Dict[str, str],
    speaker_refs: Dict[str, str],
    pool: List[str],
    spk_ids: List[str],
) -> None:
    """层 1 遗漏的 spk (embedding 提取失败) 降级到层 2 策略补齐。"""
    missing = [s for s in spk_ids if s not in result]
    if missing:
        _try_gender_match(result, {s: speaker_refs[s] for s in missing}, pool, missing)
        # 层 2 若也没覆盖, 再走层 3
        still = [s for s in spk_ids if s not in result]
        if still:
            _fill_round_robin(result, still, pool)


def _fill_round_robin(result: Dict[str, str], spk_ids: List[str], pool: List[str]) -> None:
    used = set(result.values())
    cursor = 0
    for spk_id in spk_ids:
        if spk_id in result:
            continue
        chosen = None
        for offset in range(len(pool)):
            cand = pool[(cursor + offset) % len(pool)]
            if cand not in used:
                chosen = cand
                cursor = (cursor + offset + 1) % len(pool)
                break
        if chosen is None:
            chosen = pool[cursor % len(pool)]
            cursor += 1
        result[spk_id] = chosen
        used.add(chosen)
