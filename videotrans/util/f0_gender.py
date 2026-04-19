# -*- coding: utf-8 -*-
"""
源说话人性别判定 (基于基频 F0)

基于 librosa.pyin 估计说话人音高中位数 (仅统计 voiced 帧, 过滤 BGM/钢琴干扰):
  - >= 185 Hz → 女 ('f')
  - <= 165 Hz → 男 ('m')
  - 165-185 Hz / 音频无效 / voiced 帧不足 → 'any' (保守不强制)

阈值较前版本收紧 (原 155/165), 避免男播音员 (~150-180Hz) + 背景音乐偏置
被误判为女声。中间保守带留给上层 gender-aware round-robin 去自行平衡。
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from videotrans.configure.config import logger


F0_FEMALE_MIN = 185.0
F0_MALE_MAX = 165.0
MIN_VOICED_FRAMES = 10
VOICING_PROB_THRESHOLD = 0.5


def detect_gender_from_wav(wav_path: str) -> str:
    """返回 'f' / 'm' / 'any'. 失败一律 'any', 不阻断上游。"""
    p = Path(wav_path)
    if not p.exists() or p.stat().st_size < 2048:
        return 'any'
    try:
        import numpy as np
        import soundfile as sf
        import librosa
    except Exception as e:
        logger.warning(f'[f0_gender] 缺依赖: {e}')
        return 'any'

    try:
        audio, sr = sf.read(str(p), dtype='float32', always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if len(audio) < sr * 0.5:  # 少于 0.5s 不够
            return 'any'

        # librosa.pyin: 带 voiced 概率的概率版 yin. 过滤 BGM/钢琴等非人声帧
        # fmin=60Hz (成年男性下限), fmax=400Hz (女性/儿童上限), frame_length 2048 (~128ms @16k)
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=60, fmax=400, sr=sr, frame_length=2048
            )
        except Exception:
            # pyin 在部分 librosa 版本对过短音频/全静音段会抛异常, 回退 yin
            f0 = librosa.yin(audio, fmin=60, fmax=400, sr=sr, frame_length=2048)
            voiced_probs = None

        if f0 is None or len(f0) == 0:
            return 'any'

        f0 = np.asarray(f0, dtype=float)
        if voiced_probs is not None:
            # 只统计高置信度 voiced 帧, 排除背景钢琴/环境声
            mask = np.isfinite(f0) & (np.asarray(voiced_probs) > VOICING_PROB_THRESHOLD)
            f0 = f0[mask]
        else:
            f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 > 70) & (f0 < 400)]  # 裁剪异常
        if len(f0) < MIN_VOICED_FRAMES:
            logger.info(f'[f0_gender] {p.name} voiced 帧不足 ({len(f0)}<{MIN_VOICED_FRAMES}) → any')
            return 'any'
        median = float(np.median(f0))
        if median >= F0_FEMALE_MIN:
            gender = 'f'
        elif median <= F0_MALE_MAX:
            gender = 'm'
        else:
            gender = 'any'
        logger.info(f'[f0_gender] {p.name} F0 中位数={median:.1f}Hz voiced_frames={len(f0)} → {gender}')
        return gender
    except Exception as e:
        logger.warning(f'[f0_gender] 估计失败 {wav_path}: {e}')
        return 'any'
