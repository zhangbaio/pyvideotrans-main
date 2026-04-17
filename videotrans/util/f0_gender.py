# -*- coding: utf-8 -*-
"""
源说话人性别判定 (基于基频 F0)

基于 librosa.yin 估计说话人音高中位数:
  - >= 165 Hz → 女 ('f')
  - <= 155 Hz → 男 ('m')
  - 155-165 Hz / 音频无效 → 'any' (不强制)

小孩嗓音与某些女声重叠, 本模块不区分 (交给 voice_tagger 的年龄标签)。
返回值只代表 "这个片段的典型基频落哪一档", 非严格性别识别。
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from videotrans.configure.config import logger


F0_FEMALE_MIN = 165.0
F0_MALE_MAX = 155.0


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

        # librosa.yin: F0 估计. fmin=60Hz (成年男性下限), fmax=400Hz (女性/儿童上限)
        # frame_length 根据 sr 选 2048 (~128ms @16k) 足够包一个周期
        f0 = librosa.yin(audio, fmin=60, fmax=400, sr=sr, frame_length=2048)
        if f0 is None or len(f0) == 0:
            return 'any'

        # 只保留有声帧: yin 在无声帧会给 60Hz 下限附近的值. 取中位数稳过均值
        f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 > 70) & (f0 < 400)]  # 裁剪异常
        if len(f0) < 10:
            return 'any'
        median = float(np.median(f0))
        logger.debug(f'[f0_gender] {p.name} F0 中位数={median:.1f}Hz')
        if median >= F0_FEMALE_MIN:
            return 'f'
        if median <= F0_MALE_MAX:
            return 'm'
        return 'any'
    except Exception as e:
        logger.warning(f'[f0_gender] 估计失败 {wav_path}: {e}')
        return 'any'
