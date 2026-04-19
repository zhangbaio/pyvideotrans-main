# -*- coding: utf-8 -*-
"""
轻量情感检测 (P1 A.2)

输入: 原声该句的 wav 片段 + 源文本
输出: (label, instruct_zh)
  label ∈ {neutral, happy, sad, angry, surprised, whisper, shout}
  instruct_zh: 喂给 Qwen3-TTS 的中文风格指令; neutral 时为空串

设计原则:
- 零额外模型依赖, 纯启发式 (F0 / RMS / 文本标点)
- 任何异常一律回退 neutral, 不阻断上游 TTS
- 后续想换 emotion2vec / SenseVoice 带情感标签, 只替换 detect_emotion() 内部实现即可
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, Optional

from videotrans.configure.config import logger

# -------- 情感标签 -> Qwen3-TTS 中文 instruct --------
# 实测 Qwen3-TTS 对这种 prompt 响应很灵敏: 语速/音量/音高都会跟着走
_EMOTION_TO_INSTRUCT = {
    'neutral':   '',
    'happy':     '用开心欢快、轻松明亮的语气说',
    'sad':       '用悲伤低沉、带点难过的语气说',
    'angry':     '用生气愤怒、情绪激动的语气说',
    'surprised': '用惊讶吃惊、语调上扬的语气说',
    'whisper':   '用轻声耳语、压低嗓音的方式说',
    'shout':     '用大声呐喊、情绪高昂的语气说',
}


def instruct_for(label: str) -> str:
    return _EMOTION_TO_INSTRUCT.get(label or 'neutral', '')


# -------- 文本情感线索 (不依赖音频, 100% 准) --------
_INTERJ_ZH = re.compile(r'(啊|哦|哇|呀|嘿|咦|喂|哼|呜|唉|天哪|完了|救命)')
_INTERJ_EN = re.compile(r'\b(oh|wow|huh|hey|ah|ugh|damn|god|jesus|help)\b', re.I)


def _text_cues(text: str) -> dict:
    if not text:
        return {'excl': 0, 'quest': 0, 'ellip': 0, 'caps_ratio': 0.0, 'interj': False}
    excl = text.count('!') + text.count('！')
    quest = text.count('?') + text.count('？')
    ellip = text.count('...') + text.count('…')
    # Latin 大写比例
    latin = [c for c in text if c.isascii() and c.isalpha()]
    caps_ratio = sum(1 for c in latin if c.isupper()) / max(1, len(latin))
    interj = bool(_INTERJ_ZH.search(text) or _INTERJ_EN.search(text))
    return {'excl': excl, 'quest': quest, 'ellip': ellip, 'caps_ratio': caps_ratio, 'interj': interj}


# -------- 音频韵律特征 --------
def _audio_features(wav_path: str) -> Optional[dict]:
    try:
        import numpy as np
        import soundfile as sf
        import librosa
    except Exception as e:
        logger.debug(f'[emotion] 缺依赖: {e}')
        return None

    p = Path(wav_path)
    if not p.exists() or p.stat().st_size < 2048:
        return None
    try:
        audio, sr = sf.read(str(p), dtype='float32', always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if len(audio) < sr * 0.3:  # < 300ms, 特征不可靠
            return None

        # RMS 能量 (10ms hop)
        hop = max(1, int(sr * 0.01))
        rms = librosa.feature.rms(y=audio, frame_length=hop * 2, hop_length=hop)[0]
        rms = rms[np.isfinite(rms) & (rms > 0)]
        if len(rms) < 10:
            return None
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        # 归一化到 [0, 1] 量级: 对短剧一般 RMS 在 0.02-0.3 范围
        rms_energy = min(1.0, rms_mean / 0.15)  # 0.15 ≈ 正常对话
        rms_variance = min(1.0, rms_std / 0.08)

        # F0 (pyin, 只统计 voiced 帧)
        try:
            f0, _, voiced_probs = librosa.pyin(
                audio, fmin=60, fmax=400, sr=sr, frame_length=2048
            )
            if voiced_probs is not None:
                mask = np.isfinite(f0) & (np.asarray(voiced_probs) > 0.5)
                f0_v = f0[mask]
            else:
                f0_v = f0[np.isfinite(f0)]
        except Exception:
            f0_v = np.array([])
        if len(f0_v) >= 10:
            f0_median = float(np.median(f0_v))
            f0_std = float(np.std(f0_v))
            # 说话人基线未知, 用相对变异系数判起伏
            f0_cv = f0_std / max(1.0, f0_median)
        else:
            f0_median = 0.0
            f0_cv = 0.0

        return {
            'rms_energy': rms_energy,
            'rms_variance': rms_variance,
            'f0_median': f0_median,
            'f0_cv': f0_cv,
            'duration': len(audio) / sr,
        }
    except Exception as e:
        logger.debug(f'[emotion] 音频特征提取失败 {wav_path}: {e}')
        return None


# -------- 启发式分类 --------
def detect_emotion(
    wav_path: Optional[str],
    text: str = '',
    subtitle_duration_ms: float = 0,
) -> Tuple[str, str]:
    """
    返回 (label, instruct_zh). 任何异常 → ('neutral', '').

    subtitle_duration_ms: 用于判断语速快/慢 (快语速 + 高能量 → 生气/激动; 慢 → 悲伤)
    """
    try:
        cues = _text_cues(text or '')
        feats = _audio_features(wav_path) if wav_path else None

        # 纯文本决策 (音频不可用时的回退)
        if feats is None:
            if cues['excl'] >= 2 or cues['caps_ratio'] > 0.5:
                return ('shout', _EMOTION_TO_INSTRUCT['shout'])
            if cues['excl'] >= 1 and cues['interj']:
                return ('surprised', _EMOTION_TO_INSTRUCT['surprised'])
            if cues['quest'] >= 1:
                return ('surprised', _EMOTION_TO_INSTRUCT['surprised']) if cues['excl'] else ('neutral', '')
            if cues['ellip'] >= 1:
                return ('sad', _EMOTION_TO_INSTRUCT['sad'])
            return ('neutral', '')

        energy = feats['rms_energy']   # 0-1
        var = feats['rms_variance']    # 0-1
        f0_cv = feats['f0_cv']          # 典型正常 0.1-0.2; 激动 >0.25; 平静 <0.1

        # 1. 极低能量 → whisper
        if energy < 0.25 and var < 0.3:
            return ('whisper', _EMOTION_TO_INSTRUCT['whisper'])

        # 2. 极高能量 + 标点强化 → shout
        if energy > 0.75 and (cues['excl'] >= 1 or cues['caps_ratio'] > 0.3):
            return ('shout', _EMOTION_TO_INSTRUCT['shout'])

        # 3. 高 F0 起伏 + 感叹 → angry/surprised (靠能量区分)
        if f0_cv > 0.22 and cues['excl'] >= 1:
            if energy > 0.55:
                return ('angry', _EMOTION_TO_INSTRUCT['angry'])
            return ('surprised', _EMOTION_TO_INSTRUCT['surprised'])

        # 4. 问句 + 高 F0 起伏 → surprised
        if cues['quest'] >= 1 and f0_cv > 0.18:
            return ('surprised', _EMOTION_TO_INSTRUCT['surprised'])

        # 5. 低能量 + 省略号 / 慢语速 → sad
        if cues['ellip'] >= 1 and energy < 0.5:
            return ('sad', _EMOTION_TO_INSTRUCT['sad'])
        # 慢语速判定: 文本长度 / 时长 过低
        if subtitle_duration_ms > 1500 and text:
            chars_per_sec = len(text) / (subtitle_duration_ms / 1000.0)
            # 中文正常 ~5, 英文 ~14; 过低视为拖长
            if chars_per_sec < 2.5 and energy < 0.5:
                return ('sad', _EMOTION_TO_INSTRUCT['sad'])

        # 6. 中等能量 + 高起伏 + 无负面标点 → happy
        if energy > 0.5 and f0_cv > 0.18 and cues['excl'] == 0 and cues['ellip'] == 0:
            return ('happy', _EMOTION_TO_INSTRUCT['happy'])

        return ('neutral', '')
    except Exception as e:
        logger.debug(f'[emotion] detect 失败, 回退 neutral: {e}')
        return ('neutral', '')
