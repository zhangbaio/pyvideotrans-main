# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from videotrans.configure.config import logger


def analyze_reference_prosody(wav_path: str, text: str = "", duration_ms: int = 0) -> Dict:
    info = {
        "style_tags": [],
        "suggested_rate": 0,
        "energy": None,
        "f0_mean": None,
        "f0_std": None,
    }
    p = Path(wav_path)
    if not p.exists() or p.stat().st_size < 1024:
        return info

    try:
        import numpy as np
        import soundfile as sf
        import librosa
    except Exception as e:
        logger.warning(f"[prosody] 缺依赖: {e}")
        return _fallback_text_prosody(info, text, duration_ms)

    try:
        audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio[:, 0]
        if len(audio) < max(int(sr * 0.2), 1):
            return _fallback_text_prosody(info, text, duration_ms)

        rms = librosa.feature.rms(y=audio).flatten()
        rms = rms[np.isfinite(rms)]
        if len(rms) > 0:
            info["energy"] = round(float(np.median(rms)), 4)

        f0 = librosa.yin(audio, fmin=60, fmax=400, sr=sr, frame_length=2048)
        f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 > 70) & (f0 < 400)]
        if len(f0) > 0:
            info["f0_mean"] = round(float(np.mean(f0)), 2)
            info["f0_std"] = round(float(np.std(f0)), 2)
    except Exception as e:
        logger.warning(f"[prosody] 分析失败 {wav_path}: {e}")
        return _fallback_text_prosody(info, text, duration_ms)

    tags: List[str] = []
    if "!" in text or "！" in text:
        tags.append("excited")
    if "?" in text or "？" in text:
        tags.append("question")
    if info["energy"] is not None and info["energy"] < 0.03:
        tags.append("soft")
    if info["energy"] is not None and info["energy"] > 0.08:
        tags.append("energetic")
    if info["f0_std"] is not None and info["f0_std"] > 35:
        tags.append("expressive")
    if not tags:
        tags.append("neutral")
    info["style_tags"] = tags

    text_len = len((text or "").strip())
    if duration_ms > 0 and text_len > 0:
        cps = text_len / max(duration_ms / 1000.0, 0.2)
        if cps > 8:
            info["suggested_rate"] = min(35, int((cps - 8) * 6))
        elif cps < 4:
            info["suggested_rate"] = max(-15, int((cps - 4) * 5))

    if "excited" in tags or "energetic" in tags:
        info["suggested_rate"] = min(40, info["suggested_rate"] + 5)
    elif "soft" in tags:
        info["suggested_rate"] = max(-20, info["suggested_rate"] - 5)
    return info


def _fallback_text_prosody(info: Dict, text: str, duration_ms: int) -> Dict:
    tags = []
    if "!" in text or "！" in text:
        tags.append("excited")
    if "?" in text or "？" in text:
        tags.append("question")
    if not tags:
        tags.append("neutral")
    info["style_tags"] = tags
    text_len = len((text or "").strip())
    if duration_ms > 0 and text_len > 0:
        cps = text_len / max(duration_ms / 1000.0, 0.2)
        if cps > 8:
            info["suggested_rate"] = min(35, int((cps - 8) * 6))
        elif cps < 4:
            info["suggested_rate"] = max(-15, int((cps - 4) * 5))
    return info
