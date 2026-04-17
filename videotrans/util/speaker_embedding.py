# -*- coding: utf-8 -*-
"""
说话人声纹 (embedding) 提取与匹配

统一走 sherpa_onnx 的 ERes2Net (zh) 提取器, 与现有 built speaker diariz 复用同一模型,
避免重复下载 / 多依赖 / pyannote 可选。

模型: 3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx (≈75MB, 192-dim, 16kHz)
  - built 模式做 diariz 时已下载该模型, 这里直接复用
  - pyannote/reverb/ali_CAM 模式首次调用本模块时按需拉取 (modelscope 源)

L2 归一化后存 drama.json, 余弦相似度 = 点积, 阈值默认 0.70
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import threading

from videotrans.configure.config import ROOT_DIR, logger

_EMBEDDING_MODEL = f"{ROOT_DIR}/models/onnx/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"
_EMBEDDING_MODEL_URLS = [
    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx",
]

DEFAULT_MATCH_THRESHOLD = 0.70

_extractor = None
_extractor_lock = threading.Lock()


def _ensure_model() -> bool:
    """首次使用时按需下载 embedding 模型。返回模型是否就绪。"""
    if Path(_EMBEDDING_MODEL).exists():
        return True
    try:
        from videotrans.util import tools
        tools.down_file_from_ms(f"{ROOT_DIR}/models/onnx", _EMBEDDING_MODEL_URLS)
    except Exception as e:
        logger.warning(f"[speaker_embedding] 模型下载失败: {e}")
    return Path(_EMBEDDING_MODEL).exists()


def _get_extractor():
    """延迟初始化 sherpa_onnx 提取器 (单例)。失败返回 None, 调用方需自行降级。"""
    global _extractor
    if _extractor is not None:
        return _extractor
    with _extractor_lock:
        if _extractor is not None:
            return _extractor
        if not _ensure_model():
            return None
        try:
            import sherpa_onnx
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=_EMBEDDING_MODEL)
            _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            logger.info(f"[speaker_embedding] 提取器就绪 (dim={_extractor.dim}, sr={_extractor.sample_rate})")
        except Exception as e:
            logger.warning(f"[speaker_embedding] 初始化失败: {e}")
            _extractor = None
    return _extractor


def compute_embedding(wav_path: str) -> Optional[List[float]]:
    """读 wav → 提取 L2 归一化 embedding → list[float]。失败 None。"""
    try:
        import numpy as np
        import soundfile as sf
    except Exception as e:
        logger.warning(f"[speaker_embedding] 缺依赖: {e}")
        return None

    extractor = _get_extractor()
    if extractor is None:
        return None
    p = Path(wav_path)
    if not p.exists() or p.stat().st_size < 1024:
        return None

    try:
        audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        # 重采样到提取器期望的 sr (通常 16kHz)
        target_sr = extractor.sample_rate
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=target_sr, waveform=audio)
        stream.input_finished()
        emb = extractor.compute(stream)
        if not emb:
            return None
        arr = np.asarray(emb, dtype=np.float32)
        n = float(np.linalg.norm(arr))
        if n <= 0:
            return None
        return (arr / n).tolist()
    except Exception as e:
        logger.warning(f"[speaker_embedding] compute 失败 {wav_path}: {e}")
        return None


def cosine_sim(a: List[float], b: List[float]) -> float:
    """两个 L2 归一化向量的余弦相似度 = 点积。未归一化时结果偏差可接受。"""
    try:
        import numpy as np
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        if va.shape != vb.shape:
            return 0.0
        return float(np.dot(va, vb))
    except Exception:
        return 0.0


def match_best(
    query: List[float],
    candidates: List[Tuple[str, List[float]]],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Optional[Tuple[str, float]]:
    """对候选 (name, embedding) 列表求最高相似度。过阈值返回 (name, score), 否则 None。"""
    if not query or not candidates:
        return None
    best_name, best_score = None, -1.0
    for name, emb in candidates:
        if not emb:
            continue
        s = cosine_sim(query, emb)
        if s > best_score:
            best_name, best_score = name, s
    if best_name is not None and best_score >= threshold:
        return best_name, best_score
    return None
