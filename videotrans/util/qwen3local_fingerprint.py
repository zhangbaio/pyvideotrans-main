# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict

from videotrans.configure.config import ROOT_DIR, defaulelang, logger

_CUSTOM_VOICES = ["Vivian", "Serena", "Uncle_fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_anna", "Sohee"]
_TEXT = "你好，我是用于音色匹配的参考音频。"


def ensure_qwen3local_fingerprint(is_cuda: bool = False, model_name: str = "1.7B") -> Dict[str, dict]:
    out_path = Path(ROOT_DIR) / "videotrans" / "voicejson" / "qwen3local_emb.json"
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass

    try:
        from videotrans.util import tools
        from videotrans.util.speaker_embedding import compute_embedding
        from qwen_tts import Qwen3TTSModel
        import torch
        import soundfile as sf
    except Exception as e:
        logger.warning(f"[qwen3local_fingerprint] 缺依赖: {e}")
        return {}

    meta_path = Path(ROOT_DIR) / "videotrans" / "voicejson" / "qwentts_emb.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except Exception:
        meta = {}

    try:
        if defaulelang == "zh":
            tools.check_and_down_ms(
                f"Qwen/Qwen3-TTS-12Hz-{model_name}-CustomVoice",
                local_dir=f"{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{model_name}-CustomVoice",
            )
        else:
            tools.check_and_down_hf(
                model_id=f"Qwen3-TTS-12Hz-{model_name}-CustomVoice",
                repo_id=f"Qwen/Qwen3-TTS-12Hz-{model_name}-CustomVoice",
                local_dir=f"{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{model_name}-CustomVoice",
            )
    except Exception as e:
        logger.warning(f"[qwen3local_fingerprint] 模型准备失败: {e}")
        return {}

    atten = None
    if is_cuda and torch.cuda.is_available():
        device_map = "cuda:0"
        dtype = torch.float16
        try:
            import flash_attn  # noqa: F401
        except Exception:
            pass
        else:
            atten = "flash_attention_2"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_map = "mps"
        dtype = torch.float16
    else:
        device_map = "cpu"
        dtype = torch.float32

    model = None
    tmp_dir = Path(tempfile.gettempdir()) / "pyvideotrans" / "qwen3local_emb"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, dict] = {}
    try:
        torch.set_num_threads(1)
        model = Qwen3TTSModel.from_pretrained(
            f"{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{model_name}-CustomVoice",
            device_map=device_map,
            dtype=dtype,
            attn_implementation=atten,
        )
        for voice_name in _CUSTOM_VOICES:
            try:
                wavs, sr = model.generate_custom_voice(
                    text=_TEXT,
                    language="Chinese",
                    speaker=voice_name,
                    instruct="",
                )
                wav_path = tmp_dir / f"{voice_name}.wav"
                sf.write(str(wav_path), wavs[0], sr)
                embedding = compute_embedding(str(wav_path))
                if not embedding:
                    continue
                item = dict(meta.get(voice_name, {})) if isinstance(meta.get(voice_name), dict) else {}
                item["embedding"] = embedding
                results[voice_name] = item
            except Exception as e:
                logger.warning(f"[qwen3local_fingerprint] 生成 {voice_name} 指纹失败: {e}")
        if results:
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"[qwen3local_fingerprint] 已生成 {len(results)} 条音色指纹: {out_path}")
    except Exception as e:
        logger.warning(f"[qwen3local_fingerprint] 构建失败: {e}")
    finally:
        try:
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception:
            pass
    return results
