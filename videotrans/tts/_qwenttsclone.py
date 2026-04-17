# -*- coding: utf-8 -*-
"""
阿里云百炼 (DashScope) Qwen3-TTS / CosyVoice-v2 声音克隆适配器

依赖 Step 3 产物: queue_tts 每条需含 'ref_wav' (cache_folder/spkN_ref.wav),
以及可选 'ref_text'。由 trans_create._build_speaker_ref_map 注入。

流程:
    1. 从 ref_wav 反推 spk_id (spkN_ref.wav → spkN)
    2. 查 target_dir/qwen_voice_map.json 找 voice_id; 命中跳注册
    3. 未命中:
        a) 上传 spkN_ref.wav 到阿里云 OSS, 生成 1h signed URL
        b) 调 VoiceEnrollmentService.create_voice(target_model, prefix, url)
        c) 拿到 voice_id, 落盘缓存
    4. 合成: SpeechSynthesizer(model=target_model, voice=voice_id).call(text)

配额保护: volcenginetts_clone_max_voices 上限 (默认 10, DashScope 免费额度)
  超限直接 StopRetry 停, 防止扣费失控

OSS 字段 (params.json):
    oss_endpoint, oss_bucket, oss_access_key_id, oss_access_key_secret, oss_prefix
"""
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import dashscope
import oss2
from dashscope.audio.tts_v2 import SpeechSynthesizer, VoiceEnrollmentService

from videotrans.configure._except import StopRetry
from videotrans.configure.config import logger, params
from videotrans.tts._base import BaseTTS
from videotrans.util import tools

DEFAULT_TARGET_MODEL = "cosyvoice-v2"
DEFAULT_MAX_VOICES = 10
SIGNED_URL_EXPIRE_SEC = 3600  # 1 小时, 足够 DashScope 同步拉取


@dataclass
class QwenttsCloneTTS(BaseTTS):
    _voice_map: Dict[str, str] = field(init=False, default_factory=dict)
    _voice_map_path: Optional[Path] = field(init=False, default=None)
    _register_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _max_voices: int = field(init=False, default=DEFAULT_MAX_VOICES)
    _api_key: str = field(init=False, default="")
    _target_model: str = field(init=False, default=DEFAULT_TARGET_MODEL)
    _prefix: str = field(init=False, default="pvt")
    # OSS
    _oss_bucket: Optional[oss2.Bucket] = field(init=False, default=None, repr=False)
    _oss_bucket_name: str = field(init=False, default="")
    _oss_endpoint: str = field(init=False, default="")
    _oss_prefix: str = field(init=False, default="")
    _stop_next_all: bool = field(init=False, default=False)

    def __post_init__(self):
        super().__post_init__()
        self._api_key = params.get("qwentts_key", "") or ""
        self._target_model = params.get("qwentts_clone_model", "") or DEFAULT_TARGET_MODEL
        try:
            self._max_voices = int(params.get("qwentts_clone_max_voices", DEFAULT_MAX_VOICES) or DEFAULT_MAX_VOICES)
        except (ValueError, TypeError):
            self._max_voices = DEFAULT_MAX_VOICES

        # OSS 配置
        self._oss_bucket_name = params.get("oss_bucket", "") or ""
        self._oss_endpoint = params.get("oss_endpoint", "") or ""
        self._oss_prefix = (params.get("oss_prefix", "") or "").strip("/")
        ak = params.get("oss_access_key_id", "") or ""
        sk = params.get("oss_access_key_secret", "") or ""
        if all([self._oss_bucket_name, self._oss_endpoint, ak, sk]):
            auth = oss2.Auth(ak, sk)
            endpoint_url = self._oss_endpoint
            if not endpoint_url.startswith("http"):
                endpoint_url = f"https://{endpoint_url}"
            self._oss_bucket = oss2.Bucket(auth, endpoint_url, self._oss_bucket_name)

        # voice_map 缓存路径: 放到 ref_wav 父目录的父目录 (通常就是 target_dir)
        sample_ref = ""
        sample_filename = ""
        for it in self.queue_tts:
            if it.get("ref_wav"):
                sample_ref = it["ref_wav"]
            if it.get("filename"):
                sample_filename = it["filename"]
            if sample_ref and sample_filename:
                break
        ref_dir = Path(sample_ref).parent if sample_ref else Path(sample_filename).parent
        # 优先放到 target_dir (ref_dir 的父), 以便 clear_cache 不会误删
        target_dir = ref_dir.parent if ref_dir.parent.exists() else ref_dir
        self._voice_map_path = target_dir / "qwen_voice_map.json"

        # 视频名前 8 位作为 prefix (DashScope prefix <= 10 字符)
        raw = target_dir.name or "pvt"
        self._prefix = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]

        # 加载已有缓存
        if self._voice_map_path.exists():
            try:
                self._voice_map = json.loads(self._voice_map_path.read_text(encoding="utf-8"))
                logger.info(f"[QwenClone] 加载 voice_map 缓存 {len(self._voice_map)} 条: {self._voice_map_path}")
            except Exception as e:
                logger.warning(f"[QwenClone] 读取 voice_map 缓存失败: {e}")
                self._voice_map = {}

    def _exec(self):
        if not self._api_key:
            raise StopRetry("请先在 params.json 配置 qwentts_key (百炼 API Key)")
        if not self._oss_bucket:
            raise StopRetry(
                "OSS 未配置完整, 请在 params.json 填 oss_bucket / oss_endpoint / "
                "oss_access_key_id / oss_access_key_secret"
            )
        # 单线程: 避免并发注册同一 spk 重复创建 voice_id
        self._local_mul_thread()

    # ----------------------------------------------------------------- utils

    def _persist_voice_map(self) -> None:
        try:
            self._voice_map_path.parent.mkdir(parents=True, exist_ok=True)
            self._voice_map_path.write_text(
                json.dumps(self._voice_map, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[QwenClone] 写入 voice_map 失败: {e}")

    def _extract_spk_id(self, ref_wav: str) -> str:
        stem = Path(ref_wav).stem
        if stem.endswith("_ref"):
            stem = stem[: -len("_ref")]
        return stem

    def _upload_ref_to_oss(self, local_path: str, spk_id: str) -> str:
        """上传 ref_wav 到 OSS 并返回 1 小时有效的 signed URL。"""
        key_parts = [p for p in [self._oss_prefix, "pyvt-ref", f"{self._prefix}_{spk_id}.wav"] if p]
        key = "/".join(key_parts)
        logger.info(f"[QwenClone] 上传 OSS: {local_path} → oss://{self._oss_bucket_name}/{key}")
        self._oss_bucket.put_object_from_file(key, local_path)
        # signed URL: 即使 bucket 是 private 也能给 DashScope 拉取
        signed = self._oss_bucket.sign_url("GET", key, SIGNED_URL_EXPIRE_SEC, slash_safe=True)
        return signed

    # --------------------------------------------------------------- register

    def _get_or_create_voice(self, spk_id: str, ref_wav_path: str) -> str:
        with self._register_lock:
            if spk_id in self._voice_map:
                return self._voice_map[spk_id]

            if len(self._voice_map) >= self._max_voices:
                raise StopRetry(
                    f"[QwenClone] 已注册 {len(self._voice_map)} 个音色, 达到上限 "
                    f"{self._max_voices} (qwentts_clone_max_voices), 拒绝继续以防扣费失控。"
                    " 去百炼控制台删除旧音色, 或减少说话人数量。"
                )

            if not Path(ref_wav_path).exists():
                raise StopRetry(f"参考音频不存在: {ref_wav_path}")

            # 1) 上传到 OSS 拿公网 URL
            signed_url = self._upload_ref_to_oss(ref_wav_path, spk_id)

            # 2) 调 DashScope 注册音色
            dashscope.api_key = self._api_key
            service = VoiceEnrollmentService()
            logger.info(
                f"[QwenClone] create_voice prefix={self._prefix} target_model={self._target_model} "
                f"spk={spk_id}"
            )
            voice_id = service.create_voice(
                target_model=self._target_model,
                prefix=self._prefix,
                url=signed_url,
            )
            if not voice_id:
                raise RuntimeError(f"create_voice 返回空 voice_id (spk={spk_id})")
            logger.info(f"[QwenClone] {spk_id} → voice_id={voice_id}")

            self._voice_map[spk_id] = voice_id
            self._persist_voice_map()
            self._signal(text=f"[QwenClone] {spk_id} → {voice_id}")
            return voice_id

    # --------------------------------------------------------------- synth

    def _synthesize(self, voice_id: str, data_item: dict) -> None:
        dashscope.api_key = self._api_key
        synthesizer = SpeechSynthesizer(model=self._target_model, voice=voice_id)
        audio = synthesizer.call(data_item["text"])
        if not audio:
            raise RuntimeError(f"合成失败, 返回空音频 (voice_id={voice_id})")

        tmp_wav = data_item["filename"] + "-tmp.wav"
        with open(tmp_wav, "wb") as f:
            f.write(audio)
        self.convert_to_wav(tmp_wav, data_item["filename"])
        try:
            os.remove(tmp_wav)
        except OSError:
            pass

    # --------------------------------------------------------------- entry

    def _item_task(self, data_item: dict = None, idx: int = -1):
        if self._stop_next_all or self._exit() or not data_item.get("text", "").strip():
            return
        if tools.vail_file(data_item["filename"]):
            return

        ref_wav = data_item.get("ref_wav", "")
        if not ref_wav:
            self._stop_next_all = True
            raise StopRetry("QwenClone 需要 ref_wav (检查 Step 3 speaker_refs.json)")

        spk_id = self._extract_spk_id(ref_wav)
        try:
            voice_id = self._get_or_create_voice(spk_id, ref_wav)
            self._synthesize(voice_id, data_item)
        except StopRetry:
            self._stop_next_all = True
            raise
        except Exception as e:
            self.error = e
            raise
