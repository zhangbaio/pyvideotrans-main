# -*- coding: utf-8 -*-
"""
字节火山引擎 声音复刻 2.0 适配器

依赖 Step 3 产物:
    queue_tts 中每条字幕需含 'ref_wav' 字段 (形如 cache_folder/spkN_ref.wav),
    以及可选的 'ref_text'。由 trans_create.py 的 _build_speaker_ref_map 注入。

流程:
    1. 根据 ref_wav 反推 spk_id (文件名 spkN_ref.wav → spkN)
    2. 从磁盘缓存 {target_dir}/doubao_voice_map.json 里找 speaker_id;
       命中则跳过上传, 未命中则走完整注册流程:
         a) POST /api/v1/mega_tts/audio/upload  (传 base64 参考音频)
         b) 轮询 /api/v1/mega_tts/status 直到 status=2 (训练成功) 或超时
         c) 将 (spk_id → speaker_id) 落盘缓存
    3. POST /api/v1/tts 合成, voice_type=speaker_id, cluster=volcano_icl

设计要点:
    - 新 speaker_id 永久占用火山账号音色配额, 用 volcenginetts_clone_max_voices 封顶 (默认 10)
    - speaker_id = "S_" + sha1(video_basename + "_" + spk_id)[:16]
      跨账号唯一 + 同视频重跑可命中 + 不同视频互不影响
    - 合成阶段继承 _doubao.py 的 /api/v1/tts HTTP 协议; 声音复刻 2.0 用 cluster=volcano_icl
"""
import base64
import datetime
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import requests

from videotrans.configure._except import StopRetry
from videotrans.configure.config import logger, params
from videotrans.tts._base import BaseTTS
from videotrans.util import tools

RETRY_NUMS = 2
RETRY_DELAY = 5

# 火山声音复刻 2.0 合成集群
DEFAULT_CLONE_CLUSTER = "volcano_icl"
# 上传 / 状态 / 合成 端点
UPLOAD_URL = "https://openspeech.bytedance.com/api/v1/mega_tts/audio/upload"
STATUS_URL = "https://openspeech.bytedance.com/api/v1/mega_tts/status"
TTS_URL = "https://openspeech.bytedance.com/api/v1/tts"
# 声音复刻专用 Resource-Id 头 (默认值, 可被 params 覆盖)
DEFAULT_RESOURCE_ID = "volc.megatts.voiceclone"

# 训练状态轮询: 成功 / 失败 / 训练中
STATUS_NOT_FOUND = 0
STATUS_TRAINING = 1
STATUS_SUCCESS = 2
STATUS_FAILED = 3
STATUS_ACTIVE = 4

# 火山声音复刻支持的 language 枚举
LANG_MAP = {
    "zh": 0, "en": 1, "ja": 2, "es": 3, "id": 4, "pt": 5,
}


@dataclass
class DoubaoCloneTTS(BaseTTS):
    error_status: ClassVar[Dict[str, str]] = {
        "3001": "无效的请求",
        "3003": "并发超限",
        "3005": "后端服务器负载高",
        "3006": "服务中断",
        "3010": "文本长度超限",
        "3011": "参数有误或者文本为空、文本与语种不匹配",
        "3040": "音色克隆链路网络异常",
        "3050": "音色克隆音色查询失败",
    }

    _voice_map: Dict[str, str] = field(init=False, default_factory=dict)
    _voice_map_path: Optional[Path] = field(init=False, default=None)
    _register_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _max_voices: int = field(init=False, default=10)
    _video_basename: str = field(init=False, default="")
    _appid: str = field(init=False, default="")
    _access_token: str = field(init=False, default="")
    _cluster: str = field(init=False, default=DEFAULT_CLONE_CLUSTER)
    _resource_id: str = field(init=False, default=DEFAULT_RESOURCE_ID)
    _clone_lang: int = field(init=False, default=0)
    _model_type: int = field(init=False, default=1)  # 1 = 2.0 效果版
    _speaker_pool: List[str] = field(init=False, default_factory=list)
    _stop_next_all: bool = field(init=False, default=False)

    def __post_init__(self):
        super().__post_init__()
        self._appid = params.get("volcenginetts_appid", "") or ""
        self._access_token = params.get("volcenginetts_access", "") or ""
        # 合成集群: 声音复刻 2.0 固定 volcano_icl (可被 params 覆盖)
        self._cluster = params.get("volcenginetts_clone_cluster", "") or DEFAULT_CLONE_CLUSTER
        # Resource-Id 头, 不同产品线可能不同
        self._resource_id = params.get("volcenginetts_clone_resource_id", "") or DEFAULT_RESOURCE_ID
        # 克隆时注册语言 (决定训练模型偏向): 默认按源语言
        lang_key = (self.language or "zh")[:2].lower()
        self._clone_lang = LANG_MAP.get(lang_key, 0)
        # 训练模型版本: 1=2.0 (默认), 0=1.0
        self._model_type = int(params.get("volcenginetts_clone_model_type", 1) or 1)
        # 配额保护
        try:
            self._max_voices = int(params.get("volcenginetts_clone_max_voices", 10) or 10)
        except (ValueError, TypeError):
            self._max_voices = 10
        # 预分配的 speaker_id 池子 (火山控制台购买音色后会给一组固定 ID)
        pool = params.get("volcenginetts_clone_speaker_pool", [])
        if isinstance(pool, str):
            # 容错: 允许 "S_a,S_b,S_c" 字符串格式
            pool = [x.strip() for x in pool.split(",") if x.strip()]
        self._speaker_pool = [x for x in pool if isinstance(x, str) and x.startswith("S_")]

        # 缓存文件路径: 用 queue_tts[0]['filename'] 反推 target_dir
        # queue_tts 里每条 filename 形如 <cache_or_target>/xxxx.wav
        # 而 ref_wav 形如 <cache_folder>/spkN_ref.wav
        # target_dir 由 TaskCfg 注入, 但 BaseTTS 没持有, 从 ref_wav 的上一层往上找
        sample_ref = ""
        sample_filename = ""
        for it in self.queue_tts:
            if it.get("ref_wav"):
                sample_ref = it["ref_wav"]
            if it.get("filename"):
                sample_filename = it["filename"]
            if sample_ref and sample_filename:
                break
        # cache_folder 就是 ref_wav 所在目录; target_dir 同父, 用 filename 也可
        # 为简单起见, 把缓存存到 ref_wav 同级 (cache_folder), trans_create 的 Step 3
        # 也同步复制到了 target_dir, 但 clear_cache=True 时 cache_folder 会被清理。
        # 所以这里优先存到 cache_folder 的 **父目录** (通常就是 target_dir) 下。
        cache_dir = Path(sample_ref).parent if sample_ref else Path(sample_filename).parent
        if cache_dir.name.startswith("_tmp") and cache_dir.parent.exists():
            # _tmp 结构 (pyvt 典型约定): target_dir/_tmp_xxx/cache -> 往上找
            self._voice_map_path = cache_dir.parent / "doubao_voice_map.json"
        else:
            self._voice_map_path = cache_dir / "doubao_voice_map.json"

        # 视频 basename 作为 speaker_id 哈希种子
        self._video_basename = cache_dir.parent.name if cache_dir.parent else "unknown"

        # 读取已有缓存
        if self._voice_map_path and self._voice_map_path.exists():
            try:
                self._voice_map = json.loads(self._voice_map_path.read_text(encoding="utf-8"))
                logger.info(f"[DoubaoClone] 加载 voice_map 缓存 {len(self._voice_map)} 条: {self._voice_map_path}")
            except Exception as e:
                logger.warning(f"[DoubaoClone] 读取 voice_map 缓存失败: {e}")
                self._voice_map = {}

    def _exec(self):
        # 强制单线程, 避免火山 QPS 限流 + 并发注册同一 speaker_id
        if not self._appid or not self._access_token:
            raise StopRetry("请先在 params.json 配置 volcenginetts_appid / volcenginetts_access")
        self._local_mul_thread()

    # ----------------------------------------------------------------- helper

    def _persist_voice_map(self) -> None:
        if not self._voice_map_path:
            return
        try:
            self._voice_map_path.parent.mkdir(parents=True, exist_ok=True)
            self._voice_map_path.write_text(
                json.dumps(self._voice_map, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[DoubaoClone] 写入 voice_map 失败: {e}")

    def _allocate_speaker_id(self, spk_id: str) -> str:
        """
        从预分配池子里按顺序给 spk_id 分配一个火山 speaker_id。
        火山 "声音复刻 2.0" 不允许自造 speaker_id, 必须用控制台里已购买的槽位 ID。
        分配规则: 已用过的忽略, 按池子顺序把下一个未用的 ID 给 spk_id, 落盘缓存。
        """
        if not self._speaker_pool:
            raise StopRetry(
                "volcenginetts_clone_speaker_pool 未配置。"
                "请去火山控制台 → 声音复刻详情, 把 5 个 S_xxx speaker_id 填进 params.json"
            )
        used = set(self._voice_map.values())
        for sid in self._speaker_pool:
            if sid not in used:
                return sid
        raise StopRetry(
            f"声音复刻池子全部用光 (已分配 {len(used)} 个: {sorted(used)}) 。"
            "去火山控制台购买更多音色槽位, 或减少说话人数量 (nums_diariz)"
        )

    def _extract_spk_id(self, ref_wav: str) -> str:
        """cache_folder/spkN_ref.wav → spkN; 其它命名 → stem。"""
        stem = Path(ref_wav).stem  # 去掉 .wav
        if stem.endswith("_ref"):
            stem = stem[: -len("_ref")]
        return stem

    def _upload_reference(self, speaker_id: str, ref_wav_path: str) -> None:
        """POST /api/v1/mega_tts/audio/upload"""
        if not Path(ref_wav_path).exists():
            raise StopRetry(f"参考音频不存在: {ref_wav_path}")

        with open(ref_wav_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")

        fmt = Path(ref_wav_path).suffix.lstrip(".").lower() or "wav"
        if fmt == "wave":
            fmt = "wav"

        body = {
            "appid": self._appid,
            "speaker_id": speaker_id,
            "audios": [
                {"audio_bytes": audio_b64, "audio_format": fmt}
            ],
            "source": 2,
            "language": self._clone_lang,
            "model_type": self._model_type,
        }
        headers = {
            "Authorization": f"Bearer;{self._access_token}",
            "Resource-Id": self._resource_id,
            "Content-Type": "application/json",
        }
        logger.info(f"[DoubaoClone] 上传参考音频 {ref_wav_path} → speaker_id={speaker_id}")
        resp = requests.post(UPLOAD_URL, json=body, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        base_resp = data.get("BaseResp") or data.get("base_resp") or {}
        if base_resp.get("StatusCode", 0) != 0:
            raise RuntimeError(f"上传失败: {base_resp}")

    def _poll_until_ready(self, speaker_id: str, timeout_sec: int = 180) -> None:
        """轮询 /api/v1/mega_tts/status 直到训练成功/激活成功或失败。"""
        headers = {
            "Authorization": f"Bearer;{self._access_token}",
            "Resource-Id": self._resource_id,
            "Content-Type": "application/json",
        }
        body = {"appid": self._appid, "speaker_id": speaker_id}
        deadline = time.time() + timeout_sec
        interval = 3
        while time.time() < deadline:
            resp = requests.post(STATUS_URL, json=body, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", STATUS_NOT_FOUND)
            logger.debug(f"[DoubaoClone] status={status} speaker_id={speaker_id}")
            if status in (STATUS_SUCCESS, STATUS_ACTIVE):
                return
            if status == STATUS_FAILED:
                raise RuntimeError(f"声音复刻训练失败: {data}")
            time.sleep(interval)
        raise RuntimeError(f"声音复刻训练超时 (speaker_id={speaker_id})")

    def _get_or_create_voice(self, spk_id: str, ref_wav_path: str) -> str:
        """带缓存+配额保护的 speaker_id 注册入口 (线程安全)。"""
        with self._register_lock:
            if spk_id in self._voice_map:
                return self._voice_map[spk_id]

            # 配额保护
            if len(self._voice_map) >= self._max_voices:
                raise StopRetry(
                    f"[DoubaoClone] 已注册 {len(self._voice_map)} 个音色, 达到上限 "
                    f"{self._max_voices} (volcenginetts_clone_max_voices), 拒绝继续注册以防扣费失控"
                )

            speaker_id = self._allocate_speaker_id(spk_id)
            # 若同名已在火山后台存在 (例如重跑但 voice_map.json 丢了), 先查 status 走 fast-path
            try:
                self._poll_until_ready(speaker_id, timeout_sec=5)
                logger.info(f"[DoubaoClone] 发现已有音色可用, 跳过重新上传: {speaker_id}")
            except Exception:
                # 不存在或未就绪 → 走完整上传流程
                self._upload_reference(speaker_id, ref_wav_path)
                self._poll_until_ready(speaker_id, timeout_sec=180)

            self._voice_map[spk_id] = speaker_id
            self._persist_voice_map()
            self._signal(text=f"[DoubaoClone] {spk_id} → {speaker_id}")
            return speaker_id

    # ---------------------------------------------------------------- synth

    def _synthesize(self, voice_type: str, data_item: dict) -> None:
        """POST /api/v1/tts  -  和 _doubao.py 保持相同协议"""
        speed = 1.0
        if self.rate:
            try:
                speed = 1.0 + float(self.rate.replace("%", "")) / 100
            except ValueError:
                pass
        volume = 1.0
        if self.volume:
            try:
                volume = 1.0 + float(self.volume.replace("%", "")) / 100
            except ValueError:
                pass
        langcode = (self.language or "zh")[:2].lower()
        if langcode == "zh":
            langcode = "cn"

        payload = {
            "app": {
                "appid": self._appid,
                "token": "access_token",  # 实际从 header 里读, 这里传常量即可
                "cluster": self._cluster,
            },
            "user": {"uid": datetime.datetime.now().strftime("%Y%m%d")},
            "audio": {
                "voice_type": voice_type,
                "encoding": "wav",
                "speed_ratio": speed,
                "volume_ratio": volume,
                "pitch_ratio": 1.0,
                "language": langcode,
            },
            "request": {
                "reqid": str(int(time.time() * 100000)),
                "text": data_item["text"],
                "text_type": "plain",
                "silence_duration": 50,
                "operation": "query",
                "pure_english_opt": 1,
            },
        }
        headers = {"Authorization": f"Bearer;{self._access_token}"}
        logger.debug(f"[DoubaoClone] synth voice_type={voice_type} text={data_item['text'][:30]}")
        resp = requests.post(TTS_URL, data=json.dumps(payload), headers=headers, verify=False)
        resp.raise_for_status()
        resp_json = resp.json()

        if "data" in resp_json:
            tmp_wav = data_item["filename"] + "-tmp.wav"
            with open(tmp_wav, "wb") as f:
                f.write(base64.b64decode(resp_json["data"]))
            self.convert_to_wav(tmp_wav, data_item["filename"])
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
            return

        msg = resp_json.get("message", "") or ""
        if "authenticate" in msg or "access denied" in msg.lower():
            self._stop_next_all = True
            raise StopRetry(msg)
        code = str(resp_json.get("code", ""))
        raise RuntimeError(self.error_status.get(code, f"合成失败: {resp_json}"))

    # --------------------------------------------------------------- item

    def _item_task(self, data_item: dict = None, idx: int = -1):
        if self._stop_next_all or self._exit() or not data_item.get("text", "").strip():
            return
        if tools.vail_file(data_item["filename"]):
            return

        ref_wav = data_item.get("ref_wav", "")
        if not ref_wav:
            # 没有参考音频 → 不能用克隆通道, 直接报错终止
            self._stop_next_all = True
            raise StopRetry("DoubaoClone 需要 ref_wav。检查 Step 3 speaker_refs.json 是否生成。")

        spk_id = self._extract_spk_id(ref_wav)
        try:
            voice_type = self._get_or_create_voice(spk_id, ref_wav)
            self._synthesize(voice_type, data_item)
        except StopRetry:
            # 致命: 鉴权 / 配额 / 参考缺失
            self._stop_next_all = True
            raise
        except Exception as e:
            self.error = e
            raise
