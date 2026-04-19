
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from videotrans.configure import config as config_module
from videotrans.configure.config import tr, params, settings, app_cfg, logger, ROOT_DIR
from videotrans.configure._base import BaseCon
from videotrans.task.taskcfg import TaskCfgBase
from videotrans.util import tools


@dataclass
class BaseTask(BaseCon):
    # 各项配置信息，例如 翻译、配音、识别渠道等
    cfg: TaskCfgBase = field(default_factory=TaskCfgBase, repr=False)
    # 进度记录
    precent: int = 1
    # 需要配音的原始字幕信息 List[dict]
    queue_tts: List = field(default_factory=list, repr=False)
    # 是否已结束
    hasend: bool = False

    # 名字规范化处理后，应该删除的文件名字
    shound_del_name: str = None

    # 是否需要语音识别
    shoud_recogn: bool = False

    # 是否需要字幕翻译
    shoud_trans: bool = False

    # 是否需要配音
    shoud_dubbing: bool = False

    # 是否需要人声分离
    shoud_separate: bool = False

    # 是否需要嵌入配音或字幕
    shoud_hebing: bool = False
    stage_timings: Dict = field(default_factory=dict, repr=False)
    task_started_at: float = 0.0
    task_finished_at: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.cfg.uuid:
            self.uuid = self.cfg.uuid
        self.task_started_at = time.time()
        self._ensure_task_logging()

    def _ensure_task_logging(self):
        if not self.uuid:
            return
        task_dir = Path(ROOT_DIR) / "logs" / "tasks" / self.uuid
        meta = {
            "basename": getattr(self.cfg, "basename", "") or "",
            "name": getattr(self.cfg, "name", "") or "",
            "target_dir": getattr(self.cfg, "target_dir", "") or "",
            "cache_folder": getattr(self.cfg, "cache_folder", "") or "",
            "task_class": self.__class__.__name__,
        }
        config_module.register_task_log(
            self.uuid,
            log_path=(task_dir / "run.log").as_posix(),
            summary_path=(task_dir / "summary.json").as_posix(),
            meta=meta,
        )

    def _signal(self, **kwargs):
        self._ensure_task_logging()
        return super()._signal(**kwargs)

    def _stage_start(self, name: str):
        stage = self.stage_timings.setdefault(name, {})
        stage["started_at"] = time.time()
        config_module.write_task_log(self.uuid, text=f"stage_start:{name}", level="INFO", event_type="stage")

    def _stage_end(self, name: str):
        stage = self.stage_timings.setdefault(name, {})
        started_at = stage.get("started_at")
        if started_at:
            stage["seconds"] = round(time.time() - started_at, 2)
            config_module.write_task_log(
                self.uuid,
                text=f"stage_end:{name}",
                level="INFO",
                event_type="stage",
                extra={"seconds": stage["seconds"]},
            )

    def _timing_summary(self) -> dict:
        total_sec = self.task_finished_at - self.task_started_at if self.task_finished_at and self.task_started_at else 0.0
        stages = {}
        for name, data in self.stage_timings.items():
            seconds = data.get("seconds")
            if seconds is not None:
                stages[name] = seconds
        return {"total_sec": round(max(total_sec, 0.0), 2), "stages": stages}

    def _task_summary_payload(self, *, status="succeed", extra=None):
        task_log = config_module.app_cfg.task_logs.get(self.uuid, {})
        persisted = task_log.get("persisted") or {}
        payload = {
            "uuid": self.uuid,
            "status": status,
            "task_class": self.__class__.__name__,
            "input": {
                "name": getattr(self.cfg, "name", None),
                "basename": getattr(self.cfg, "basename", None),
            },
            "paths": {
                "target_dir": getattr(self.cfg, "target_dir", None),
                "cache_folder": getattr(self.cfg, "cache_folder", None),
            },
            "timing": self._timing_summary(),
            "task_log": {
                "log_path": persisted.get("log_path") or task_log.get("log_path"),
                "summary_path": persisted.get("summary_path") or task_log.get("summary_path"),
            },
        }
        if extra:
            payload["extra"] = extra
        return payload

    def _finalize_task_logging(self, *, status="succeed", extra=None):
        self._ensure_task_logging()
        persist_dir = getattr(self.cfg, "target_dir", None)
        if getattr(self.cfg, "only_out_mp4", False) and persist_dir:
            persist_dir = Path(persist_dir).parent.as_posix()
        persisted = config_module.persist_task_log_artifacts(
            self.uuid,
            persist_dir,
            basename=getattr(self.cfg, "basename", None),
        )
        if persisted:
            task_log = config_module.app_cfg.task_logs.get(self.uuid, {})
            task_log["persisted"] = persisted
        payload = self._task_summary_payload(status=status, extra=extra)
        config_module.write_task_summary(self.uuid, payload)
        config_module.persist_task_log_artifacts(
            self.uuid,
            persist_dir,
            basename=getattr(self.cfg, "basename", None),
        )

    # 预先处理，例如从视频中拆分音频、人声背景分离、转码等
    def prepare(self):
        pass

    # 语音识别创建原始语言字幕
    def recogn(self):
        pass
    
    # 说话人识别，Funasr/豆包语音识别大模型 /Deepgram 除外，再判断是否已有说话人，Gemini/openai gpt4-dia 会生成说话人
    def diariz(self):
        pass

    # 将原始语言字幕翻译到目标语言字幕
    def trans(self):
        pass

    # 根据 queue_tts 进行配音
    def dubbing(self):
        pass

    # 配音加速、视频慢速对齐
    def align(self):
        pass

    # 视频、音频、字幕合并生成结果文件
    def assembling(self):
        pass

    # 删除临时文件，移动或复制，发送成功消息
    def task_done(self):
        pass

    # 字幕是否存在并且有效
    def _srt_vail(self, file):
        if not file:
            return False
        if not tools.vail_file(file):
            return False
        try:
            tools.get_subtitle_from_srt(file)
        except Exception:
            try:
                Path(file).unlink(missing_ok=True)
            except OSError:
                pass
            return False
        return True

    # 删掉尺寸为0的无效文件
    def _unlink_size0(self, file):
        if not file:
            return
        p = Path(file)
        if p.exists() and p.stat().st_size == 0:
            p.unlink(missing_ok=True)

    # 保存字幕文件 到目标文件夹
    def _save_srt_target(self, srtstr, file):
        # 是字幕列表形式，重新组装
        try:
            txt = tools.get_srt_from_list(srtstr)
            with open(file, "w", encoding="utf-8",errors="ignore") as f:
                f.write(txt)
        except Exception:
            raise
        self._signal(text=Path(file).read_text(encoding='utf-8',errors="ignore"), type='replace_subtitle')
        return True

    def _check_target_sub(self, source_srt_list, target_srt_list):
        import re, copy
        if len(source_srt_list) == 1 or len(target_srt_list) == 1:
            target_srt_list[0]['line'] = 1
            return target_srt_list[:1]
        source_len = len(source_srt_list)
        target_len = len(target_srt_list)
        
        if source_len==target_len:
            for i,it in enumerate(source_srt_list):
                tmp = copy.deepcopy(it)
                tmp['text']=target_srt_list[i]['text']
                target_srt_list[i]=tmp
            return target_srt_list

        if target_len>source_len:
            logger.debug(f'翻译结果行数大于原始字幕行，截取0-{source_len}')
            return target_srt_list[:source_len]
        
        
        logger.debug(f'翻译结果行数少于原始字幕行，追加')
        for i,it in enumerate(source_srt_list):
            if i>=target_len:
                tmp=copy.deepcopy(it)
                tmp['text']=' '
                target_srt_list.append(tmp)
        return target_srt_list
        
    


    async def _edgetts_single(self,target_audio,kwargs):
        from edge_tts import Communicate
        from edge_tts.exceptions import NoAudioReceived
        import aiohttp,asyncio
        from io import BytesIO
        
        useproxy_initial = None if not self.proxy_str or Path(f'{ROOT_DIR}/edgetts-noproxy.txt').exists() else self.proxy_str
        proxies_to_try = [useproxy_initial]
        if useproxy_initial is not None:
            proxies_to_try.append(None)
        last_exception = None
        for proxy in proxies_to_try:
            try:
                audio_buffer = BytesIO()
                communicate_task = Communicate(
                            text=kwargs['text'],
                            voice=kwargs['voice'],
                            rate=kwargs['rate'],
                            volume=kwargs['volume'],
                            proxy=proxy,
                            pitch=kwargs['pitch']
                        )
                idx=0
                async for chunk in communicate_task.stream():
                    if chunk["type"] == "audio":
                        audio_buffer.write(chunk["data"])
                        self._signal(text=f'{idx} segment')
                        idx+=1
                audio_buffer.seek(0)        
                from pydub import AudioSegment
                au=AudioSegment.from_file(audio_buffer,format="mp3")
                au.export(target_audio,format='mp3')
                return
            except (NoAudioReceived, aiohttp.ClientError) as e:
                last_exception = e
            except Exception:
                raise
        raise last_exception if last_exception else RuntimeError(f'Dubbing error')
    # 完整流程判断是否需退出，子功能需重写
    def _exit(self):
        if app_cfg.exit_soft or app_cfg.current_status != 'ing':
            self.hasend=True
            return True
        return False
