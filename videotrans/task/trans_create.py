import copy, json, threading
import subprocess
import platform,glob,sys
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union

from videotrans import translator
from videotrans.configure.config import ROOT_DIR,tr,app_cfg,settings,params,TEMP_DIR,logger,defaulelang,HOME_DIR
from videotrans.recognition import run as run_recogn, Faster_Whisper_XXL, Whisper_CPP, \
    is_allow_lang as recogn_allow_lang, FASTER_WHISPER
from videotrans.translator import run as run_trans, get_audio_code
from videotrans.tts import run as run_tts, EDGE_TTS, AZURE_TTS, SUPPORT_CLONE, QWEN_TTS, QWEN3LOCAL_TTS
from videotrans.task.simple_runnable_qt import run_in_threadpool
from videotrans.util import tools, contants
from ._base import BaseTask


# Step 4: 基于 speaker.json 的多角色音色池（仅对 EDGE_TTS 生效）
# 男女交替排列，以便不同 speaker 依次取到差异明显的音色
EDGE_VOICE_POOLS = {
    'en': [
        'Guy(Male/US)',
        'Aria(Female/US)',
        'Christopher(Male/US)',
        'Jenny(Female/US)',
        'Eric(Male/US)',
        'Ava(Female/US)',
        'Brian(Male/US)',
        'Michelle(Female/US)',
    ],
    'zh': [
        'Yunxi(Male/CN)',
        'Xiaoxiao(Female/CN)',
        'Yunyang(Male/CN)',
        'Xiaoyi(Female/CN)',
        'Yunjian(Male/CN)',
        'Yunxia(Male/CN)',
        'Xiaoni(Female/CN)',
        'Xiaobei(Female/CN)',
    ],
}


@dataclass
class TransCreate(BaseTask):
    # 存放原始语言字幕
    source_srt_list: List = field(default_factory=list)
    # 存放目标语言字幕
    target_srt_list: List = field(default_factory=list)
    # 原始视频时长  在慢速处理合并后，时长更新至此
    video_time: float = 0.0
    # 视频信息
    """
    {
        "video_fps":0,
        "video_codec_name":"h264",
        "audio_codec_name":"aac",
        "width":0,
        "height":0,
        "time":0
    }
    """
    video_info: Dict = field(default_factory=dict, repr=False)
    # 对视频是否执行 c:v copy 操作
    is_copy_video: bool = False
    # 需要输出的mp4编码类型 264 265
    video_codec_num: int = 264
    # 是否忽略音频和视频对齐
    ignore_align: bool = False

    # 是否是音频翻译任务，如果是，则到配音完毕即结束，无需合并
    is_audio_trans: bool = False
    queue_tts: List = field(default_factory=list, repr=False)

    def __post_init__(self):
        # 首先，处理本类的默认配置
        super().__post_init__()
        if getattr(self.cfg, 'replace_voice_only', False):
            self.cfg.target_language = self.cfg.source_language
            self.cfg.target_language_code = self.cfg.source_language_code
        if self.cfg.clear_cache and Path(self.cfg.target_dir).is_dir():
            shutil.rmtree(self.cfg.target_dir, ignore_errors=True)
        self._signal(text=tr('kaishichuli'))
        # -1=不启用说话人，0=启用并且不限制说话人数量，>0+1 为最大说话人数量
        self.max_speakers = self.cfg.nums_diariz if self.cfg.enable_diariz else -1
        if self.max_speakers > 0:
            self.max_speakers += 1
        self.shoud_recogn = True
        # 输出编码，  264 或 265
        self.video_codec_num = int(settings.get('video_codec', 264))
        # 是否存在手动添加的背景音频
        if tools.vail_file(self.cfg.back_audio):
            self.cfg.background_music = Path(self.cfg.back_audio).as_posix()

        # 临时文件夹
        if not self.cfg.cache_folder:
            self.cfg.cache_folder = f"{TEMP_DIR}/{self.uuid}"
        # 输出文件夹，去掉可能存在的双斜线
        self.cfg.target_dir = re.sub(r'/{2,}', '/', self.cfg.target_dir, flags=re.I | re.S)
        # 检测字幕原始语言
        self.cfg.detect_language = get_audio_code(show_source=self.cfg.source_language_code)

        # 存放分离后的无声mp4到临时文件夹
        self.cfg.novoice_mp4 = f"{self.cfg.cache_folder}/novoice.mp4"

        # 原始语言字幕文件：输出文件夹
        self.cfg.source_sub = f"{self.cfg.target_dir}/{self.cfg.source_language_code}.srt"
        # 原始语言音频文件：输出文件夹
        self.cfg.source_wav_output = f"{self.cfg.target_dir}/{self.cfg.source_language_code}.m4a"
        # 原始语言音频文件：临时文件夹
        self.cfg.source_wav = f"{self.cfg.cache_folder}/{self.cfg.source_language_code}.wav"

        # 目标语言字幕：输出文件夹
        self.cfg.target_sub = f"{self.cfg.target_dir}/{self.cfg.target_language_code}.srt"
        # 配音后的目标音频文件：输出文件夹
        self.cfg.target_wav_output = f"{self.cfg.target_dir}/{self.cfg.target_language_code}.m4a"
        # 配音后的目标音频文件：临时文件夹
        self.cfg.target_wav = f"{self.cfg.cache_folder}/target.wav"

        # 最终需要输出的mp4视频
        self.cfg.targetdir_mp4 = f"{self.cfg.target_dir}/{self.cfg.noextname}.mp4"

        # 如果配音角色不是No 则需要配音
        if self.cfg.voice_role and self.cfg.voice_role != 'No' and self.cfg.target_language_code:
            self.shoud_dubbing = True

        # 如果不是 tiqu，则均需要合并视频音频字幕
        if self.cfg.app_mode != 'tiqu' and (self.shoud_dubbing or self.cfg.subtitle_type > 0):
            self.shoud_hebing = True

        # 是否需要翻译:存在目标语言代码并且不等于原始语言，则需要翻译
        if self.cfg.target_language_code and self.cfg.target_language_code != self.cfg.source_language_code:
            self.shoud_trans = True

        # 如果原语言和目标语言相等，并且存在配音角色，则替换配音
        if self.cfg.voice_role and self.cfg.voice_role != 'No' and self.cfg.source_language_code == self.cfg.target_language_code:
            self.cfg.target_wav_output = f"{self.cfg.target_dir}/{self.cfg.target_language_code}-dubbing.m4a"
            self.cfg.target_wav = f"{self.cfg.cache_folder}/target-dubbing.wav"
            self.shoud_dubbing = True

        if getattr(self.cfg, 'replace_voice_only', False):
            self.shoud_trans = False
            self.shoud_dubbing = bool(self.cfg.voice_role and self.cfg.voice_role != 'No')

        # 判断如果是音频，则到生成音频结束，无需合并，并且无需分离视频、无需背景音处理
        if self.cfg.ext in contants.AUDIO_EXITS:
            self.is_audio_trans = True
            #self.cfg.is_separate = False
            self.shoud_hebing = False

        # 没有设置目标语言，不配音不翻译
        if not self.cfg.target_language_code:
            self.shoud_dubbing = False
            self.shoud_trans = False

        if self.cfg.voice_role == 'No':
            self.shoud_dubbing = False

        if self.cfg.app_mode == 'tiqu':
            #self.cfg.is_separate = False
            self.cfg.enable_diariz = False
            self.shoud_dubbing = False

        # 记录最终使用的配置信息
        logger.debug(f"最终配置信息：{self.cfg=}")
        # 禁止修改字幕
        self._load_lipsync_config()
        self._signal(text="forbid", type="disabled_edit")

        # 开启一个线程显示进度
        def runing():
            t = time.time()
            while not self.hasend:
                if self._exit(): return
                time.sleep(1)
                self._signal(text=f"{int(time.time() - t)}???{self.precent}", type="set_precent")
        if app_cfg.exec_mode != 'cli':
            threading.Thread(target=runing, daemon=True).start()

    # 1. 预处理，分离音视频、分离人声等
    def _load_lipsync_config(self):
        def _to_bool(val, default=False):
            if isinstance(val, bool):
                return val
            if val is None:
                return default
            if isinstance(val, str):
                return val.strip().lower() in {'1', 'true', 'yes', 'on'}
            return bool(val)

        def _to_int(val, default):
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        home_dir = Path.home()
        auto_lipsync = (
            self._pick_existing_dir(Path(ROOT_DIR).parent / 'MuseTalk-main') is not None
            and self._pick_existing_file(
                home_dir / 'miniconda3' / 'envs' / 'MuseTalk' / 'python.exe',
                home_dir / 'anaconda3' / 'envs' / 'MuseTalk' / 'python.exe',
                home_dir / 'miniconda3' / 'envs' / 'MuseTalk' / 'bin' / 'python',
                home_dir / 'anaconda3' / 'envs' / 'MuseTalk' / 'bin' / 'python',
            ) is not None
        )
        self.cfg.enable_lipsync = _to_bool(params.get('enable_lipsync', auto_lipsync if not self.cfg.enable_lipsync else self.cfg.enable_lipsync), self.cfg.enable_lipsync)
        self.cfg.lipsync_engine = str(params.get('lipsync_engine', self.cfg.lipsync_engine) or self.cfg.lipsync_engine or 'musetalk').strip().lower()
        self.cfg.lipsync_model_root = str(params.get('lipsync_model_root', self.cfg.lipsync_model_root) or self.cfg.lipsync_model_root or '').strip()
        self.cfg.lipsync_python = str(params.get('lipsync_python', self.cfg.lipsync_python) or self.cfg.lipsync_python or '').strip()
        self.cfg.lipsync_ffmpeg_dir = str(params.get('lipsync_ffmpeg_dir', self.cfg.lipsync_ffmpeg_dir) or self.cfg.lipsync_ffmpeg_dir or '').strip()
        self.cfg.lipsync_version = str(params.get('lipsync_version', self.cfg.lipsync_version) or self.cfg.lipsync_version or 'v15').strip().lower()
        self.cfg.lipsync_batch_size = max(1, _to_int(params.get('lipsync_batch_size', self.cfg.lipsync_batch_size), self.cfg.lipsync_batch_size))
        self.cfg.lipsync_bbox_shift = _to_int(params.get('lipsync_bbox_shift', self.cfg.lipsync_bbox_shift), self.cfg.lipsync_bbox_shift)
        self.cfg.lipsync_extra_margin = max(0, _to_int(params.get('lipsync_extra_margin', self.cfg.lipsync_extra_margin), self.cfg.lipsync_extra_margin))
        self.cfg.lipsync_audio_padding_length_left = max(0, _to_int(params.get('lipsync_audio_padding_length_left', self.cfg.lipsync_audio_padding_length_left), self.cfg.lipsync_audio_padding_length_left))
        self.cfg.lipsync_audio_padding_length_right = max(0, _to_int(params.get('lipsync_audio_padding_length_right', self.cfg.lipsync_audio_padding_length_right), self.cfg.lipsync_audio_padding_length_right))
        self.cfg.lipsync_use_fp16 = _to_bool(params.get('lipsync_use_fp16', self.cfg.lipsync_use_fp16), self.cfg.lipsync_use_fp16)
        self.cfg.lipsync_parsing_mode = str(params.get('lipsync_parsing_mode', self.cfg.lipsync_parsing_mode) or self.cfg.lipsync_parsing_mode or 'jaw').strip()
        self.cfg.lipsync_left_cheek_width = max(1, _to_int(params.get('lipsync_left_cheek_width', self.cfg.lipsync_left_cheek_width), self.cfg.lipsync_left_cheek_width))
        self.cfg.lipsync_right_cheek_width = max(1, _to_int(params.get('lipsync_right_cheek_width', self.cfg.lipsync_right_cheek_width), self.cfg.lipsync_right_cheek_width))

    def _pick_existing_file(self, *candidates):
        for candidate in candidates:
            if not candidate:
                continue
            p = Path(candidate).expanduser()
            if p.is_file():
                return p.resolve().as_posix()
        return None

    def _pick_existing_dir(self, *candidates):
        for candidate in candidates:
            if not candidate:
                continue
            p = Path(candidate).expanduser()
            if p.is_dir():
                return p.resolve().as_posix()
        return None

    def _resolve_musetalk_setup(self):
        home_dir = Path.home()
        root_candidates = [
            self.cfg.lipsync_model_root,
            os.environ.get('PYVIDEOTRANS_MUSETALK_ROOT'),
            Path(ROOT_DIR).parent / 'MuseTalk-main',
        ]
        musetalk_root = self._pick_existing_dir(*root_candidates)
        if not musetalk_root:
            raise RuntimeError('MuseTalk root not found. Configure params.lipsync_model_root.')

        root_path = Path(musetalk_root)
        inference_script = root_path / 'scripts' / 'inference.py'
        if not inference_script.exists():
            raise RuntimeError(f'MuseTalk inference script not found: {inference_script.as_posix()}')

        python_candidates = [
            self.cfg.lipsync_python,
            os.environ.get('PYVIDEOTRANS_MUSETALK_PYTHON'),
            home_dir / 'miniconda3' / 'envs' / 'MuseTalk' / 'python.exe',
            home_dir / 'anaconda3' / 'envs' / 'MuseTalk' / 'python.exe',
            home_dir / 'miniconda3' / 'envs' / 'MuseTalk' / 'bin' / 'python',
            home_dir / 'anaconda3' / 'envs' / 'MuseTalk' / 'bin' / 'python',
        ]
        python_exe = self._pick_existing_file(*python_candidates)
        if not python_exe:
            raise RuntimeError('MuseTalk python interpreter not found. Configure params.lipsync_python.')

        env_root = Path(python_exe).resolve().parent.parent
        ffmpeg_candidates = [
            self.cfg.lipsync_ffmpeg_dir,
            os.environ.get('PYVIDEOTRANS_LIPSYNC_FFMPEG_DIR'),
            env_root / 'Library' / 'bin',
            Path(ROOT_DIR).parent / 'video-subtitle-remover-main' / 'backend' / 'ffmpeg' / 'win_x64',
            Path(ROOT_DIR) / 'ffmpeg',
        ]
        ffmpeg_dir = self._pick_existing_dir(*ffmpeg_candidates)
        if not ffmpeg_dir:
            ffmpeg_cmd = shutil.which('ffmpeg')
            if ffmpeg_cmd:
                ffmpeg_dir = Path(ffmpeg_cmd).resolve().parent.as_posix()
        if not ffmpeg_dir:
            raise RuntimeError('MuseTalk ffmpeg directory not found. Configure params.lipsync_ffmpeg_dir.')
        ffmpeg_name = 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg'
        if not Path(ffmpeg_dir, ffmpeg_name).exists():
            raise RuntimeError(f'MuseTalk ffmpeg binary not found in: {ffmpeg_dir}')

        unet_dir = root_path / 'models' / ('musetalkV15' if self.cfg.lipsync_version == 'v15' else 'musetalk')
        model_file = unet_dir / ('unet.pth' if self.cfg.lipsync_version == 'v15' else 'pytorch_model.bin')
        config_file = unet_dir / 'musetalk.json'
        if not model_file.exists() or not config_file.exists():
            raise RuntimeError(f'MuseTalk model files missing under: {unet_dir.as_posix()}')

        return {
            'root': root_path.as_posix(),
            'python': python_exe,
            'ffmpeg_dir': ffmpeg_dir,
            'unet_model': model_file.as_posix(),
            'unet_config': config_file.as_posix(),
        }

    def _run_musetalk_lipsync(self):
        setup = self._resolve_musetalk_setup()
        input_video = Path(self.cfg.novoice_mp4).resolve().as_posix()
        input_audio = Path(self.cfg.target_wav).resolve().as_posix()
        if not tools.vail_file(input_video):
            raise RuntimeError(f'Lip sync input video missing: {input_video}')
        if not tools.vail_file(input_audio):
            raise RuntimeError(f'Lip sync input audio missing: {input_audio}')

        result_dir = Path(self.cfg.cache_folder) / 'musetalk_results'
        result_dir.mkdir(parents=True, exist_ok=True)
        output_name = 'lipsync.mp4'
        output_path = result_dir / self.cfg.lipsync_version / output_name
        infer_cfg = result_dir / 'infer.yaml'
        infer_cfg.write_text(
            "\n".join([
                "task_0:",
                f'  video_path: "{input_video}"',
                f'  audio_path: "{input_audio}"',
                f'  result_name: "{output_name}"',
                f'  bbox_shift: {self.cfg.lipsync_bbox_shift}',
                "",
            ]),
            encoding='utf-8'
        )

        cmd = [
            setup['python'],
            '-m',
            'scripts.inference',
            '--inference_config',
            infer_cfg.as_posix(),
            '--result_dir',
            result_dir.as_posix(),
            '--unet_model_path',
            setup['unet_model'],
            '--unet_config',
            setup['unet_config'],
            '--version',
            self.cfg.lipsync_version,
            '--ffmpeg_path',
            setup['ffmpeg_dir'],
            '--batch_size',
            str(self.cfg.lipsync_batch_size),
            '--output_vid_name',
            output_name,
            '--bbox_shift',
            str(self.cfg.lipsync_bbox_shift),
            '--extra_margin',
            str(self.cfg.lipsync_extra_margin),
            '--audio_padding_length_left',
            str(self.cfg.lipsync_audio_padding_length_left),
            '--audio_padding_length_right',
            str(self.cfg.lipsync_audio_padding_length_right),
            '--parsing_mode',
            self.cfg.lipsync_parsing_mode,
            '--left_cheek_width',
            str(self.cfg.lipsync_left_cheek_width),
            '--right_cheek_width',
            str(self.cfg.lipsync_right_cheek_width),
        ]
        if self.cfg.lipsync_use_fp16 and self.cfg.is_cuda:
            cmd.append('--use_float16')

        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PATH'] = os.pathsep.join([
            str(Path(setup['python']).parent),
            str(Path(setup['python']).parent.parent / 'Library' / 'bin'),
            setup['ffmpeg_dir'],
            env.get('PATH', ''),
        ])
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        logger.debug(f'[MuseTalk-CMD]\n{" ".join(cmd)}\n')
        proc = subprocess.run(
            cmd,
            cwd=setup['root'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            creationflags=creationflags,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or '').strip()
            if detail:
                detail = "\n".join(detail.splitlines()[-20:])
            raise RuntimeError(f'MuseTalk failed with code {proc.returncode}: {detail}')
        if not output_path.exists():
            raise RuntimeError(f'MuseTalk output not found: {output_path.as_posix()}')

        self.cfg.novoice_mp4 = output_path.as_posix()
        self.video_time = tools.get_video_duration(self.cfg.novoice_mp4)
        logger.debug(f'[MuseTalk] generated {self.cfg.novoice_mp4}')
        return True

    def lipsync(self) -> None:
        if self._exit():
            return
        if not self.cfg.enable_lipsync or self.is_audio_trans:
            return
        if not self.shoud_dubbing or self.cfg.app_mode == 'tiqu':
            return
        if self.cfg.lipsync_engine != 'musetalk':
            logger.debug(f'[LipSync] skip unsupported engine: {self.cfg.lipsync_engine}')
            return
        if not tools.vail_file(self.cfg.novoice_mp4) or not tools.vail_file(self.cfg.target_wav):
            logger.debug('[LipSync] skip because input video or audio is missing')
            return

        self._stage_start("lipsync")
        self._signal(text='Starting lip sync with MuseTalk')
        self.precent += 2
        try:
            self._run_musetalk_lipsync()
        except Exception:
            self.hasend = True
            raise
        self._signal(text='Lip sync finished')
        self._stage_end("lipsync")

    def prepare(self) -> None:
        if self._exit(): return
        self._stage_start("prepare")
        self._signal(text=tr("Hold on a monment..."))
        Path(self.cfg.cache_folder).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.target_dir).mkdir(parents=True, exist_ok=True)
        # 删掉可能存在的无效文件
        self._unlink_size0(self.cfg.source_sub)
        self._unlink_size0(self.cfg.target_sub)
        self._unlink_size0(self.cfg.targetdir_mp4)

        try:
            # 删掉已存在的，可能会失败
            Path(self.cfg.source_wav).unlink(missing_ok=True)
            Path(self.cfg.source_wav_output).unlink(missing_ok=True)
            Path(self.cfg.target_wav).unlink(missing_ok=True)
            Path(self.cfg.target_wav_output).unlink(missing_ok=True)
        except Exception as e:
            logger.exception(f'删除已存在的文件时失败:{e}', exc_info=True)

        self.video_info = tools.get_video_info(self.cfg.name)
        # 毫秒
        self.video_time = self.video_info['time']
        audio_stream_len = self.video_info.get('streams_audio', 0)

        # 无视频流，不是音频，并且不是提取，报错
        if self.video_info.get('video_streams', 0) < 1 and not self.is_audio_trans and self.cfg.app_mode != 'tiqu':
            self.hasend = True
            raise RuntimeError(
                tr('The video file {} does not contain valid video data and cannot be processed.', self.cfg.name))

        # 无音频流，不存在原语言字幕，报错。存在则是无声视频流
        if audio_stream_len < 1 and not tools.vail_file(self.cfg.source_sub):
            self.hasend = True
            raise RuntimeError(
                tr('There is no valid audio in the file {} and it cannot be processed. Please play it manually to confirm that there is sound.',
                   self.cfg.name))

        # 如果获得原始视频编码格式是 h264，并且色素 yuv420p, 则直接复制视频流 is_copy_video=True
        if self.video_info['video_codec_name'] == 'h264' and self.video_info['color'] == 'yuv420p':
            self.is_copy_video = True

        # 如果存在字幕文本，则视为原始语言字幕，不再识别
        if self.cfg.subtitles.strip():
            with open(self.cfg.source_sub, 'w', encoding="utf-8", errors="ignore") as f:
                txt = re.sub(r':\d+\.\d+', lambda m: m.group().replace('.', ','),
                             self.cfg.subtitles.strip(), flags=re.I | re.S)
                f.write(txt)
            self.shoud_recogn = False

        # 是否需要背景音分离
        if self.cfg.is_separate:
            self.cfg.vocal = f"{self.cfg.cache_folder}/vocal.wav"
            self.cfg.instrument = f"{self.cfg.cache_folder}/instrument.wav"
            self._unlink_size0(self.cfg.instrument)
            self._unlink_size0(self.cfg.vocal)

            # 判断是否已存在
            raw_instrument = f"{self.cfg.target_dir}/instrument.wav"
            raw_vocal = f"{self.cfg.target_dir}/vocal.wav"
            if tools.vail_file(raw_instrument) and tools.vail_file(raw_vocal):
                try:
                    shutil.copy2(raw_instrument, self.cfg.instrument)
                    shutil.copy2(raw_vocal, self.cfg.vocal)
                except shutil.SameFileError:
                    pass
            self.shoud_separate = True

        # 将原始视频分离为无声视频
        if not self.is_audio_trans and self.cfg.app_mode != 'tiqu':
            app_cfg.queue_novice[self.uuid] = 'ing'
            if not self.is_copy_video:
                self._signal(text=tr("Video needs transcoded and take a long time.."))
            run_in_threadpool(self._split_novoice_byraw)
        else:
            app_cfg.queue_novice[self.uuid] = 'end'

        # 需要人声背景声分离，并且不存在已分离好的文件
        if audio_stream_len > 0 and self.cfg.is_separate and (
                not tools.vail_file(self.cfg.vocal) or not tools.vail_file(self.cfg.instrument)):
            self._signal(text=tr('Separating background music'))
            try:
                self._split_audio_byraw(True)
            except Exception as e:
                logger.exception(f'分离人声背景声失败', exc_info=True)
            finally:
                if not tools.vail_file(self.cfg.vocal) or not tools.vail_file(self.cfg.instrument):
                    # 分离失败
                    self.cfg.instrument = None
                    self.cfg.vocal = None
                    self.cfg.is_separate = False
                    self.shoud_separate = False

        # 如果还不存在原音频 self.cfg.source_wav,可能原因上一步分离人声背景声失败
        if audio_stream_len > 0 and not tools.vail_file(self.cfg.source_wav):
            self._split_audio_byraw()

        self._signal(text=tr('endfenliyinpin'))
        self._stage_end("prepare")

    # 开始识别
    def recogn(self) -> None:
        if self._exit(): return
        if not self.shoud_recogn: return
        self._stage_start("recogn")
        self.precent += 3
        self._signal(text=tr("kaishishibie"))
        if tools.vail_file(self.cfg.source_sub):
            self.source_srt_list = tools.get_subtitle_from_srt(self.cfg.source_sub, is_file=True)
            if Path(self.cfg.target_dir + "/speaker.json").exists():
                shutil.copy2(self.cfg.target_dir + "/speaker.json", self.cfg.cache_folder + "/speaker.json")
            self._recogn_succeed()
            self._stage_end("recogn")
            return

        if not tools.vail_file(self.cfg.source_wav):
            error = tr("Failed to separate audio, please check the log or retry")
            self.hasend = True
            raise RuntimeError(error)

        # 若已执行背景声人声分离，则不再进行降噪
        if not self.cfg.is_separate and self.cfg.remove_noise:
            title = tr("Starting to process speech noise reduction, which may take a long time, please be patient")
            _remove_noise_wav=f"{self.cfg.cache_folder}/remove_noise.wav"
            _cache_noise_wav=f"{self.cfg.target_dir}/removed_noise.wav"
            if not Path(_cache_noise_wav).exists():
                tools.check_and_down_ms(model_id='iic/speech_frcrn_ans_cirm_16k', callback=self._process_callback)
                from videotrans.process.prepare_audio import remove_noise
                kw = {
                    "input_file": self.cfg.source_wav,
                    "output_file": _remove_noise_wav,
                    "is_cuda": self.cfg.is_cuda
                }
                try:
                    _rs = self._new_process(callback=remove_noise, title=title, is_cuda=self.cfg.is_cuda, kwargs=kw)
                    if _rs:
                        self.cfg.source_wav = _rs
                        shutil.copy2(_rs,_cache_noise_wav)
                    self._signal(text='remove noise end')
                except:
                    pass
            else:
                shutil.copy2(_cache_noise_wav,_remove_noise_wav)
                self.cfg.source_wav = _remove_noise_wav

        self._signal(text=tr("Speech Recognition to Word Processing"))

        if self.cfg.recogn_type == Faster_Whisper_XXL:
            xxl_path = settings.get('Faster_Whisper_XXL', 'Faster_Whisper_XXL.exe')
            cmd = [
                xxl_path,
                self.cfg.source_wav,
                "-pp",
                "-f", "srt"
            ]
            cmd.extend(['-l', self.cfg.detect_language.split('-')[0]])
            prompt = None
            prompt = settings.get(f'initial_prompt_{self.cfg.detect_language}')
            if prompt:
                cmd += ['--initial_prompt', prompt]
            cmd.extend(['--model', self.cfg.model_name, '--output_dir', self.cfg.target_dir])

            txt_file = Path(xxl_path).parent.resolve().as_posix() + '/pyvideotrans.txt'

            if Path(txt_file).exists():
                cmd.extend(Path(txt_file).read_text(encoding='utf-8').strip().split(' '))

            cmdstr = " ".join(cmd)
            outsrt_file = self.cfg.target_dir + '/' + Path(self.cfg.source_wav).stem + ".srt"
            logger.debug(f'Faster_Whisper_XXL: {cmdstr=}\n{outsrt_file=}\n{self.cfg.source_sub=}')

            self._external_cmd_with_wrapper(cmd)

            try:
                shutil.copy2(outsrt_file, self.cfg.source_sub)
            except shutil.SameFileError:
                pass
            self.source_srt_list = tools.get_subtitle_from_srt(self.cfg.source_sub, is_file=True)
        elif self.cfg.recogn_type == Whisper_CPP:
            cpp_path = settings.get('Whisper_cpp', 'whisper-cli')
            cmd = [
                cpp_path,
                "-f",
                self.cfg.source_wav,
                "-osrt",
                "-np"

            ]
            cmd += ["-l", self.cfg.detect_language.split('-')[0]]
            prompt = None
            prompt = settings.get(f'initial_prompt_{self.cfg.detect_language}')
            if prompt:
                cmd += ['--prompt', prompt]
            cpp_folder = Path(cpp_path).parent.resolve().as_posix()
            if not Path(f'{cpp_folder}/models/{self.cfg.model_name}').is_file():
                raise RuntimeError(tr('The model does not exist. Please download the model to the {} directory first.',
                                      f'{cpp_folder}/models'))
            txt_file = cpp_folder + '/pyvideotrans.txt'

            if Path(txt_file).exists():
                cmd.extend(Path(txt_file).read_text(encoding='utf-8').strip().split(' '))

            cmd.extend(['-m', f'models/{self.cfg.model_name}', '-of', self.cfg.source_sub[:-4]])

            logger.debug(f'Whisper.cpp: {cmd=}')

            self._external_cmd_with_wrapper(cmd)
            self.source_srt_list = tools.get_subtitle_from_srt(self.cfg.source_sub, is_file=True)
        else:
            # -1不启用，0不限制数量，>0加1为指定的说话人数量
            logger.debug(f'[trans_create]:run_recogn() {time.time()=}')
            raw_subtitles = run_recogn(
                recogn_type=self.cfg.recogn_type,
                uuid=self.uuid,
                model_name=self.cfg.model_name,
                audio_file=self.cfg.source_wav,
                detect_language=self.cfg.detect_language,
                cache_folder=self.cfg.cache_folder,
                is_cuda=self.cfg.is_cuda,
                subtitle_type=self.cfg.subtitle_type,
                max_speakers=self.max_speakers,
                llm_post=self.cfg.rephrase == 1
            )
            if self._exit(): return
            if not raw_subtitles:
                raise RuntimeError(self.cfg.basename + tr('recogn result is empty'))
            self._save_srt_target(raw_subtitles, self.cfg.source_sub)
            self.source_srt_list = raw_subtitles

        # 中英恢复标点符号
        if self.cfg.fix_punc and self.cfg.detect_language[:2] in ['zh', 'en']:
            tools.check_and_down_ms(model_id='iic/punc_ct-transformer_cn-en-common-vocab471067-large',
                                    callback=self._process_callback)
            from videotrans.process.prepare_audio import fix_punc
            # 预先删掉已有的标点
            text_dict = {f'{it["line"]}': re.sub(r'[,.?!，。？！]', ' ', it["text"]) for it in self.source_srt_list}
            kw = {"text_dict": text_dict, "is_cuda": self.cfg.is_cuda}
            try:
                _rs = self._new_process(callback=fix_punc, title=tr("Restoring punct"), is_cuda=self.cfg.is_cuda,
                                        kwargs=kw)
                if _rs:
                    for it in self.source_srt_list:
                        it['text'] = _rs.get(f'{it["line"]}', it['text'])
                        if self.cfg.detect_language[:2] == 'en':
                            it['text'] = it['text'].replace('，', ',').replace('。', '. ').replace('？', '?').replace('！',
                                                                                                                   '!')
                    self._save_srt_target(self.source_srt_list, self.cfg.source_sub)
            except:
                pass

        self._signal(text=Path(self.cfg.source_sub).read_text(encoding='utf-8'), type='replace_subtitle')
        # whisperx-api
        # openairecogn并且模型是gpt-4o-transcribe-diarize
        # funasr并且模型是paraformer-zh
        # deepgram
        # 以上这些本身已有说话人识别，如果已有说话人识别结果，就不再重新断句
        if Path(self.cfg.cache_folder + "/speaker.json").exists():
            self._recogn_succeed()
            self._signal(text=tr('endtiquzimu'))
            self._stage_end("recogn")
            return

        if self.cfg.rephrase == 1:
            # LLM重新断句
            try:
                from videotrans.translator._chatgpt import ChatGPT

                ob = ChatGPT(uuid=self.uuid)
                self._signal(text=tr("Re-segmenting..."))
                srt_list = ob.llm_segment(self.source_srt_list, settings.get('llm_ai_type', 'openai'))
                if srt_list and len(srt_list) > len(self.source_srt_list) / 2:
                    self.source_srt_list = srt_list
                    shutil.copy2(self.cfg.source_sub, f'{self.cfg.source_sub}-No-{tr("LLM Rephrase")}.srt')
                    self._save_srt_target(self.source_srt_list, self.cfg.source_sub)
                else:
                    raise
            except Exception as e:
                self._signal(text=tr("Re-segmenting Error"))
                logger.warning(f"重新断句失败[except]，已恢复原样 {e}")

        self._recogn_succeed()
        self._signal(text=tr('endtiquzimu'))
        self._stage_end("recogn")

    def _recogn_succeed(self) -> None:
        self.precent += 5
        if self.cfg.app_mode == 'tiqu':
            dest_name = f"{self.cfg.target_dir}/{self.cfg.noextname}"
            if not self.shoud_trans:
                self.hasend = True
                self.precent = 100
                dest_name += '.srt'
                shutil.copy2(self.cfg.source_sub, dest_name)
                Path(self.cfg.source_sub).unlink(missing_ok=True)
            else:
                dest_name += f"-{self.cfg.source_language_code}.srt"
                shutil.copy2(self.cfg.source_sub, dest_name)
        self._signal(text=tr('endtiquzimu'))

    # 配音后再次对配音文件进行识别，以便生成简短的字幕，
    # 开始识别
    def recogn2pass(self) -> None:
        if not self.shoud_dubbing or not self.cfg.recogn2pass or self._exit(): 
            return
        # 如果不嵌入字幕，或嵌入双字幕，则跳过
        if self.cfg.subtitle_type > 2 and (self.cfg.source_language_code != self.cfg.target_language_code):
            logger.debug(f'跳过二次识别, 因设置了嵌入双字幕，二次识别后双字幕时间戳将无法保持一致，因此跳过：{self.cfg.subtitle_type=}')
            return

        if not tools.vail_file(self.cfg.target_wav):
            logger.debug(f'跳过二次识别，因无配音音频文件')
            return
             
        self._stage_start("recogn2pass")
        self.precent += 3
        self._signal(text=tr("Secondary speech recognition of dubbing files"))
        logger.debug(f'进入二次识别')

        shibie_audio = f'{self.cfg.cache_folder}/recogn2pass-{time.time()}.wav'
        outsrt_file = f'{self.cfg.cache_folder}/recogn2pass-{time.time()}.srt'
        try:
            tools.conver_to_16k(self.cfg.target_wav, shibie_audio)
        except Exception as e:
            logger.exception(f'二次识别配音音频生成字幕时，预处理音频失败，静默跳过:{e}', exc_info=True)
            return
        finally:
            if not tools.vail_file(shibie_audio):
                logger.exception(f'二次识别配音音频生成字幕时，预处理音频失败，静默跳过:{e}', exc_info=True)
                return
        
        try:
            # 判断原渠道是否支持目标语言的识别 self.cfg.target_language_code
            recogn_type = self.cfg.recogn_type
            model_name = self.cfg.model_name
            detect_language = self.cfg.target_language_code.split('-')[0]

            if recogn_allow_lang(langcode=self.cfg.target_language_code, recogn_type=recogn_type,
                                 model_name=model_name) is not True:
                recogn_type = FASTER_WHISPER
                model_name = 'large-v3-turbo'

            if recogn_type == Faster_Whisper_XXL:
                xxl_path = settings.get('Faster_Whisper_XXL', 'Faster_Whisper_XXL.exe')
                cmd = [
                    xxl_path,
                    shibie_audio,
                    "-pp",
                    "-f", "srt"
                ]
                cmd.extend(['-l', detect_language.split('-')[0]])
                prompt = settings.get(f'initial_prompt_{detect_language}')
                if prompt:
                    cmd += ['--initial_prompt', prompt]
                cmd.extend(['--model', model_name, '--output_dir', self.cfg.cache_folder])

                txt_file = Path(xxl_path).parent.resolve().as_posix() + '/pyvideotrans.txt'

                if Path(txt_file).exists():
                    cmd.extend(Path(txt_file).read_text(encoding='utf-8').strip().split(' '))

                cmdstr = " ".join(cmd)
                logger.debug(f'Faster_Whisper_XXL: {cmdstr=}\n{outsrt_file=}')
                self._external_cmd_with_wrapper(cmd)
            elif recogn_type == Whisper_CPP:
                cpp_path = settings.get('Whisper_cpp', 'whisper-cli')
                cmd = [
                    cpp_path,
                    "-f",
                    shibie_audio,
                    "-osrt",
                    "-np"

                ]
                cmd += ["-l", detect_language]
                prompt = settings.get(f'initial_prompt_{detect_language}')
                if prompt:
                    cmd += ['--prompt', prompt]
                cpp_folder = Path(cpp_path).parent.resolve().as_posix()
                if not Path(f'{cpp_folder}/models/{model_name}').is_file():
                    logger.error(tr('The model does not exist. Please download the model to the {} directory first.',
                                           f'{cpp_folder}/models'))
                    return
                txt_file = cpp_folder + '/pyvideotrans.txt'
                if Path(txt_file).exists():
                    cmd.extend(Path(txt_file).read_text(encoding='utf-8').strip().split(' '))
                cmd.extend(['-m', f'models/{model_name}', '-of', outsrt_file[:-4]])
                logger.debug(f'Whisper.cpp: {cmd=}')
                self._external_cmd_with_wrapper(cmd)
            else:
                # -1不启用，0不限制数量，>0加1为指定的说话人数量
                logger.debug(f'[trans_create]:二次识别')
                raw_subtitles = run_recogn(
                    recogn_type=recogn_type,
                    uuid=self.uuid,
                    model_name=model_name,
                    audio_file=shibie_audio,
                    detect_language=detect_language,
                    cache_folder=self.cfg.cache_folder,
                    is_cuda=self.cfg.is_cuda,
                    recogn2pass=True  # 二次识别
                )
                if self._exit(): return
                if not raw_subtitles:
                    logger.error('二次识别出错：' + tr('recogn result is empty'))
                self._save_srt_target(raw_subtitles, outsrt_file)

            if not tools.vail_file(outsrt_file):
                logger.error(f'二次识别配音文件失败，原因未知')
                return
            # 覆盖
            try:
                translated_subtitles = tools.get_subtitle_from_srt(self.cfg.target_sub, is_file=True)
                recognized_subtitles = tools.get_subtitle_from_srt(outsrt_file, is_file=True)
            except Exception as merge_error:
                logger.warning(f'二次识别结果无法用于时间轴合并，保留原翻译字幕: {merge_error}')
            else:
                if len(translated_subtitles) == len(recognized_subtitles) and len(translated_subtitles) > 0:
                    merged_subtitles = []
                    for idx, rec_item in enumerate(recognized_subtitles):
                        merged_item = copy.deepcopy(rec_item)
                        merged_item['text'] = translated_subtitles[idx].get('text', rec_item.get('text', '')).strip()
                        merged_subtitles.append(merged_item)
                    self._save_srt_target(merged_subtitles, self.cfg.target_sub)
                    logger.debug('二次识别仅更新时间轴，已保留原始翻译字幕文本')
                else:
                    logger.warning(
                        f'二次识别行数与翻译字幕不一致，保留原翻译字幕不覆盖: '
                        f'{len(translated_subtitles)=}, {len(recognized_subtitles)=}'
                    )
            self._signal(text='STT 2 pass end')
            self._stage_end("recogn2pass")
            logger.debug('二次识别成功完成')
        
        except Exception as e:
            logger.exception(f'二次识别配音音频生成字幕时，预处理音频失败，静默跳过:{e}', exc_info=True)
            return

        return True

    def diariz(self):
        # 说话人设为1，不进行分离
        if self._exit() or not self.cfg.enable_diariz or self.max_speakers == 1:
            return
        self._stage_start("diariz")
        # speaker.json 已存在时直接提取参考音频（跳过重新分离）
        if Path(self.cfg.cache_folder + "/speaker.json").exists():
            self.extract_speaker_refs()
            self._stage_end("diariz")
            return
        # built pyannote reverb ali_CAM
        speaker_type = settings.get('speaker_type', 'built')
        hf_token = settings.get('hf_token')
        if speaker_type == 'built' and self.cfg.detect_language[:2] not in ['zh', 'en']:
            logger.error(f'当前选择 built 说话人分离模型，但不支持当前语言:{self.cfg.detect_language}')
            return
        if speaker_type in ['pyannote', 'reverb'] and not hf_token:
            logger.error(f'当前选择 pyannote 说话人分离模型，但未设置 huggingface.co 的token: {self.cfg.detect_language}')
            return
        if speaker_type in ['pyannote', 'reverb']:
            # 判断是否可访问 huggingface.co
            # 先测试能否连接 huggingface.co, 中国大陆地区不可访问，除非使用VPN
            try:
                import requests
                requests.head('https://huggingface.co', timeout=5)
            except Exception:
                logger.error(f'当前选择 {speaker_type} 说话人分离模型，但无法连接到 https://huggingface.co,可能会失败')

        try:
            self.precent += 3
            title = tr(f'Begin separating the speakers') + f':{speaker_type}'
            spk_list = None
            kw = {
                "input_file": self.cfg.source_wav,
                "subtitles": [[it['start_time'], it['end_time']] for it in self.source_srt_list],
                "num_speakers": self.max_speakers,
                "is_cuda": self.cfg.is_cuda
            }
            if speaker_type == 'built':
                tools.down_file_from_ms(f'{ROOT_DIR}/models/onnx', [
                    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/seg_model.onnx",
                    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/nemo_en_titanet_small.onnx",
                    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"
                ], callback=self._process_callback)
                from videotrans.process.prepare_audio import built_speakers as _run_speakers
                del kw['is_cuda']
                kw['num_speakers'] = -1 if self.max_speakers < 1 else self.max_speakers
                kw['language'] = self.cfg.detect_language
            elif speaker_type == 'ali_CAM':
                tools.check_and_down_ms(model_id='iic/speech_campplus_speaker-diarization_common',
                                        callback=self._process_callback)
                from videotrans.process.prepare_audio import cam_speakers as _run_speakers
            elif speaker_type == 'pyannote':
                from videotrans.process.prepare_audio import pyannote_speakers as _run_speakers
            elif speaker_type == 'reverb':
                from videotrans.process.prepare_audio import reverb_speakers as _run_speakers
            else:
                logger.error(f'当前所选说话人分离模型不支持:{speaker_type=}')
                return
            if speaker_type in ['pyannote', 'reverb']:
                self._signal(text='Downloading speakers models')
                from huggingface_hub import snapshot_download
                print(f'下载 token: {speaker_type},{hf_token=}')
                snapshot_download(
                    repo_id="pyannote/speaker-diarization-3.1" if speaker_type == 'pyannote' else "Revai/reverb-diarization-v1",
                    token=hf_token
                )

            spk_list = self._new_process(callback=_run_speakers, title=title,
                                         is_cuda=self.cfg.is_cuda and speaker_type != 'built', kwargs=kw)

            if spk_list:
                Path(self.cfg.cache_folder + "/speaker.json").write_text(json.dumps(spk_list), encoding='utf-8')
                logger.debug('分离说话人成功完成')
                shutil.copy2(self.cfg.cache_folder + "/speaker.json", self.cfg.target_dir + "/speaker.json")
                self.extract_speaker_refs(spk_list)
                # L2 跨集匹配: diariz 刚完成, spkN_ref.wav 已就绪, 按声纹匹配
                # drama.json 里已有角色, 自动写 spk_to_character.json 供下游 Step5 复用
                self._auto_match_speakers_by_embedding()
            self._signal(text=tr('separating speakers end'))
            self._stage_end("diariz")
        except:
            pass

    def extract_speaker_refs(self, spk_list=None) -> None:
        """
        Step 3/5: 为每个说话人提取参考音频片段
        - 音频文件：cache_folder/spk{N}_ref.wav   （同步复制到 target_dir）
        - 索引文件：cache_folder/speaker_refs.json = {spk_id: {wav, text, start_ms, end_ms}}
          其中 text 为该 speaker 参考区间内所有字幕行拼接的原文，供 CosyVoice 等克隆 TTS 用作 ref_text
        """
        if not self.source_srt_list:
            return
        if spk_list is None:
            spk_json = Path(self.cfg.cache_folder + "/speaker.json")
            if not spk_json.exists():
                return
            spk_list = json.loads(spk_json.read_text(encoding='utf-8'))

        # 按说话人聚合字幕行，记录 (start_ms, end_ms, line_index)
        from collections import defaultdict
        spk_segs: dict = defaultdict(list)
        for i, spk_id in enumerate(spk_list):
            if i >= len(self.source_srt_list):
                break
            seg = self.source_srt_list[i]
            duration_ms = seg['end_time'] - seg['start_time']
            if duration_ms > 0:
                spk_segs[spk_id].append((seg['start_time'], seg['end_time'], i))

        source_audio = self.cfg.source_wav
        if not tools.vail_file(source_audio):
            logger.warning('extract_speaker_refs: source_wav 不存在，跳过参考音频提取')
            return

        MIN_MS = 3000   # 最短3秒
        MAX_MS = 20000  # 最长20秒（火山声音复刻推荐 10-30s；DashScope/CosyVoice 也能吃）

        ref_info = {}  # spk_id → {wav, text, start_ms, end_ms}, 供 Step 5 克隆 TTS 使用
        line_ref_info = {}  # subtitle_line -> local ref info
        total_audio_ms = tools.get_audio_time(source_audio) if tools.vail_file(source_audio) else 0
        try:
            from videotrans.util.prosody_analyzer import analyze_reference_prosody
        except Exception:
            analyze_reference_prosody = None
        for spk_id, segs in spk_segs.items():
            out_path = f"{self.cfg.cache_folder}/{spk_id}_ref.wav"

            # 找最长的连续片段合并 (相邻行间隔 ≤500ms 视为连续)
            best_start, best_end = segs[0][0], segs[0][1]
            cur_start, cur_end = segs[0][0], segs[0][1]
            for start_ms, end_ms, _ in segs[1:]:
                if start_ms - cur_end <= 500:
                    cur_end = end_ms
                else:
                    if cur_end - cur_start > best_end - best_start:
                        best_start, best_end = cur_start, cur_end
                    cur_start, cur_end = start_ms, end_ms
            if cur_end - cur_start > best_end - best_start:
                best_start, best_end = cur_start, cur_end

            # 裁剪到 MIN_MS~MAX_MS
            duration = best_end - best_start
            if duration < MIN_MS:
                # 片段太短，选单个时长最长的字幕行
                best_seg = max(segs, key=lambda x: x[1] - x[0])
                best_start, best_end = best_seg[0], best_seg[1]
            elif duration > MAX_MS:
                best_end = best_start + MAX_MS

            # 收集该区间内所有字幕行文本作为 ref_text
            ref_text_parts = []
            for seg in self.source_srt_list:
                if seg['start_time'] >= best_start and seg['end_time'] <= best_end:
                    t = (seg.get('text') or '').strip()
                    if t:
                        ref_text_parts.append(t)
            ref_text = ''.join(ref_text_parts) if self.cfg.source_language_code[:2].lower() == 'zh' else ' '.join(ref_text_parts)

            ss = tools.ms_to_time_string(ms=best_start)
            to = tools.ms_to_time_string(ms=best_end)

            # 音频文件：如果已存在，跳过重切，但仍刷新 ref_info
            try:
                if not Path(out_path).exists():
                    tools.cut_from_audio(audio_file=source_audio, ss=ss, to=to, out_file=out_path)
                    # 同步复制到 target_dir，方便调试查看（clear_cache 会删除 cache_folder）
                    target_copy = f"{self.cfg.target_dir}/{spk_id}_ref.wav"
                    shutil.copy2(out_path, target_copy)
                ref_info[spk_id] = {
                    'wav': out_path,
                    'text': ref_text,
                    'start_ms': best_start,
                    'end_ms': best_end,
                }
                logger.debug(f'已提取参考音频: {spk_id} → {out_path} [{ss}~{to}] text="{ref_text[:40]}..."')
            except Exception as e:
                logger.warning(f'提取参考音频失败 {spk_id}: {e}')

            # 每句附近同 speaker 参考音频: 优先保留局部语气、停顿和节奏
            for idx, (start_ms, end_ms, line_index) in enumerate(segs):
                if line_index >= len(self.source_srt_list):
                    continue
                cur_seg = self.source_srt_list[line_index]
                local_start, local_end = start_ms, end_ms
                chosen = [(start_ms, end_ms, line_index)]
                left = idx - 1
                right = idx + 1
                while (local_end - local_start) < 4000 and (left >= 0 or right < len(segs)):
                    left_gap = None
                    right_gap = None
                    if left >= 0:
                        left_gap = start_ms - segs[left][1]
                    if right < len(segs):
                        right_gap = segs[right][0] - end_ms
                    pick = None
                    if left_gap is not None and left_gap <= 1800 and (right_gap is None or left_gap <= right_gap):
                        pick = ("left", segs[left])
                        left -= 1
                    elif right_gap is not None and right_gap <= 1800:
                        pick = ("right", segs[right])
                        right += 1
                    else:
                        break
                    _, picked = pick
                    chosen.append(picked)
                    local_start = min(local_start, picked[0])
                    local_end = max(local_end, picked[1])

                if local_end - local_start < 1500:
                    pad = (1500 - (local_end - local_start)) // 2 + 1
                    local_start = max(0, local_start - pad)
                    local_end = min(total_audio_ms or local_end + pad, local_end + pad)
                if local_end - local_start > 5000:
                    center = (start_ms + end_ms) // 2
                    local_start = max(0, center - 2500)
                    local_end = min(total_audio_ms or center + 2500, center + 2500)

                local_path = f"{self.cfg.cache_folder}/{spk_id}_line_{cur_seg['line']}_ref.wav"
                local_ss = tools.ms_to_time_string(ms=local_start)
                local_to = tools.ms_to_time_string(ms=local_end)
                local_text_parts = []
                for _, _, li in sorted(chosen, key=lambda x: x[2]):
                    try:
                        txt = (self.source_srt_list[li].get('text') or '').strip()
                    except Exception:
                        txt = ''
                    if txt:
                        local_text_parts.append(txt)
                local_text = ''.join(local_text_parts) if self.cfg.source_language_code[:2].lower() == 'zh' else ' '.join(local_text_parts)
                try:
                    if not Path(local_path).exists():
                        tools.cut_from_audio(audio_file=source_audio, ss=local_ss, to=local_to, out_file=local_path)
                    prosody = analyze_reference_prosody(local_path, local_text, cur_seg['end_time'] - cur_seg['start_time']) if analyze_reference_prosody else {}
                    line_ref_info[str(cur_seg['line'])] = {
                        'speaker': spk_id,
                        'wav': local_path,
                        'text': local_text,
                        'start_ms': local_start,
                        'end_ms': local_end,
                        'prosody': prosody,
                    }
                except Exception as e:
                    logger.warning(f'提取局部参考音频失败 line={cur_seg.get("line")} {spk_id}: {e}')

        # 保存 speaker_refs.json (Step 5 的入口)
        if ref_info:
            refs_json_path = Path(f"{self.cfg.cache_folder}/speaker_refs.json")
            refs_json_path.write_text(
                json.dumps(ref_info, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            try:
                shutil.copy2(refs_json_path, self.cfg.target_dir + "/speaker_refs.json")
            except Exception:
                pass
        if line_ref_info:
            line_refs_json_path = Path(f"{self.cfg.cache_folder}/speaker_line_refs.json")
            line_refs_json_path.write_text(
                json.dumps(line_ref_info, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            try:
                shutil.copy2(line_refs_json_path, self.cfg.target_dir + "/speaker_line_refs.json")
            except Exception:
                pass

    def _build_speaker_voice_map(self):
        """
        Step 4: 读取 speaker.json 为每个说话人分配不同 TTS 音色。
        返回 (spk_list, spk_voice_map)；任一条件不满足时返回 (None, None)。
        """
        spk_json = Path(self.cfg.cache_folder + "/speaker.json")
        if not spk_json.exists():
            return None, None
        try:
            spk_list = json.loads(spk_json.read_text(encoding='utf-8'))
        except Exception as e:
            logger.warning(f'Step4: 璇诲彇 speaker.json 澶辫触: {e}')
            return None, None
        if not spk_list:
            return None, None

        # 目前只对 Edge-TTS 分配音色池；其他 TTS 类型回退到全局 voice_role
        # (CosyVoice 等克隆型 TTS 在 Step 5 通过 ref_wav 映射实现角色区分)
        if self.cfg.tts_type == QWEN3LOCAL_TTS and self.cfg.voice_role == 'auto-match':
            refs_json = Path(self.cfg.cache_folder + "/speaker_refs.json")
            if not refs_json.exists():
                logger.warning('Step4: auto-match 缺少 speaker_refs.json')
                return None, None
            try:
                fingerprint_path = Path(ROOT_DIR) / "videotrans" / "voicejson" / "qwen3local_emb.json"
                if not fingerprint_path.exists():
                    logger.info('Step4: auto-match 缺少 qwen3local_emb.json，跳过同步指纹预热，直接使用降级匹配')
            except Exception as e:
                logger.warning(f'Step4: auto-match 准备 qwen3local 指纹库失败: {e}')
            try:
                ref_info = json.loads(refs_json.read_text(encoding='utf-8'))
            except Exception as e:
                logger.warning(f'Step4: 璇诲彇 speaker_refs.json 澶辫触: {e}')
                return None, None
            speaker_refs = {}
            for spk_id, data in (ref_info or {}).items():
                if isinstance(data, dict) and data.get('wav') and Path(data['wav']).exists():
                    speaker_refs[spk_id] = data['wav']
            if not speaker_refs:
                logger.warning('Step4: auto-match 无可用 speaker 参考音频')
                return None, None
            try:
                from videotrans.util.voice_matcher import match_voices_to_speakers
                rolelist = tools.get_qwenttslocal_rolelist()
                spk_voice_map = match_voices_to_speakers(
                    speaker_refs=speaker_refs,
                    all_voices=list(rolelist.keys()),
                    tts_type=self.cfg.tts_type,
                )
            except Exception as e:
                logger.warning(f'Step4: auto-match 匹配失败: {e}')
                return None, None
            if not spk_voice_map:
                return None, None
            logger.info(f'Step4 auto-match Qwen3Local Speaker→Voice {spk_voice_map}')
            self._signal(text=f'Speaker 鈫?Voice: {spk_voice_map}')
            return spk_list, spk_voice_map

        if self.cfg.tts_type != EDGE_TTS:
            return None, None

        lang_key = (self.cfg.target_language_code or '')[:2].lower()
        voice_pool = EDGE_VOICE_POOLS.get(lang_key)
        if not voice_pool:
            logger.debug(f'Step4: 当前目标语言 {lang_key} 无预设音色池，跳过说话人映射')
            return None, None

        try:
            spk_list = json.loads(spk_json.read_text(encoding='utf-8'))
        except Exception as e:
            logger.warning(f'Step4: 读取 speaker.json 失败: {e}')
            return None, None
        if not spk_list:
            return None, None

        # 按首次出现顺序取唯一 speaker_id，保证主角（通常先出现且占行多）拿到首位音色
        unique_spks = []
        seen = set()
        for s in spk_list:
            if s not in seen:
                unique_spks.append(s)
                seen.add(s)

        spk_voice_map = {
            spk_id: voice_pool[idx % len(voice_pool)]
            for idx, spk_id in enumerate(unique_spks)
        }
        logger.info(f'Step4 说话人音色映射: {spk_voice_map}')
        # 同时输出到 CLI，方便调试
        self._signal(text=f'Speaker → Voice: {spk_voice_map}')
        return spk_list, spk_voice_map

    def _build_qwen3local_auto_match_map(self):
        spk_json = Path(self.cfg.cache_folder + "/speaker.json")
        refs_json = Path(self.cfg.cache_folder + "/speaker_refs.json")
        if not spk_json.exists() or not refs_json.exists():
            return None, None
        try:
            fingerprint_path = Path(ROOT_DIR) / "videotrans" / "voicejson" / "qwen3local_emb.json"
            if not fingerprint_path.exists():
                logger.info('Step4 auto-match: 缺少 qwen3local_emb.json，跳过同步指纹预热，直接使用降级匹配')
        except Exception as e:
            logger.warning(f'Step4 auto-match 准备 qwen3local 指纹库失败: {e}')
        try:
            spk_list = json.loads(spk_json.read_text(encoding='utf-8'))
            ref_info = json.loads(refs_json.read_text(encoding='utf-8'))
        except Exception as e:
            logger.warning(f'Step4 auto-match 璇绘枃浠跺け璐?: {e}')
            return None, None
        if not spk_list or not ref_info:
            return None, None
        speaker_refs = {}
        for spk_id, data in (ref_info or {}).items():
            if isinstance(data, dict) and data.get('wav') and Path(data['wav']).exists():
                speaker_refs[spk_id] = data['wav']
        if not speaker_refs:
            return None, None
        try:
            from videotrans.util.voice_matcher import match_voices_to_speakers_verbose
            rolelist = tools.get_qwenttslocal_rolelist()
            matched = match_voices_to_speakers_verbose(
                speaker_refs=speaker_refs,
                all_voices=list(rolelist.keys()),
                tts_type=self.cfg.tts_type,
            )
            spk_voice_map = matched.get('mapping', {})
            detail_path = Path(self.cfg.cache_folder + "/auto_match_detail.json")
            detail_path.write_text(
                json.dumps(matched.get('details', {}), ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            logger.warning(f'Step4 auto-match 鍖归厤澶辫触: {e}')
            return None, None
        if not spk_voice_map:
            return None, None
        logger.info(f'Step4 auto-match Qwen3Local Speaker→Voice {spk_voice_map}')
        # 把每个 spk 的匹配依据 (embedding/gender/round_robin + 分数) 送进任务 log, 便于排错
        details = matched.get('details', {}) or {}
        for spk_id, info in details.items():
            method = info.get('method', '?')
            voice = info.get('voice', '?')
            score = info.get('score')
            gender = info.get('gender')
            extra = []
            if score is not None:
                extra.append(f'score={score}')
            if gender:
                extra.append(f'gender={gender}')
            suffix = f' ({", ".join(extra)})' if extra else ''
            self._signal(text=f'  {spk_id} -> {voice} [{method}]{suffix}')
        self._signal(text=f'Speaker -> Voice: {spk_voice_map}')
        return spk_list, spk_voice_map

    def _build_qwentts_auto_match_map(self):
        spk_json = Path(self.cfg.cache_folder + "/speaker.json")
        refs_json = Path(self.cfg.cache_folder + "/speaker_refs.json")
        if not spk_json.exists() or not refs_json.exists():
            return None, None
        try:
            spk_list = json.loads(spk_json.read_text(encoding='utf-8'))
            ref_info = json.loads(refs_json.read_text(encoding='utf-8'))
        except Exception as e:
            logger.warning(f'Step4 qwentts auto-match read failed: {e}')
            return None, None
        if not spk_list or not ref_info:
            return None, None

        speaker_refs = {}
        for spk_id, data in (ref_info or {}).items():
            if isinstance(data, dict) and data.get('wav') and Path(data['wav']).exists():
                speaker_refs[spk_id] = data['wav']
        if not speaker_refs:
            return None, None

        try:
            from videotrans.util.voice_matcher import match_voices_to_speakers_verbose
            rolelist = tools.get_qwen3tts_rolelist()
            available_voices = list(dict.fromkeys(
                voice for voice in rolelist.values()
                if voice not in ('No', 'auto-match', 'clone', '')
            ))
            matched = match_voices_to_speakers_verbose(
                speaker_refs=speaker_refs,
                all_voices=available_voices,
                tts_type=self.cfg.tts_type,
            )
            spk_voice_map = matched.get('mapping', {})
            detail_path = Path(self.cfg.cache_folder + "/auto_match_detail_qwentts.json")
            detail_path.write_text(
                json.dumps(matched.get('details', {}), ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            logger.warning(f'Step4 qwentts auto-match failed: {e}')
            return None, None

        if not spk_voice_map:
            return None, None
        logger.info(f'Step4 auto-match QwenTTS Speaker→Voice {spk_voice_map}')
        self._signal(text=f'Speaker → Voice: {spk_voice_map}')
        return spk_list, spk_voice_map

    def _build_speaker_ref_map(self):
        """
        Step 5: 读取 speaker_refs.json 构造说话人→克隆参考信息的映射。
        返回 (spk_list, spk_ref_map)；任一条件不满足时返回 (None, None)。
        spk_ref_map[spk_id] = {'wav': path, 'text': ref_text, ...}
        """
        # 只对支持克隆的 TTS 渠道生效
        if self.cfg.tts_type not in SUPPORT_CLONE:
            return None, None
        spk_json = Path(self.cfg.cache_folder + "/speaker.json")
        refs_json = Path(self.cfg.cache_folder + "/speaker_refs.json")
        if not spk_json.exists() or not refs_json.exists():
            return None, None
        try:
            spk_list = json.loads(spk_json.read_text(encoding='utf-8'))
            spk_ref_map = json.loads(refs_json.read_text(encoding='utf-8'))
        except Exception as e:
            logger.warning(f'Step5: 读取 speaker_refs 失败: {e}')
            return None, None
        if not spk_list or not spk_ref_map:
            return None, None
        # 过滤 wav 不存在的条目
        valid_map = {
            k: v for k, v in spk_ref_map.items()
            if isinstance(v, dict) and v.get('wav') and Path(v['wav']).exists()
        }
        if not valid_map:
            return None, None

        # M3: 声音克隆库接管 —— 若字幕对话框已写入 spk_to_character.json,
        # 按角色名走 voice_library:
        #   a) 已有角色 → 复用库里的 ref.wav (跨集复用, 命中即走)
        #   b) 新角色 → 用本集 spkN_ref.wav 作为素材扩充入库, 然后替换 ref_wav
        # 匿名说话人 (未在映射里) 完全走原路径, 不入库
        self._apply_voice_library(valid_map)

        summary = ', '.join(f'{k}→{Path(v["wav"]).name}' for k, v in valid_map.items())
        logger.info(f'Step5 克隆参考音频映射: {summary}')
        self._signal(text=f'Speaker → Clone Ref: {summary}')
        return spk_list, valid_map

    def _auto_match_speakers_by_embedding(self) -> None:
        """L2 核心: diariz 完成后按声纹相似度把本集 spk_id 匹配到 drama.json 已录角色。

        产出 {cache_folder}/spk_to_character.json (若命中任何角色),
        下游 _apply_voice_library 会按这个映射复用 ref.wav / fixed_voice。

        命中规则:
          - 余弦相似度 >= 阈值 (默认 0.70) → 自动匹配
          - 多 spk 匹到同一 character: 取分最高的 spk, 其他留空
          - 未命中的 spk 完全不写 (等 UI 对话框让用户手动命名 → 同时录声纹)

        失败静默降级, 不阻塞主流程。批量模式 (无 UI) 下这是唯一的跨集链路。
        """
        try:
            from videotrans.util.voice_library import get_drama_dir, list_embeddings
            from videotrans.util.speaker_embedding import match_best

            refs_json = Path(self.cfg.cache_folder + "/speaker_refs.json")
            if not refs_json.exists() or not self.cfg.name:
                return
            ref_info = json.loads(refs_json.read_text(encoding='utf-8'))
            if not ref_info:
                return

            drama_dir = get_drama_dir(self.cfg.name)
            candidates = list_embeddings(drama_dir)
            if not candidates:
                # 首次入库, 无候选声纹, 跳过自动匹配 (依赖用户手动命名录声纹)
                logger.info(f'[voice_library] {drama_dir.name}: 无候选声纹, 跳过自动匹配 (首集)')
                return

            # 逐 spk 提取声纹 → 与 drama.json 候选求最相似
            from videotrans.util.speaker_embedding import compute_embedding

            # 收集 (spk_id, match_name, score) 用于冲突去重
            hits: list = []
            for spk_id, info in ref_info.items():
                wav = info.get('wav', '')
                if not wav or not Path(wav).exists():
                    continue
                emb = compute_embedding(wav)
                if not emb:
                    continue
                m = match_best(emb, candidates)
                if m is None:
                    continue
                name, score = m
                hits.append((spk_id, name, score))
                logger.info(f'[voice_library] 匹配 {spk_id} → "{name}" (score={score:.3f})')

            if not hits:
                logger.info(f'[voice_library] {drama_dir.name}: 本集无角色通过阈值')
                return

            # 冲突去重: 同一 character 名下只保留分最高的 spk
            # (避免两个 spk 都指到同一角色, 后者会覆盖前者的库写入)
            best_per_name: dict = {}
            for spk_id, name, score in hits:
                prev = best_per_name.get(name)
                if prev is None or score > prev[1]:
                    best_per_name[name] = (spk_id, score)

            mapping = {spk_id: name for name, (spk_id, _) in best_per_name.items()}

            mapping_path = Path(f'{self.cfg.cache_folder}/spk_to_character.json')
            # 若 UI 已写过 (不太可能, diariz 在对话框前), 不覆盖用户操作
            if mapping_path.exists():
                try:
                    existing = json.loads(mapping_path.read_text(encoding='utf-8'))
                    if existing.get('mapping'):
                        return
                except Exception:
                    pass

            mapping_path.write_text(json.dumps({
                'drama_dir': str(drama_dir),
                'mapping': mapping,
                'auto_matched': True,
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            logger.info(f'[voice_library] 自动匹配写入 {len(mapping)} 条: {mapping_path}')
            self._signal(text=f'[VoiceLib] auto-matched: {mapping}')
        except Exception as e:
            logger.warning(f'[voice_library] 自动声纹匹配失败, 降级: {e}')

    def _apply_voice_library(self, valid_map: dict) -> None:
        """根据 spk_to_character.json 用库里的 ref.wav 替换 valid_map 中的 wav 路径。

        失败静默降级 —— 库系统完全可选, 任何异常都不该阻断主流程。
        """
        try:
            mapping_path = Path(self.cfg.cache_folder + "/spk_to_character.json")
            if not mapping_path.exists():
                return
            data = json.loads(mapping_path.read_text(encoding='utf-8'))
            drama_dir_str = data.get('drama_dir', '')
            mapping = data.get('mapping', {}) or {}
            if not drama_dir_str or not mapping:
                return

            from videotrans.util.voice_library import (
                get_ref_path, add_or_extend_character, pick_top_segments,
            )
            drama_dir = Path(drama_dir_str)
            if not drama_dir.exists():
                return

            episode_id = Path(self.cfg.name).stem if self.cfg.name else drama_dir.name

            for spk_id, character_name in mapping.items():
                if spk_id not in valid_map:
                    continue
                ref_entry = valid_map[spk_id]
                src_wav = ref_entry.get('wav', '')
                if not src_wav or not Path(src_wav).exists():
                    continue

                # (a) 复用: 库里已有该角色 → 直接换 ref.wav 路径
                lib_ref = get_ref_path(drama_dir, character_name)
                if lib_ref is not None:
                    ref_entry['wav'] = str(lib_ref)
                    logger.info(f'[voice_library] {spk_id} 复用角色 "{character_name}" → {lib_ref}')
                    self._signal(text=f'[VoiceLib] {spk_id} reuse "{character_name}"')
                    continue

                # (b) 新角色: 本集 spkN_ref.wav 通常 3-20s, 作为首次素材入库
                try:
                    import subprocess
                    dur_out = subprocess.run(
                        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                         '-of', 'default=noprint_wrappers=1:nokey=1', src_wav],
                        capture_output=True, text=True, timeout=10,
                    )
                    dur = float((dur_out.stdout or '0').strip() or 0.0)
                except Exception:
                    dur = 10.0
                if dur < 1.5:
                    continue

                segs = pick_top_segments([(0.0, dur)])
                new_ref = add_or_extend_character(
                    drama_dir=drama_dir,
                    character_name=character_name,
                    source_wav=src_wav,
                    segments=segs,
                    episode_id=episode_id,
                    ref_text=ref_entry.get('text', ''),
                )
                if new_ref is not None:
                    ref_entry['wav'] = str(new_ref)
                    logger.info(f'[voice_library] {spk_id} 新建角色 "{character_name}" → {new_ref}')
                    self._signal(text=f'[VoiceLib] {spk_id} new "{character_name}"')
        except Exception as e:
            logger.warning(f'[voice_library] _apply_voice_library 失败, 降级使用原 ref_wav: {e}')

    # 翻译字幕文件
    def trans(self) -> None:
        if self._exit(): return
        if not self.shoud_trans: return
        self._stage_start("trans")
        self.precent += 3
        self._signal(text=tr('starttrans'))

        # 如果存在目标语言字幕，无需继续翻译，前台直接使用该字幕替换
        if self._srt_vail(self.cfg.target_sub):
            self._signal(
                text=Path(self.cfg.target_sub).read_text(encoding="utf-8", errors="ignore"),
                type='replace_subtitle'
            )
            self._stage_end("trans")
            return
        try:
            rawsrt = tools.get_subtitle_from_srt(self.cfg.source_sub, is_file=True)
            self._signal(text=tr('kaishitiquhefanyi'))

            target_srt = run_trans(
                translate_type=self.cfg.translate_type,
                text_list=copy.deepcopy(rawsrt),
                uuid=self.uuid,
                source_code=self.cfg.source_language_code,
                target_code=self.cfg.target_language_code
            )
            if self._exit():
                return
            # 一一核对每条字幕
            target_srt = self._check_target_sub(rawsrt, target_srt)

            # 仅提取，并且双语输出
            if self.cfg.app_mode == 'tiqu' and self.cfg.output_srt > 0 and self.cfg.source_language_code != self.cfg.target_language_code:
                _source_srt_len = len(rawsrt)
                for i, it in enumerate(target_srt):
                    if i < _source_srt_len and self.cfg.output_srt == 1:
                        # 目标语言在下
                        it['text'] = ("\n".join([rawsrt[i]['text'].strip(), it['text'].strip()])).strip()
                    elif i < _source_srt_len and self.cfg.output_srt == 2:
                        it['text'] = ("\n".join([it['text'].strip(), rawsrt[i]['text'].strip()])).strip()

            self._save_srt_target(target_srt, self.cfg.target_sub)

            if self.cfg.app_mode == 'tiqu':
                _output_file = f"{self.cfg.target_dir}/{self.cfg.noextname}.srt"
                if self.cfg.copysrt_rawvideo:
                    p = Path(self.cfg.name)
                    _output_file = f'{p.parent.as_posix()}/{p.stem}.srt'
                if not Path(_output_file).exists() or not Path(_output_file).samefile(Path(self.cfg.target_sub)):
                    shutil.copy2(self.cfg.target_sub, _output_file)
                    self._del_sub()

                self.hasend = True
                self.precent = 100
        except Exception as e:
            self.hasend = True
            raise
        self._signal(text=tr('endtrans'))
        self._stage_end("trans")

    def _del_sub(self):
        try:
            Path(self.cfg.source_sub).unlink(missing_ok=True)
            Path(self.cfg.target_sub).unlink(missing_ok=True)
        except:
            pass

    # 对字幕进行配音
    def dubbing(self) -> None:
        if self._exit():
            return
        if self.cfg.app_mode == 'tiqu' or not self.shoud_dubbing:
            return

        self._stage_start("dubbing")
        self._signal(text=tr('kaishipeiyin'))
        self.precent += 3
        try:
            self._tts()
            # 判断下一步重新调整字幕
        except Exception as e:
            self.hasend = True
            raise
        self._signal(text=tr('The dubbing is finished'))
        self._stage_end("dubbing")

    # 音画字幕对齐
    def align(self) -> None:
        if self._exit():
            return
        if self.cfg.app_mode == 'tiqu' or not self.shoud_dubbing or self.ignore_align:
            return

        self._stage_start("align")
        self._signal(text=tr('duiqicaozuo'))
        self.precent += 3
        if self.cfg.voice_autorate or self.cfg.video_autorate:
            self._signal(text=tr("Sound & video speed alignment stage"))
        try:
            # 需要视频慢速，则判断无声视频是否已分离完毕
            if self.cfg.video_autorate:
                tools.is_novoice_mp4(self.cfg.novoice_mp4, self.uuid)
            # 存在视频，则以视频长度为准
            if tools.vail_file(self.cfg.novoice_mp4):
                self.video_time = tools.get_video_duration(self.cfg.novoice_mp4)

            from videotrans.task._rate import SpeedRate
            rate_inst = SpeedRate(
                queue_tts=self.queue_tts,
                uuid=self.uuid,
                shoud_audiorate=self.cfg.voice_autorate,
                # 视频是否需慢速，需要时对 novoice_mp4进行处理
                shoud_videorate=self.cfg.video_autorate if not self.is_audio_trans else False,
                novoice_mp4=self.cfg.novoice_mp4 if not self.is_audio_trans else None,
                # 原始总时长
                raw_total_time=self.video_time,

                target_audio=self.cfg.target_wav,
                cache_folder=self.cfg.cache_folder,
                align_sub_audio=self.cfg.align_sub_audio,  # 均在未启用音频加速和视频慢速时才起作用
                remove_silent_mid=self.cfg.remove_silent_mid  # 均在未启用音频加速和视频慢速时才起作用
            )
            self.queue_tts = rate_inst.run()
            # 慢速处理后，更新新视频总时长，用于音视频对齐
            if tools.vail_file(self.cfg.novoice_mp4):
                self.video_time = tools.get_video_duration(self.cfg.novoice_mp4)

            # 对齐字幕
            if self.cfg.voice_autorate or self.cfg.video_autorate or self.cfg.align_sub_audio:
                srt = ""
                for (idx, it) in enumerate(self.queue_tts):
                    startraw = tools.ms_to_time_string(ms=it['start_time'])
                    endraw = tools.ms_to_time_string(ms=it['end_time'])
                    srt += f"{idx + 1}\n{startraw} --> {endraw}\n{it['text']}\n\n"
                # 字幕保存到目标文件夹
                with  Path(self.cfg.target_sub).open('w', encoding="utf-8") as f:
                    f.write(srt.strip())
        except Exception as e:
            self.hasend = True
            raise

        # 成功后，如果存在 音量，则调节音量
        if self.cfg.tts_type not in [EDGE_TTS, AZURE_TTS] and self.cfg.volume != '+0%' and tools.vail_file(
                self.cfg.target_wav):
            volume = self.cfg.volume.replace('%', '').strip()
            try:
                volume = 1 + float(volume) / 100
                if volume != 1.0:
                    tmp_name = self.cfg.cache_folder + f'/volume-{volume}-{Path(self.cfg.target_wav).name}'
                    tools.runffmpeg(['-y', '-i', os.path.basename(self.cfg.target_wav), '-af', f"volume={volume}",
                                     os.path.basename(tmp_name)], cmd_dir=self.cfg.cache_folder)
                    shutil.copy2(tmp_name, self.cfg.target_wav)
            except:
                pass

        self._signal(text=tr('Alignment phase complete, awaiting the next step'))
        self._stage_end("align")

    # 将 视频、音频、字幕合成
    def assembling(self) -> None:
        if self._exit(): return
        # 音频翻译， 提取模式 无需合并
        if self.is_audio_trans or self.cfg.app_mode == 'tiqu' or not self.shoud_hebing:
            return
        self._stage_start("assembling")
        if self.precent < 95:
            self.precent += 3
        self._signal(text=tr('kaishihebing'))
        try:
            self._join_video_audio_srt()
        except Exception as e:
            self.hasend = True
            raise
        self._stage_end("assembling")

    # 收尾，根据 output和 linshi_output是否相同，不相同，则移动
    def task_done(self) -> None:
        # 正常完成仍是 ing，手动停止变为 stop
        if self._exit(): return
        self._stage_start("task_done")
        self.precent = 99

        # 提取时，删除
        if self.cfg.app_mode == 'tiqu':
            try:
                Path(f"{self.cfg.target_dir}/{self.cfg.source_language_code}.srt").unlink(
                    missing_ok=True)
                Path(f"{self.cfg.target_dir}/{self.cfg.target_language_code}.srt").unlink(
                    missing_ok=True)
            except:
                pass  # 忽略删除失败
        else:    
            if self.is_audio_trans and tools.vail_file(self.cfg.target_wav):
                try:
                    shutil.copy2(self.cfg.target_wav, f"{self.cfg.target_dir}/{self.cfg.target_language_code}-{self.cfg.noextname}.wav")
                except shutil.SameFileError:
                    pass

            try:
                if self.cfg.shound_del_name:
                    Path(self.cfg.shound_del_name).unlink(missing_ok=True)
                if self.cfg.only_out_mp4:
                    shutil.move(self.cfg.targetdir_mp4, Path(self.cfg.target_dir).parent / f'{self.cfg.noextname}.mp4')
                    shutil.rmtree(self.cfg.target_dir, ignore_errors=True)
            except Exception as e:
                logger.exception(e, exc_info=True)
        self.hasend = True
        self.precent = 100
        self.task_finished_at = time.time()
        self._stage_end("task_done")
        self._finalize_task_logging(
            status="succeed",
            extra={
                "target_mp4": self.cfg.targetdir_mp4,
                "target_wav_output": self.cfg.target_wav_output,
                "target_sub": self.cfg.target_sub,
                "subtitle_type": self.cfg.subtitle_type,
                "tts_type": self.cfg.tts_type,
            }
        )
        try:
            shutil.rmtree(self.cfg.cache_folder, ignore_errors=True)
        except:
            pass
        payload = self._task_summary_payload(
            status="succeed",
            extra={
                "target_mp4": self.cfg.targetdir_mp4,
                "target_wav_output": self.cfg.target_wav_output,
                "target_sub": self.cfg.target_sub,
                "subtitle_type": self.cfg.subtitle_type,
                "tts_type": self.cfg.tts_type,
            }
        )
        payload["name"] = self.cfg.name
        self._signal(text=json.dumps(payload, ensure_ascii=False), type='succeed')
        tools.send_notification(tr('Succeed'), f"{self.cfg.basename}")

    # 从原始视频分离出 无声视频
    def _split_novoice_byraw(self):
        cmd = [
            "-y",
            "-fflags",
            "+genpts",
            "-i",
            self.cfg.name,
            "-an",
            "-c:v",
            "copy" if self.is_copy_video else f"libx264"
        ]
        _name=os.path.basename(self.cfg.novoice_mp4)
        enc_qua=[] if self.is_copy_video else ['-crf','18']
        if self.is_copy_video or settings.get('force_lib'):            
            return tools.runffmpeg(cmd+enc_qua+[_name], noextname=self.uuid, cmd_dir=self.cfg.cache_folder)
        
        try:
            hw_decode_args,_,vcodec,enc_args=self._get_hard_cfg(codec="264")
            cmd = [
                "-y",
                "-fflags",
                "+genpts",
                "-i",
            ]
            cmd+=hw_decode_args
            
            cmd+=[
                self.cfg.name,
                "-an",
                "-c:v",
                vcodec,
                _name
            ]
            self._subprocess(cmd)
        except:
            return tools.runffmpeg(cmd+enc_qua+[_name], noextname=self.uuid, cmd_dir=self.cfg.cache_folder)
            

    # 从原始视频中分离出音频
    def _split_audio_byraw(self, is_separate=False):
        cmd = [
            "-y",
            "-i",
            self.cfg.name,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            self.cfg.source_wav
        ]
        rs = tools.runffmpeg(cmd)
        if not is_separate:
            return rs

        # 继续人声分离
        tmpfile = self.cfg.cache_folder + "/441000_ac2_raw.wav"
        tools.runffmpeg([
            "-y",
            "-i",
            self.cfg.name,
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            tmpfile
        ])

        if tools.vail_file(self.cfg.vocal) and tools.vail_file(self.cfg.instrument):
            return
        title = tr('Separating vocals and background music, which may take a longer time')
        uvr_models=settings.get('uvr_models')
        tools.down_file_from_ms(f'{ROOT_DIR}/models/onnx', [
            f"https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/{uvr_models}.onnx"
        ], callback=self._process_callback)
        from videotrans.process.prepare_audio import vocal_bgm
        # 返回 False None 失败
        kw = {"input_file": tmpfile, "vocal_file": self.cfg.vocal, "instr_file": self.cfg.instrument,"uvr_models":uvr_models}
        try:
            rs = self._new_process(callback=vocal_bgm, title=title, is_cuda=False, kwargs=kw)
            if rs and tools.vail_file(self.cfg.vocal) and tools.vail_file(self.cfg.instrument):
                cmd = [
                    "-y",
                    "-i",
                    self.cfg.vocal,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    '-af',
                    "volume=1.5",
                    self.cfg.source_wav
                ]
                tools.runffmpeg(cmd)
                shutil.copy2(self.cfg.vocal, f'{self.cfg.target_dir}/vocal.wav')
                shutil.copy2(self.cfg.instrument, f'{self.cfg.target_dir}/instrument.wav')
        except Exception as e:
            logger.exception(f'人声背景声分离失败：{e}', exc_info=True)

    # 配音预处理，去掉无效字符，整理开始时间
    def _tts(self, daz_json=None) -> None:
        queue_tts = []
        subs = tools.get_subtitle_from_srt(self.cfg.target_sub)
        source_subs = tools.get_subtitle_from_srt(self.cfg.source_sub)
        if len(subs) < 1:
            raise RuntimeError(f"SRT file error:{self.cfg.target_sub}")
        try:
            rate = int(str(self.cfg.voice_rate).replace('%', ''))
        except:
            rate = 0
        if rate >= 0:
            rate = f"+{rate}%"
        else:
            rate = f"{rate}%"
        # 取出设置的每行角色
        line_roles = app_cfg.line_roles
        voice_role = self.cfg.voice_role
        if voice_role == 'auto-match' and self.cfg.tts_type == QWEN_TTS:
            voice_role = params.get('qwentts_role', '') or 'Chelsie'
        if voice_role == 'auto-match' and self.cfg.tts_type == QWEN3LOCAL_TTS:
            voice_role = 'Vivian'
        line_ref_map = {}
        line_refs_path = Path(self.cfg.cache_folder + "/speaker_line_refs.json")
        if line_refs_path.exists():
            try:
                line_ref_map = json.loads(line_refs_path.read_text(encoding='utf-8'))
            except Exception as e:
                logger.warning(f'读取 speaker_line_refs.json 失败: {e}')

        # Step 4: 非克隆 TTS，按说话人分配不同音色
        spk_list_v, spk_voice_map = self._build_speaker_voice_map()
        if self.cfg.tts_type == QWEN3LOCAL_TTS and self.cfg.voice_role == 'auto-match':
            spk_list_v, spk_voice_map = self._build_qwen3local_auto_match_map()
        elif self.cfg.tts_type == QWEN_TTS and self.cfg.voice_role == 'auto-match':
            spk_list_v, spk_voice_map = self._build_qwentts_auto_match_map()
        # Step 5: 克隆型 TTS，按说话人绑定克隆参考音频
        spk_list_c, spk_ref_map = self._build_speaker_ref_map()
        # 两者 tts_type 互斥，最多一个生效
        spk_list = spk_list_c or spk_list_v

        # Voice 选择优先级: line_roles (手动) > spk_voice_map (Step4) > 'clone' (Step5) > voice_role (全局)
        # 取出每一条字幕，行号\n开始时间 --> 结束时间\n内容
        for i, it in enumerate(subs):
            if it['end_time'] < it['start_time'] or not it['text'].strip():
                continue
            line_key = f'{it["line"]}'
            if line_key in line_roles:
                voice = line_roles[line_key]
            elif spk_voice_map and spk_list and i < len(spk_list):
                voice = spk_voice_map.get(spk_list[i], voice_role)
            elif spk_ref_map and spk_list and i < len(spk_list):
                # Step5 克隆模式：voice='clone' 触发下方 ref_wav 逻辑
                voice = voice_role if self.cfg.tts_type == QWEN_TTS else 'clone'
            else:
                voice = voice_role

            tmp_dict = {
                "text": it['text'],
                "line": it['line'],
                "start_time": it['start_time'],
                "end_time": it['end_time'],
                "startraw": it['startraw'],
                "endraw": it['endraw'],
                "ref_text": source_subs[i]['text'] if source_subs and i < len(source_subs) else '',
                "start_time_source": source_subs[i]['start_time'] if source_subs and i < len(source_subs) else it[
                    'start_time'],
                "end_time_source": source_subs[i]['end_time'] if source_subs and i < len(source_subs) else it[
                    'end_time'],
                "role": voice,
                "rate": rate,
                "volume": self.cfg.volume,
                "pitch": self.cfg.pitch,
                "tts_type": self.cfg.tts_type,
                "filename": f"{self.cfg.cache_folder}/dubb-{i}.wav"
            }
            local_ref = line_ref_map.get(line_key) if isinstance(line_ref_map, dict) else None
            if local_ref:
                tmp_dict['prosody'] = local_ref.get('prosody', {})
                suggested_rate = (local_ref.get('prosody') or {}).get('suggested_rate')
                if isinstance(suggested_rate, int) and suggested_rate:
                    tmp_dict['rate'] = f"+{suggested_rate}%" if suggested_rate >= 0 else f"{suggested_rate}%"
                    logger.debug(f"[Prosody] Line={line_key} suggested_rate={tmp_dict['rate']} style={tmp_dict['prosody'].get('style_tags')}")
                logger.debug(
                    f"[LocalRef] Line={line_key} speaker={local_ref.get('speaker')} "
                    f"ref={Path(local_ref.get('wav', '')).name if local_ref.get('wav') else ''} "
                    f"window={local_ref.get('start_ms')}->{local_ref.get('end_ms')}"
                )
            # 克隆类型 TTS: 设置 ref_wav
            if voice in ('clone', 'auto-match') and self.cfg.tts_type in SUPPORT_CLONE:
                # Step 5 优先: 按 speaker 复用 extract_speaker_refs 提前提取好的参考音频
                ref = None
                if spk_ref_map and spk_list and i < len(spk_list):
                    ref = spk_ref_map.get(spk_list[i])
                if local_ref and local_ref.get('wav') and Path(local_ref['wav']).exists():
                    tmp_dict['ref_wav'] = local_ref['wav']
                    if local_ref.get('text'):
                        tmp_dict['ref_text'] = local_ref['text']
                elif ref and Path(ref['wav']).exists():
                    tmp_dict['ref_wav'] = ref['wav']
                    # ref_text 用 speaker 参考片段的合并原文（即 ref_wav 对应文本），而非当前字幕行
                    if ref.get('text'):
                        tmp_dict['ref_text'] = ref['text']
                else:
                    # 回退到按字幕行逐个截取
                    tmp_dict['ref_wav'] = f"{self.cfg.cache_folder}/clone-{i}.wav"
                tmp_dict['ref_language'] = self.cfg.detect_language[:2]
            queue_tts.append(tmp_dict)

        self.queue_tts = copy.deepcopy(queue_tts)

        if not self.queue_tts or len(self.queue_tts) < 1:
            raise RuntimeError(f'Queue tts length is 0')

        # 如果存在有 ref_wav 即需要clone，存在参考音频的
        if len([it.get("ref_wav") for it in self.queue_tts if it.get("ref_wav")]) > 0:
            self._create_ref_from_vocal()

        # 具体配音操作
        run_tts(
            queue_tts=self.queue_tts,
            language=self.cfg.target_language_code,
            uuid=self.uuid,
            tts_type=self.cfg.tts_type,
            is_cuda=self.cfg.is_cuda
        )
        if settings.get('save_segment_audio', False):
            outname = self.cfg.target_dir + f'/segment_audio_{self.cfg.noextname}'
            Path(outname).mkdir(parents=True, exist_ok=True)
            for it in self.queue_tts:
                text = re.sub(r'["\'*?\\/\|:<>\r\n\t]+', '', it['text'], flags=re.I | re.S)
                name = f'{outname}/{it["line"]}-{text[:60]}.wav'
                if Path(it['filename']).exists():
                    shutil.copy2(it['filename'], name)

    # 多线程实现裁剪参考音频
    def _create_ref_from_vocal(self):
        # 背景分离人声如果失败则直接使用原始音频
        vocal = self.cfg.source_wav

        # 裁切对应片段为参考音频
        def _cutaudio_from_vocal(it):
            try:
                # Step 5 兼容: 若 ref_wav 文件已存在（例如预先提取的 spkN_ref.wav），
                # 不再按字幕行时间覆盖，避免破坏说话人级参考音频
                if Path(it['ref_wav']).exists():
                    return
                logger.debug(f"裁切对应片段为参考音频：{it['startraw']}->{it['endraw']}\n当前{it=}")
                tools.cut_from_audio(
                    audio_file=vocal,
                    ss=it['startraw'],
                    to=it['endraw'],
                    out_file=it['ref_wav']
                )
            except Exception as e:
                logger.exception(f'裁切参考音频失败:{e}', exc_info=True)

        all_task = []
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(12, len(self.queue_tts), os.cpu_count())) as pool:
            for item in self.queue_tts:
                if item.get('ref_wav'):
                    all_task.append(pool.submit(_cutaudio_from_vocal, item))
            if len(all_task) > 0:
                _ = [i.result() for i in all_task]

    # 添加背景音乐
    def _back_music(self) -> None:
        if self._exit() or not self.shoud_dubbing:
            return

        if not tools.vail_file(self.cfg.target_wav) or not tools.vail_file(self.cfg.background_music):
            return
        try:
            self._signal(text=tr("Adding background audio"))
            # 获取视频长度
            vtime = tools.get_audio_time(self.cfg.target_wav)
            # 获取背景音频长度
            atime = tools.get_audio_time(self.cfg.background_music)
            bgm_file = self.cfg.cache_folder + f'/bgm_file.wav'
            self.convert_to_wav(self.cfg.background_music, bgm_file)
            self.cfg.background_music = bgm_file
            beishu = math.ceil(vtime / atime)
            if settings.get('loop_backaudio') and beishu > 1 and vtime - 1000 > atime:
                # 获取延长片段
                file_list = [self.cfg.background_music for n in range(beishu + 1)]
                concat_txt = self.cfg.cache_folder + f'/{time.time()}.txt'
                tools.create_concat_txt(file_list, concat_txt=concat_txt)
                tools.concat_multi_audio(
                    concat_txt=concat_txt,
                    out=self.cfg.cache_folder + "/bgm_file_extend.wav")
                self.cfg.background_music = self.cfg.cache_folder + "/bgm_file_extend.wav"
            # 背景音频降低音量
            tools.runffmpeg(
                ['-y',
                 '-i', self.cfg.background_music,
                 "-filter:a", f"volume={settings.get('backaudio_volume', 0.8)}",
                 '-c:a', 'pcm_s16le',
                 self.cfg.cache_folder + f"/bgm_file_extend_volume.wav"
                 ])
            # 背景音频和配音合并
            cmd = ['-y',
                   '-i', os.path.basename(self.cfg.target_wav),
                   '-i', "bgm_file_extend_volume.wav",
                   '-filter_complex', "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2",
                   '-ac', '2',
                   '-c:a', 'pcm_s16le',
                   "lastend.wav"
                   ]
            tools.runffmpeg(cmd, cmd_dir=self.cfg.cache_folder)
            self.cfg.target_wav = self.cfg.cache_folder + f"/lastend.wav"
        except Exception as e:
            logger.exception(f'添加背景音乐失败:{str(e)}', exc_info=True)

    def _separate(self) -> None:
        if self._exit() or not self.shoud_separate or not self.cfg.embed_bgm:
            return
        # 如果背景音频分离失败，则静默返回
        if not tools.vail_file(self.cfg.instrument):
            return
        if not tools.vail_file(self.cfg.target_wav):
            return
        try:
            self._signal(text=tr("Re-embedded background sounds"))
            vtime = tools.get_audio_time(self.cfg.target_wav)
            atime = tools.get_audio_time(self.cfg.instrument)
            beishu = math.ceil(vtime / atime)

            instrument_file = self.cfg.instrument
            logger.debug(f'合并背景音 {beishu=},{atime=},{vtime=}')
            if atime + 1000 < vtime:
                if int(settings.get('loop_backaudio'))==1:
                    # 背景音连接延长片段
                    file_list = [instrument_file for n in range(beishu + 1)]
                    concat_txt = self.cfg.cache_folder + f'/{time.time()}.txt'
                    tools.create_concat_txt(file_list, concat_txt=concat_txt)
                    tools.concat_multi_audio(concat_txt=concat_txt,
                                             out=self.cfg.cache_folder + "/instrument-concat.wav")
                else:
                    # 延长背景音
                    tools.change_speed_rubberband(instrument_file, self.cfg.cache_folder + f"/instrument-concat.wav", vtime)
                instrument_file=self.cfg.cache_folder + f"/instrument-concat.wav"
            # 背景音合并配音
            self._backandvocal(instrument_file, self.cfg.target_wav)
        except Exception as e:
            logger.exception(e, exc_info=True)

    # 合并后最后文件仍为 人声文件，时长需要等于人声
    def _backandvocal(self, backwav, peiyinm4a):
        backwav = Path(backwav).as_posix()
        tmpdir = self.cfg.cache_folder
        tmpwav = Path(tmpdir + f'/{time.time()}-1.wav').as_posix()
        tmpm4a = Path(tmpdir + f'/{time.time()}.wav').as_posix()
        # 背景转为m4a文件,音量降低为0.8
        self.convert_to_wav(backwav, tmpm4a, ["-filter:a", f"volume={settings.get('backaudio_volume', 0.8)}"])
        tools.runffmpeg(['-y', '-i', os.path.basename(peiyinm4a), '-i', os.path.basename(tmpm4a), '-filter_complex',
                         "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2", '-ac', '2', "-b:a", "128k",
                         '-c:a', 'pcm_s16le', os.path.basename(tmpwav)], cmd_dir=self.cfg.cache_folder)
        shutil.copy2(tmpwav, peiyinm4a)

    def _normalize_dubbing_audio(self):
        if self._exit() or not self.shoud_dubbing or not tools.vail_file(self.cfg.target_wav):
            return
        if settings.get('normalize_dubbing_audio', True) is False:
            return
        target_i = settings.get('dubbing_loudnorm_i', -16)
        target_lra = settings.get('dubbing_loudnorm_lra', 11)
        target_tp = settings.get('dubbing_loudnorm_tp', -1.5)
        tmpwav = Path(self.cfg.cache_folder + f'/{time.time()}-loudnorm.wav').as_posix()
        try:
            logger.debug(
                f"[Audio-Norm] loudnorm target={self.cfg.target_wav} "
                f"I={target_i} LRA={target_lra} TP={target_tp}"
            )
            tools.runffmpeg([
                '-y',
                '-i', os.path.basename(self.cfg.target_wav),
                '-af', f'loudnorm=I={target_i}:LRA={target_lra}:TP={target_tp}',
                '-ar', '48000',
                '-ac', '2',
                '-c:a', 'pcm_s16le',
                os.path.basename(tmpwav)
            ], cmd_dir=self.cfg.cache_folder, force_cpu=True)
            if tools.vail_file(tmpwav):
                shutil.copy2(tmpwav, self.cfg.target_wav)
        except Exception as e:
            logger.warning(f'[Audio-Norm] loudnorm failed, keep original dubbing audio: {e}')
        finally:
            Path(tmpwav).unlink(missing_ok=True)

    # 处理所需字幕
    def _process_subtitles(self) -> Union[tuple[str, str], None]:
        logger.debug(f"\n======准备要嵌入的字幕:{self.cfg.subtitle_type=}=====")
        if not Path(self.cfg.target_sub).exists():
            logger.error(tr("No valid subtitle file exists"))
            return

        # 如果原始语言和目标语言相同，或不存原始语言字幕，则强制单字幕
        if not Path(self.cfg.source_sub).exists() or (self.cfg.source_language_code == self.cfg.target_language_code):
            if self.cfg.subtitle_type == 3:
                self.cfg.subtitle_type = 1
            elif self.cfg.subtitle_type == 4:
                self.cfg.subtitle_type = 2

        process_end_subtitle = self.cfg.cache_folder + f'/end.srt'
        # 单行字符数
        maxlen = int(
            settings.get('cjk_len', 15) if self.cfg.target_language_code[:2] in ["zh", "ja", "jp", "ko",
                                                                                        'yu'] else
            settings.get('other_len', 60))
        target_sub_list = tools.get_subtitle_from_srt(self.cfg.target_sub)
        
        srt_string = ""
        # 双硬字幕时的两种语言字幕分割符，用于定义不同样式
        _join_flag=''
        # 双硬 双软字幕组装
        if self.cfg.subtitle_type in [3, 4]:
            source_sub_list = tools.get_subtitle_from_srt(self.cfg.source_sub)
            source_length = len(source_sub_list)
            # 原语言单行字符长度
            source_maxlen = int(
                settings.get('cjk_len', 15) if self.cfg.source_language_code[:2] in ["zh", "ja", "jp", "ko",
                                                                                            'yu'] else
                settings.get('other_len', 60))

            # 双语字幕
            # 判断 双硬字幕 and 存在 ass.json 文件 and  (Bottom_Fontsize != Fontsize or PrimaryColour!=Bottom_PrimaryColour) 需要 对双语字幕的第2行设置不同颜色和尺寸
            _join_flag=self._get_join_flag()
                    
            for i, it in enumerate(target_sub_list):
                # 换行
                _text = tools.simple_wrap(it['text'].strip(), maxlen, self.cfg.target_language_code)
                srt_string += f"{it['line']}\n{it['time']}\n"
                if source_length > 0 and i < source_length:
                    _text_source=tools.simple_wrap(source_sub_list[i]['text'], source_maxlen, self.cfg.source_language_code)
                    _text=f'{_text_source}\n{_join_flag}{_text}' if self.cfg.output_srt==1 else f'{_text}\n{_join_flag}{_text_source}'
                srt_string += f"{_text}\n\n"
            srt_string=srt_string.strip()
            process_end_subtitle = f"{self.cfg.cache_folder}/shuang.srt"
            Path(process_end_subtitle).write_text(srt_string, encoding='utf-8')
            Path(self.cfg.target_dir + "/shuang.srt").write_text(srt_string.replace('###','') if _join_flag=='###' else srt_string, encoding='utf-8')
        else:
            # 单字幕，需处理字符数换行
            for i, it in enumerate(target_sub_list):
                tmp = tools.simple_wrap(it['text'].strip(), maxlen, self.cfg.target_language_code)
                srt_string += f"{it['line']}\n{it['time']}\n{tmp.strip()}\n\n"
            with Path(process_end_subtitle).open('w', encoding='utf-8') as f:
                f.write(srt_string)

        # 目标字幕语言
        subtitle_langcode = translator.get_subtitle_code(show_target=self.cfg.target_language)
        logger.debug(
            f'最终确定字幕嵌入类型:{self.cfg.subtitle_type} ,目标字幕语言:{subtitle_langcode}, 字幕文件:{process_end_subtitle}\n')
        # 单软 或双软
        if self.cfg.subtitle_type in [2, 4]:
            return os.path.basename(process_end_subtitle), subtitle_langcode

        # 硬字幕转为ass格式 并设置样式
        process_end_subtitle_ass = tools.set_ass_font(process_end_subtitle)
        basename = os.path.basename(process_end_subtitle_ass)
        return basename, subtitle_langcode


    def _get_join_flag(self):
        _join_flag=""
        if self.cfg.subtitle_type!=3 or not Path(f'{ROOT_DIR}/videotrans/ass.json').exists():
            return _join_flag
        try:
            assjson=json.loads(Path(f'{ROOT_DIR}/videotrans/ass.json').read_text(encoding='utf-8'))
        except Exception as e:
            logger.debug(f'读取 ass.json 错误，忽略:{e}')
            return _join_flag
        else:
            for k,v in assjson.items():
                if k.startswith('Bottom_') and v!= assjson.get(k[7:]):
                    _join_flag='###'
                    break
        return _join_flag



    # 视频定格最后一帧
    def _video_extend(self, duration_ms=1000):
        sec = duration_ms / 1000.0
        final_video_path = Path(f'{self.cfg.cache_folder}/final_video_with_freeze_lastend.mp4').as_posix()
        cmd = ['-y', '-i', os.path.basename(self.cfg.novoice_mp4),
               '-vf', f'tpad=stop_mode=clone:stop_duration={sec:.3f}',
               '-c:v', 'libx264',
               '-crf', f'{settings.get("crf", 23)}',
               '-preset', settings.get('preset', 'veryfast'),
               '-an', 'final_video_with_freeze_lastend.mp4']

        if tools.runffmpeg(cmd, force_cpu=True, cmd_dir=self.cfg.cache_folder) and Path(final_video_path).exists():
            shutil.copy2(final_video_path, self.cfg.novoice_mp4)
            logger.debug(f"视频定格应延长{duration_ms}ms，实际向上取整秒延长{sec}s,操作成功。")
        else:
            logger.warning("视频定格延长操作失败！")

    # 最终合成视频
    def _join_video_audio_srt(self) -> None:
        if self._exit():
            return
        if not self.shoud_hebing:
            return True

        # 判断 novoice_mp4 是否完成
        tools.is_novoice_mp4(self.cfg.novoice_mp4, self.uuid)
        if not Path(self.cfg.novoice_mp4).exists():
            raise RuntimeError(f'{self.cfg.novoice_mp4} 不存在')
        
        # 需要配音但没有配音文件
        if self.shoud_dubbing and not tools.vail_file(self.cfg.target_wav):
            raise RuntimeError(f"{tr('Dubbing')}{tr('anerror')}:{self.cfg.target_wav}")

        self.precent = min(max(90, self.precent), 95)


        target_m4a = self.cfg.cache_folder + "/origin_audio.m4a"
        # 用于判断输出原始音频是否结束，is True是结束，
        output_source_output = True
        if not self.shoud_dubbing:
            # 无配音的使用原始音频
            self._get_origin_audio(target_m4a)
            shutil.copy2(target_m4a, self.cfg.source_wav_output)
        else:
            try:
                output_source_output = False
                # 高质量 原始音频输出到目标目录，单独线程执行，不影响继续运行
                cmd = [
                    "-y",
                    "-i",
                    self.cfg.name,
                    "-vn",
                    "-b:a", "128k",
                    "-c:a",
                    "aac",
                    self.cfg.source_wav_output
                ]

                def _output():
                    nonlocal output_source_output
                    try:
                        tools.runffmpeg(cmd)
                    except Exception:
                        pass
                    finally:
                        output_source_output = True
                threading.Thread(target=_output, daemon=True).start()
            except Exception:
                pass

            # 添加背景音乐
            self._normalize_dubbing_audio()
            self._back_music()
            # 重新嵌入分离出的背景音
            self._separate()
            
            tools.runffmpeg([
                "-y",
                "-i",
                os.path.basename(self.cfg.target_wav),
                "-ac", "2", "-b:a", "128k", "-c:a", "aac",
                os.path.basename(target_m4a)
            ], cmd_dir=self.cfg.cache_folder)
            shutil.copy2(target_m4a, self.cfg.target_wav_output)

        self.precent = min(max(95, self.precent), 98)
        
        
        # 处理所需字幕
        subtitles_file, subtitle_langcode = None, None
        if self.cfg.subtitle_type > 0:
            subtitles_file, subtitle_langcode = self._process_subtitles()

        # 字幕嵌入时进入视频目录下
        os.chdir(self.cfg.cache_folder)

        # 末尾对齐
        duration_ms = int(tools.get_video_duration(self.cfg.novoice_mp4))
        duration_s = f'{duration_ms / 1000.0:.6f}'
        audio_ms = tools.get_audio_time(target_m4a)
        if duration_ms < audio_ms:
            self._video_extend(audio_ms - duration_ms)
            duration_ms = int(tools.get_video_duration(self.cfg.novoice_mp4))
            duration_s = f'{duration_ms / 1000.0:.6f}'

        # 将生成的视频先导出到临时目录，防止包含各种奇怪符号的targetdir_mp4导致ffmpeg失败
        tmp_target_mp4 = self.cfg.cache_folder + f"/laste_target.mp4"
        self._signal(text=tr("Video + Subtitles + Dubbing in merge"))

        try:
            protxt = self.cfg.cache_folder + f"/compose{time.time()}.txt"
            protxt_basename = os.path.basename(protxt)
            threading.Thread(target=self._hebing_pro, args=(protxt, self.video_time), daemon=True).start()
            
            # 如果需要输出的视频是 264 编码，因开始和中间编码均为264，可以考虑使用copy (如果无硬字幕嵌入的话)
            is_copy_mode = (str(self.video_codec_num) == '264')
            # 无音频视频流
            novoice_mp4_basename = os.path.basename(self.cfg.novoice_mp4)
            # 需要嵌入的音频
            target_m4a_basename = os.path.basename(target_m4a)
            # 合成后的结果视频
            tmp_target_mp4_basename = os.path.basename(tmp_target_mp4)

            # 获取可用的硬件
            if not app_cfg.video_codec:
                app_cfg.video_codec = tools.get_video_codec()                        


            cmd0 = [
                "-y",
                "-progress",
                protxt_basename
            ]
            
            cmd1=[
                "-i",
                novoice_mp4_basename,
                "-i",
                target_m4a_basename
            ]
            enc_qua=['-crf', f'{settings.get("crf", 23)}','-preset', settings.get('preset','medium')]
            
            # 无字幕 或 软字幕
            if self.cfg.subtitle_type not in [1,3]:               
                #软字幕
                if self.cfg.subtitle_type in [2, 4]:
                    cmd1.extend(["-i",subtitles_file])               
                cmd1.extend([
                    '-map', 
                    '0:v',
                    '-map', 
                    '1:a'
                ])
                if self.cfg.subtitle_type in [2, 4]:
                    cmd1.extend(['-map', '2:s'])
                
                cmd1.extend([
                    "-c:v",
                    f"libx{self.video_codec_num}",
                    "-c:a",
                    "copy",
                ])
                if self.cfg.subtitle_type in [2, 4]:
                    cmd1.extend([
                        "-c:s",
                        "mov_text",
                        "-metadata:s:s:0",
                        f"language={subtitle_langcode}"
                    ])
                
                cmd2=[
                    "-movflags",
                    "+faststart",
                ]
                if self.cfg.video_autorate:
                    cmd2.extend(["-fps_mode", "vfr"])
                
                cmd2.extend(["-t", str(duration_s),  tmp_target_mp4_basename])
                if is_copy_mode:
                    cmd1[cmd1.index('-c:v')+1]='copy'
                    logger.debug(f'[最终视频合成]copy模式，无需重新编码:\n{cmd0+cmd1+cmd2}')
                    tools.runffmpeg(cmd0+cmd1+cmd2, cmd_dir=self.cfg.cache_folder, force_cpu=True)
                elif app_cfg.video_codec.startswith('libx') or settings.get('force_lib'):
                    # 不支持硬件编码的就无需尝试硬件了
                    logger.debug(f'[最终视频合成]不支持硬件编码或指定了强制软编解码:\n{cmd0+cmd1+cmd2}')
                    tools.runffmpeg(cmd0+cmd1+enc_qua+cmd2, cmd_dir=self.cfg.cache_folder, force_cpu=True)                    
                else:
                    # 尝试使用硬件编解码
                    hw_decode_args,_,vcodec,enc_args=self._get_hard_cfg()
                    cmd1[cmd1.index('-c:v')+1]=vcodec
                    # 如果硬件处理失败，回退软编
                    try:
                        self._subprocess(cmd0+hw_decode_args+cmd1+enc_args+cmd2)
                    except:
                        cmd1[cmd1.index('-c:v')+1]=f'libx{self.video_codec_num}'
                        logger.warning(f'硬件处理视频合成失败，回退软编')
                        tools.runffmpeg(cmd0+cmd1+enc_qua+cmd2, cmd_dir=self.cfg.cache_folder, force_cpu=True)
                   
            # 硬字幕
            else:
                cmd1.append('-filter_complex')          
                subtitle_filter=[f"[0:v]subtitles=filename='{subtitles_file}'[v_out]"]
                cmd2=[
                    "-map", 
                    "[v_out]",
                    "-map", 
                    "1:a",
                    "-c:v",
                    f'libx{self.video_codec_num}',
                    '-c:a',
                    'copy',
                ]                 
                cmd3=["-movflags", "+faststart"]
                
                if self.cfg.video_autorate:
                    cmd3.extend(["-fps_mode", "vfr"])
                    
                cmd3.extend(["-t", str(duration_s), tmp_target_mp4_basename])
                if app_cfg.video_codec.startswith('libx')  or settings.get('force_lib'):
                    logger.debug(f'[最终视频合成]不支持硬件编解码或指定了强制软编解码:\n{cmd0+cmd1+cmd2}')
                    tools.runffmpeg(cmd0+cmd1+subtitle_filter+cmd2+enc_qua+cmd3, cmd_dir=self.cfg.cache_folder, force_cpu=True)
                else:
                    # 如果硬件处理失败，回退软编
                    try:
                        hw_decode_args,vf_string,vcodec,enc_args=self._get_hard_cfg(subtitles_file)
                        cmd2[cmd2.index('-c:v')+1]=vcodec
                        self._subprocess(cmd0+hw_decode_args+cmd1+[vf_string]+cmd2+enc_args+cmd3)
                    except:
                        cmd2[cmd2.index('-c:v')+1]=f'libx{self.video_codec_num}'
                        logger.warning(f'硬件处理视频合成失败，回退软编')
                        tools.runffmpeg(cmd0+cmd1+subtitle_filter+cmd2+enc_qua+cmd3, cmd_dir=self.cfg.cache_folder, force_cpu=True)
        except Exception as e:
            msg = tr('Error in embedding the final step of the subtitle dubbing')
            raise RuntimeError(msg)

        os.chdir(ROOT_DIR)
        if Path(tmp_target_mp4).exists():
            try:
                shutil.copy2(tmp_target_mp4, self.cfg.targetdir_mp4)
            except Exception as e:
                raise RuntimeError(tr('Translation successful but transfer failed. ', tmp_target_mp4))

        self._create_txt()
        time.sleep(1)
        # 有可能输出原始音频到目标文件夹的程序仍在执行，但不影响
        while output_source_output is not True:
            print(f'{output_source_output=}')
            time.sleep(1)
        return True

    def _get_origin_audio(self, output):
        # 无需配音的场景下取出原始音频
        if self.video_info.get('streams_audio', 0) == 0:
            # 无音频流
            return
        cmd = [
            "-y",
            "-i",
            self.cfg.name,
            "-vn"
        ]
        if self.video_info['audio_codec_name'] == 'aac':
            cmd.extend(['-c:a', 'copy'])
        else:
            cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        cmd.append(output)
        return tools.runffmpeg(cmd)

    # ffmpeg进度日志
    def _hebing_pro(self, protxt, video_time=0) -> None:
        while 1:
            if app_cfg.exit_soft or self.hasend or self.precent >= 100: return

            content = tools.read_last_n_lines(protxt)
            if not content:
                time.sleep(0.5)
                continue

            if content[-1] == 'progress=end':
                return
            idx = len(content) - 1
            end_time = "00:00:00"
            while idx > 0:
                if content[idx].startswith('out_time='):
                    end_time = content[idx].split('=')[1].strip()
                    break
                idx -= 1
            self._signal(text=tr('kaishihebing') + f' {end_time}')
            time.sleep(0.5)

    # 创建说明txt
    def _create_txt(self) -> None:
        try:

            with open(self.cfg.target_dir + f'/{tr("readme")}.txt',
                      'w', encoding="utf-8", errors="ignore") as f:
                f.write(f"""以下是可能生成的全部文件, 根据执行时配置的选项不同, 某些文件可能不会生成, 之所以生成这些文件和素材，是为了方便有需要的用户, 进一步使用其他软件进行处理, 而不必再进行语音导出、音视频分离、字幕识别等重复工作

        *.mp4 = 最终完成的目标视频文件
        {self.cfg.source_language_code}.m4a = 原始视频中的音频文件
        {self.cfg.target_language_code}.m4a = 配音后的音频文件
        removed_noise.wav = 降噪后的原始音频文件
        {self.cfg.source_language_code}.srt = 原始视频中根据声音识别出的字幕文件
        {self.cfg.target_language_code}.srt = 翻译为目标语言后字幕文件
        speaker.json = 说话人标志
        -Noxxx.srt = 未进行重新断句之前的字幕
        shuang.srt = 双语字幕
        vocal.wav = 原始视频中分离出的人声音频文件
        instrument.wav = 原始视频中分离出的背景音乐音频文件


        如果觉得该项目对你有价值，并希望该项目能一直稳定持续维护，欢迎各位小额赞助，有了一定资金支持，我将能够持续投入更多时间和精力
        捐助地址：https://pvt9.com/about

        ====

        Here are the descriptions of all possible files that might exist. Depending on the configuration options when executing, some files may not be generated.

        *.mp4 = The final completed target video file
        {self.cfg.source_language_code}.m4a = The audio file in the original video
        {self.cfg.target_language_code}.m4a = dubbing audio
        removed_noise.wav = original video after removed noise
        {self.cfg.source_language_code}.srt = Subtitles recognized in the original video
        {self.cfg.target_language_code}.srt = Subtitles translated into the target language
        shuang.srt = Source language and target language subtitles srt 
        vocal.wav = The vocal audio file separated from the original video
        instrument.wav = The background music audio file separated from the original video


        If you feel that this project is valuable to you and hope that it can be maintained consistently, we welcome small sponsorships. With some financial support, I will be able to continue to invest more time and energy
        Donation address: https://ko-fi.com/jianchang512


        ====

        Github: https://github.com/jianchang512/pyvideotrans
        Docs: https://pvt9.com

                        """)
        except:
            pass





    # 视频合成时，返回可用的硬件解码参数、字幕嵌入参数、视频编码参数、质量相关参数
    def _get_hard_cfg(self, subtitles_file=None,codec=None):
        os_name = platform.system()
        # 仅用于确定编码器部分，具体 264或265由 codec 决定
        hw_type=app_cfg.video_codec
        logger.debug(f'原始{hw_type=}')
        
        if '_' in hw_type:
            _hw_type_list = hw_type.lower().split('_')
            if _hw_type_list[0]=='vaapi':
                hw_type='vaapi'
            else:
                hw_type=_hw_type_list[1]
        
        
        logger.debug(f'整理后{hw_type=}')
        
        
        # 硬字幕由于是软过滤，必须先在内存中压制。
        # 不同的硬件编码器可能需要在软过滤后，将画面重新上传到显存（hwupload）
        
        # 默认回退为软编码
        codec=f'{self.video_codec_num}' if not codec else codec
        vcodec = f"libx{codec}"
        _crf=f'{settings.get("crf", 23)}'

        # 全局参数，硬件解码相关
        global_args = []
        # 硬字幕嵌入参数，软字幕忽略
        vf_string = f"[0:v]subtitles=filename='{subtitles_file}'[v_out]"
        
        # 硬件兼容有限，防止出错
        _preset=settings.get('preset','medium')
        if 'fast' in _preset:
            _preset='fast'
        elif 'slow' in _preset:
            _preset='slow'
        
        if _preset not in ['fast','slow','medium']:
            _preset='medium'
        enc_args = ['-crf', _crf,'-preset', _preset]
        
        
        # --- 参数映射表 ---
        PRESET_MAP = {
            # NVENC: p1 (最快) - p7 (最慢/质量最好)
            'nvenc': {'fast': 'p2', 'medium': 'p4', 'slow': 'p7'}, 
            # QSV: veryfast, faster, fast, medium, slow, slower, veryslow
            'qsv': {'fast': 'fast', 'medium': 'medium', 'slow': 'slow'},
            # AMF: speed, balanced, quality
            'amf': {'fast': 'speed', 'medium': 'balanced', 'slow': 'quality'},
            # VAAPI: 通常也接受 standard presets
            'vaapi': {'fast': 'fast', 'medium': 'medium', 'slow': 'slow'},
            # VideoToolbox: 通常不支持 -preset 参数，留空以跳过处理
            'videotoolbox': None 
        }
        
        # --- Nvidia (NVENC) ---
        if hw_type in ['nvenc']:
            vcodec = "h264_nvenc" if codec == '264' else "hevc_nvenc"
            # nvenc 使用 -cq (Constant Quality) 替代 crf，p4 预设在速度和质量间平衡较好
            enc_args = ['-cq', _crf, '-preset', PRESET_MAP.get('nvenc').get(_preset,'p4')]
            # 优先硬件解码
            if settings.get('hw_decode'):
                global_args=['-hwaccel','cuda','-hwaccel_output_format', 'cuda']
                vf_string = f"[0:v]hwdownload,format=nv12,subtitles=filename='{subtitles_file}',hwupload_cuda[v_out]"
            else:
                vf_string = f"[0:v]subtitles=filename='{subtitles_file}'[v_out]"

            return global_args,vf_string,vcodec,enc_args
        # --- Mac (VideoToolbox) ---
        if hw_type in ['videotoolbox']:
            vcodec = "h264_videotoolbox" if codec == '264' else "hevc_videotoolbox"
            # videotoolbox 质量控制，通常用 -q:v (范围约在 40-60 之间视觉无损)
            quality = 100 - (int(_crf) * 1.4)
            enc_args = ['-q:v', f'{int(max(1, min(quality, 100)))}']
            return global_args,vf_string,vcodec,enc_args
            

        # --- Intel (QSV) & AMD (AMF) ---
        if hw_type in ['qsv', 'amf', 'vaapi']:
            if os_name == 'Linux':
                # 【Linux 特殊处理】
                # 在 Linux 下，Intel 和 AMD 开源驱动通常统一走 VAAPI 接口
                devices = glob.glob('/dev/dri/renderD*')
                device= devices[0] if devices else '/dev/dri/renderD128'
                if settings.get('hw_decode'):
                    global_args = ['-hwaccel', 'vaapi', '-hwaccel_device', device, '-hwaccel_output_format', 'vaapi']
                    vf_string = f"[0:v]hwdownload,format=nv12,subtitles=filename='{subtitles_file}',format=nv12,hwupload[v_out]"                
                else:
                    global_args = [
                        '-init_hw_device', f'vaapi=vaapi:{device}'
                    ]
                    vf_string = f"[0:v]subtitles=filename='{subtitles_file}',format=nv12,hwupload[v_out]"                
                vcodec = "h264_vaapi" if codec == '264' else "hevc_vaapi"
                enc_args = ['-qp', _crf,'-preset', PRESET_MAP.get('vaapi').get(_preset,'medium')]
                return global_args,vf_string,vcodec,enc_args
                # VAAPI 要求在软滤镜（字幕）处理完后，转换像素格式并上传到显存
            
            # Windows 环境
            if hw_type in ['qsv']:
                vcodec = "h264_qsv" if codec == '264' else "hevc_qsv"
                # QSV 使用 ICQ 模式 (Intelligent Constant Quality)
                enc_args = ['-global_quality', _crf, '-preset', PRESET_MAP.get('qsv').get(_preset,'medium')]
            else:
                vcodec = "h264_amf" if codec == '264' else "hevc_amf"
                # AMF 使用恒定质量参数 (CQP)
                enc_args = ['-rc', 'cqp', '-qp_p', _crf, '-qp_i', _crf, '-quality', PRESET_MAP.get('amf').get(_preset,'balanced')]
            return global_args,vf_string,vcodec,enc_args
        
        return global_args,vf_string,vcodec,enc_args

    
    def _subprocess(self,cmd):
        logger.debug(f'[尝试硬件编解码执行命令]\n{" ".join(cmd)}\n')
        try:
            creationflags = 0
            if sys.platform == 'win32':
                creationflags = subprocess.CREATE_NO_WINDOW
            if app_cfg.exit_soft:
                return
            cmd.insert(0,"ffmpeg")
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors='replace',
                check=True,
                text=True,
                creationflags=creationflags,
                cwd=self.cfg.cache_folder
            )
            return True
        except subprocess.CalledProcessError as e:
            error_message = e.stderr or ""
            logger.error(f"尝试使用硬件执行命令出错[CalledProcessError]:{error_message}")
            raise
        except Exception as e:
            logger.error(f"尝试使用硬件执行命令出错[Exception]:{e}")
            raise




    # todo 尝试在有硬字幕参与时，强制使用指定的硬件加速
    def _hard_subtitle_use_hw(self,hw_type: str, codec: str, video_in: str, audio_in: str, subtitles_file: str, output_file: str,progress:str,duration_s:str):
        

        """
        生成兼容多硬件编码的 FFmpeg 命令
        :param hw_type: 编码类型，可选值: 为具体的编码器名称， h264_nvenc等 'nvenc'(Nvidia), 'qsv'(Intel), 'amf'(AMD), 'videotoolbox'(Mac), 'software'(软编)
        :param codec: 编码格式，可选值: '264', '265'
        :param video_in: 无声视频路径
        :param audio_in: 音频路径
        :param subtitles_file: 字幕文件路径
        :param output_file: 输出文件路径
        """
        import glob
        os_name = platform.system()
        logger.debug(f'原始{hw_type=}')
        if '_' in hw_type:
            _hw_type_list = hw_type.lower().split('_')
            if _hw_type_list[0]=='vaapi':
                hw_type='vaapi'
            else:
                hw_type=_hw_type_list[1]

        
        logger.debug(f'整理后{hw_type=}')
        # 基础命令（全局参数，例如硬件设备初始化）
        global_args = []
        
        # 基础输入和映射参数
        base_cmd = [
            "-y",
            "-progress", 
            f"{progress}",
            "-i", video_in,
            "-i", audio_in,
        ]
        
        # 字幕由于是软过滤，必须先在内存中压制。
        # 不同的硬件编码器可能需要在软过滤后，将画面重新上传到显存（hwupload）
        vf_string = f"[0:v]subtitles=filename='{subtitles_file}'[v_out]"
        
        # 默认回退为软编码
        vcodec = f"libx{codec}"
        _crf=f'{settings.get("crf", 23)}'
        enc_args = ['-crf', _crf,'-preset', 'fast']
        # 组合最终命令
        final_cmd = ['ffmpeg',"-hide_banner", "-ignore_unknown"]
        # --- Nvidia (NVENC) ---
        if hw_type in ['nvenc']:
            vcodec = "h264_nvenc" if codec == '264' else "hevc_nvenc"
            # nvenc 使用 -cq (Constant Quality) 替代 crf，p4 预设在速度和质量间平衡较好
            enc_args = ['-cq', _crf, '-preset', 'p4']
            # 优先硬件解码
            if settings.get('hw_decode'):
                global_args=['-hwaccel','cuda','-hwaccel_output_format', 'cuda']
            vf_string = f"[0:v]format=nv12,subtitles=filename='{subtitles_file}',hwupload_cuda[v_out]"


        # --- Mac (VideoToolbox) ---
        elif hw_type in ['videotoolbox']:
            vcodec = "h264_videotoolbox" if codec == '264' else "hevc_videotoolbox"
            # videotoolbox 质量控制，通常用 -q:v (范围约在 40-60 之间视觉无损)
            enc_args = ['-q:v', '50']

        # --- Intel (QSV) & AMD (AMF) ---
        elif hw_type in ['qsv', 'amf', 'vaapi']:
            if os_name == 'Linux':

                # 【Linux 特殊处理】
                # 在 Linux 下，Intel 和 AMD 开源驱动通常统一走 VAAPI 接口
                if settings.get('hw_decode'):
                    devices = glob.glob('/dev/dri/renderD*')
                    device= devices[0] if devices else '/dev/dri/renderD128'
                    global_args = ['-hwaccel', 'vaapi', '-hwaccel_device', device, '-hwaccel_output_format', 'vaapi']
                vf_string = f"[0:v]subtitles=filename='{subtitles_file}',format=nv12,hwupload[v_out]"                
                vcodec = "h264_vaapi" if codec == '264' else "hevc_vaapi"
                enc_args = ['-qp', _crf]
                # VAAPI 要求在软滤镜（字幕）处理完后，转换像素格式并上传到显存
            else: # Windows 环境
                if hw_type in ['qsv']:
                    vcodec = "h264_qsv" if codec == '264' else "hevc_qsv"
                    # QSV 使用 ICQ 模式 (Intelligent Constant Quality)
                    enc_args = ['-global_quality', _crf, '-preset', 'fast']
                else:
                    vcodec = "h264_amf" if codec == '264' else "hevc_amf"
                    # AMF 使用恒定质量参数 (CQP)
                    enc_args = ['-rc', 'cqp', '-qp_p', _crf, '-qp_i', _crf, '-quality', 'balanced']


        
        # 全局参数必须在输入文件 -i 之前
        if global_args:
            final_cmd.extend(global_args)
            
        final_cmd.extend(base_cmd)
        
        # 视频过滤器和编码器
        final_cmd.extend(["-filter_complex", vf_string])
        final_cmd.extend(["-map", "[v_out]","-map", "1:a"])
        
        final_cmd.extend([
            "-c:v", vcodec,
            '-c:a','copy'
        ])
        
        # 编码器特定参数
        final_cmd.extend(enc_args)
        
        # 结尾参数
        final_cmd.extend([
            "-movflags", 
            "+faststart",
            "-fps_mode", 
            "vfr",
            "-t",f'{duration_s}',
            output_file
        ])
        logger.debug(f'硬字幕合成时，优先硬件编码命令:{final_cmd}')
        creationflags = 0
        if os_name == 'Windows':
            creationflags = subprocess.CREATE_NO_WINDOW
        
        print(f'{final_cmd=}')
        subprocess.run(
            final_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors='replace',
            check=True,
            text=True,
            creationflags=creationflags,
            cwd=self.cfg.cache_folder
        )
        
        return True
