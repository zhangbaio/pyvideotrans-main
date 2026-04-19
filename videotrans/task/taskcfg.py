from dataclasses import dataclass


@dataclass
class TaskCfgBase:
    is_cuda: bool = False
    uuid: str = None
    cache_folder: str = None
    target_dir: str = None
    source_language: str = None
    source_language_code: str = None
    source_sub: str = None
    source_wav: str = None
    source_wav_output: str = None
    target_language: str = None
    target_language_code: str = None
    target_sub: str = None
    target_wav: str = None
    target_wav_output: str = None
    name: str = None
    noextname: str = None
    basename: str = None
    ext: str = None
    dirname: str = None
    shound_del_name: str = None


@dataclass
class TaskCfgSTT(TaskCfgBase):
    detect_language: str = None
    recogn_type: int = None
    model_name: str = None
    shibie_audio: str = None
    remove_noise: bool = False
    enable_diariz: bool = False
    nums_diariz: int = 0
    rephrase: int = 2
    fix_punc: bool = False


@dataclass
class TaskCfgTTS(TaskCfgBase):
    tts_type: int = None
    volume: str = "+0%"
    pitch: str = "+0Hz"
    voice_rate: str = "+0%"
    voice_role: str = None
    voice_autorate: bool = False
    video_autorate: bool = False
    remove_silent_mid: bool = False
    align_sub_audio: bool = True


@dataclass
class TaskCfgSTS(TaskCfgBase):
    translate_type: int = None


@dataclass
class TaskCfgVTT(TaskCfgSTT, TaskCfgTTS, TaskCfgSTS):
    replace_voice_only: bool = False
    subtitle_language: str = None
    app_mode: str = "biaozhun"
    subtitles: str = ""
    targetdir_mp4: str = None
    novoice_mp4: str = None
    is_separate: bool = False
    embed_bgm: bool = True
    instrument: str = None
    vocal: str = None
    back_audio: str = None
    clear_cache: bool = False
    background_music: str = None
    subtitle_type: int = 0
    only_out_mp4: bool = False
    recogn2pass: bool = False
    output_srt: int = 0
    copysrt_rawvideo: bool = False
    speaker_voice_overrides: str = ""
    remove_hardsub_before_subtitle: bool = False
    vsr_install_path: str = ""
    vsr_sub_area: str = "auto"
    vsr_inpaint_mode: str = "sttn_auto"
    vsr_timeout_sec: int = 3600
    vsr_fail_policy: str = "stop"
    enable_lipsync: bool = False
    lipsync_engine: str = "musetalk"
    lipsync_model_root: str = ""
    lipsync_python: str = ""
    lipsync_ffmpeg_dir: str = ""
    lipsync_version: str = "v15"
    lipsync_batch_size: int = 4
    lipsync_bbox_shift: int = 0
    lipsync_extra_margin: int = 10
    lipsync_audio_padding_length_left: int = 2
    lipsync_audio_padding_length_right: int = 2
    lipsync_use_fp16: bool = True
    lipsync_parsing_mode: str = "jaw"
    lipsync_left_cheek_width: int = 90
    lipsync_right_cheek_width: int = 90
