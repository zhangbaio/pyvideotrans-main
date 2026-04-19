import asyncio
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal


_TEST_AUDIO_TEXT = "\u4f60\u597d\uff0c\u8fd9\u662f\u8bed\u97f3\u8bc6\u522b\u6d4b\u8bd5\u3002"
_TEST_AUDIO_VOICE = "zh-CN-XiaoxiaoNeural"

class TestSTT(QThread):
    uito = Signal(str)

    def __init__(self, *, parent=None, recogn_type=0, model_name=''):
        super().__init__(parent=parent)
        self.recogn_type = recogn_type
        self.model_name = model_name

    async def _save_edge_tts(self, mp3_file):
        from edge_tts import Communicate

        communicate = Communicate(
            _TEST_AUDIO_TEXT,
            voice=_TEST_AUDIO_VOICE,
            rate="+0%",
            volume="+0%",
            connect_timeout=10,
        )
        await asyncio.wait_for(communicate.save(str(mp3_file)), timeout=30)

    def _create_test_audio_by_edge(self, wav_file, tools):
        mp3_file = wav_file.with_suffix(".mp3")
        asyncio.run(self._save_edge_tts(mp3_file))
        if not tools.vail_file(str(mp3_file)):
            return False
        tools.runffmpeg([
            "-y",
            "-i",
            str(mp3_file),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(wav_file),
        ], force_cpu=True)
        return tools.vail_file(str(wav_file))

    def _create_test_audio_by_sapi(self, wav_file):
        if sys.platform != "win32":
            return False

        script = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $voice = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'zh*' } | Select-Object -First 1
    if ($voice -ne $null) {
        $synth.SelectVoice($voice.VoiceInfo.Name)
    }
    $synth.SetOutputToWaveFile($args[0])
    $synth.Speak($args[1])
}
finally {
    $synth.Dispose()
}
"""
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
                str(wav_file),
                _TEST_AUDIO_TEXT,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return completed.returncode == 0 and wav_file.exists() and wav_file.stat().st_size > 0

    def _get_test_audio_file(self, *, root_dir, temp_dir, tools, logger):
        legacy_audio = Path(root_dir) / "videotrans" / "styles" / "no-remove.wav"
        if tools.vail_file(str(legacy_audio)):
            return str(legacy_audio)

        test_audio = Path(temp_dir) / "stt-test-audio.wav"
        test_audio.parent.mkdir(parents=True, exist_ok=True)
        if tools.vail_file(str(test_audio)):
            return str(test_audio)

        creators = (
            (self._create_test_audio_by_edge, (test_audio, tools)),
            (self._create_test_audio_by_sapi, (test_audio,)),
        )
        for creator, args in creators:
            try:
                test_audio.unlink(missing_ok=True)
                if creator(*args):
                    return str(test_audio)
            except Exception as e:
                logger.warning(f"Failed to create STT test audio with {creator.__name__}: {e}")

        raise RuntimeError(
            f"Cannot prepare STT test audio. Please put a short wav file at {legacy_audio}."
        )

    def run(self):
        try:
            from videotrans import recognition

            from videotrans.configure.config import ROOT_DIR,tr,app_cfg,settings,params,TEMP_DIR,logger,defaulelang,HOME_DIR
            from videotrans.util import tools
            audio_file = self._get_test_audio_file(
                root_dir=ROOT_DIR,
                temp_dir=TEMP_DIR,
                tools=tools,
                logger=logger,
            )
            res = recognition.run(
                audio_file=audio_file,
                cache_folder=TEMP_DIR,
                recogn_type=self.recogn_type,
                model_name=self.model_name,
                detect_language="zh-cn"
            )
            srt_str = tools.get_srt_from_list(res)
            self.uito.emit(f"ok:{srt_str}")
        except Exception as e:
            from videotrans.configure._except import get_msg_from_except
            import traceback
            except_msg=get_msg_from_except(e)
            msg = f'{except_msg}:\n' + traceback.format_exc()
            self.uito.emit(msg)
