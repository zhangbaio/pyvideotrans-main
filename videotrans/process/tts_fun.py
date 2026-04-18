# 语音识别，新进程执行
# 返回元组
# 失败：第一个值为False，则为失败，第二个值存储失败原因
# 成功，第一个值存在需要的返回值，不需要时返回True，第二个值为None
from videotrans.configure.config import logger,ROOT_DIR

def _write_log(file, msg):
    from pathlib import Path
    try:
        Path(file).write_text(msg, encoding='utf-8')
    except Exception as e:
        logger.exception(f'写入新进程日志时出错', exc_info=True)


def qwen3tts_fun(
        queue_tts_file=None,# 配音数据存在 json文件下，根据文件路径获取
        language='Auto',#语言
        logs_file=None,
        defaulelang="en",
        is_cuda=False,
        prompt=None,
        model_name='1.7B',
        roledict=None,
        device_index=0 # gpu索引
):
    import re, os, traceback, json, time
    import shutil
    from pathlib import Path
    from videotrans.util import tools

    import torch
    torch.set_num_threads(1)
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    
    CUSTOM_VOICE= {"Vivian", "Serena", "Uncle_fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_anna", "Sohee"}

    
    queue_tts=json.loads(Path(queue_tts_file).read_text(encoding='utf-8'))
    
    atten=None
    if is_cuda:
        device_map = f'cuda:{device_index}'
        dtype=torch.float16
        try:
            import flash_attn
        except ImportError:
            pass
        else:
            atten='flash_attention_2'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon: MPS 比 CPU 快 3-5 倍, 1.7B 上 RTF ~9x
        device_map = 'mps'
        dtype = torch.float16
    else:
        device_map = 'cpu'
        dtype=torch.float32
    
    BASE_OBJ=None
    CUSTOM_OBJ=None
    

    all_roles={ r.get('role') for r in queue_tts}
    if all_roles & CUSTOM_VOICE:
        # 存在自定义音色
        CUSTOM_OBJ=Qwen3TTSModel.from_pretrained(
            f"{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{model_name}-CustomVoice",
            device_map=device_map,
            dtype=dtype,
            attn_implementation=atten
        )
    if "clone" in all_roles or all_roles-CUSTOM_VOICE:
        # 存在克隆音色
        BASE_OBJ=Qwen3TTSModel.from_pretrained(
            f"{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{model_name}-Base",
            device_map=device_map,
            dtype=dtype,
            attn_implementation=atten
        )

    # 说话人 prompt 缓存: 同一个 ref_wav 多次合成时, 跳过 speaker-embedding 提取
    # key = (ref_wav_path, ref_text or '', x_vector_only_mode)
    _prompt_cache = {}

    def _get_prompt_item(ref_wav, ref_text):
        use_xvec = not bool(ref_text)
        cache_key = (ref_wav, ref_text or '', use_xvec)
        if cache_key in _prompt_cache:
            return _prompt_cache[cache_key]
        items = BASE_OBJ.create_voice_clone_prompt(
            ref_audio=ref_wav,
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=use_xvec,
        )
        _prompt_cache[cache_key] = items[0]
        return items[0]

    def _cascade_atempo(factor):
        """ffmpeg atempo 单次限制 [0.5, 2.0]; 超出需级联相乘。
        返回: "atempo=2.000,atempo=1.500" 这种 filter 字符串。
        """
        if factor <= 0:
            return ''
        parts = []
        f = float(factor)
        # 大于 2.0 时依次拆成 2.0 × 2.0 × ... × rest
        while f > 2.0:
            parts.append(2.0)
            f /= 2.0
        while f < 0.5:
            parts.append(0.5)
            f /= 0.5
        parts.append(f)
        return ','.join(f'atempo={x:.3f}' for x in parts)

    def _apply_rate_to_file(file_path, rate_text):
        if not rate_text or rate_text in ('+0%', '0%', '0'):
            return
        try:
            factor = 1.0 + float(str(rate_text).replace('%', '')) / 100.0
        except Exception:
            return
        if factor <= 0 or abs(factor - 1.0) < 0.01:
            return
        filt = _cascade_atempo(factor)
        if not filt:
            return
        tmp_out = file_path + '.tempo.wav'
        try:
            tools.runffmpeg([
                '-y', '-i', file_path, '-filter:a', filt, tmp_out
            ], force_cpu=True)
            if Path(tmp_out).exists():
                shutil.move(tmp_out, file_path)
        except Exception as e:
            logger.warning(f'Qwen3-TTS 调整语速失败 {file_path}: {e}')
            try:
                Path(tmp_out).unlink(missing_ok=True)
            except Exception:
                pass

    # --- 极短参考音频兜底 ---
    # Qwen3-TTS 的 speaker embedding 在 < 1s 参考上非常不稳; < 0.3s 几乎必出噪声
    MIN_REF_SEC_FOR_SYNTH = 0.3      # 低于此不合成, 直接填静音
    MIN_REF_SEC_FOR_REF_TEXT = 1.0   # 低于此忽略 ref_text, 强制 x_vector_only_mode

    def _wav_seconds(p):
        try:
            info = sf.info(str(p))
            return float(info.frames) / float(info.samplerate or 1)
        except Exception:
            return 0.0

    def _write_silence(out_path, seconds):
        """写一段指定时长的静音 wav, 16k/mono; seconds 下限 0.2s 防 0 长度"""
        import numpy as np
        seconds = max(0.2, float(seconds or 0.5))
        sr = 24000
        sf.write(str(out_path), np.zeros(int(sr * seconds), dtype='float32'), sr)

    # 同 speaker 批量合成: MPS 上 batch=4 实测 1.88x 加速 (RTF 8.24→4.15)
    # CUDA 收益更大, CPU 基本等效; 保守上限 4 防 MPS OOM
    MAX_BATCH = 4

    def _resolve_clone_item(it):
        """返回 (kind, wavfile, ref_text, filename, text) 或 None(跳过)。
        kind: 'skip' | 'custom' | 'clone'
        """
        text = it.get('text')
        if not text:
            return ('skip', None, None, None, None)
        filename = it.get('filename', '') + "-qwen3tts.wav"
        if tools.vail_file(filename):
            return ('skip', None, None, None, None)
        role = it.get('role')
        if role in CUSTOM_VOICE and CUSTOM_OBJ:
            return ('custom', role, None, filename, text)
        if not BASE_OBJ:
            return ('skip', None, None, None, None)
        if role == 'clone':
            wavfile = it.get('ref_wav', '')
            ref_text = it.get('ref_text', '')
        else:
            wavfile = f'{ROOT_DIR}/f5-tts/{role}'
            ref_text = roledict.get(role) if roledict else None
        if not wavfile or not Path(wavfile).is_file():
            msg = f"不存在参考音频,无法克隆:{role=},{wavfile=}"
            _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))
            return ('skip', None, None, None, None)
        # --- 极短参考保护 (仅对 clone 路径, custom 用不到 ref_wav) ---
        if role == 'clone':
            ref_dur = _wav_seconds(wavfile)
            if 0 < ref_dur < MIN_REF_SEC_FOR_SYNTH:
                # 源字幕时长作为静音长度; 拿不到就用 0.5s
                try:
                    src_dur = (float(it.get('end_time_source', 0) or 0) -
                               float(it.get('start_time_source', 0) or 0)) / 1000.0
                except Exception:
                    src_dur = 0.5
                _write_silence(filename, src_dur if src_dur > 0 else 0.5)
                msg = (f"参考音频过短 {ref_dur:.2f}s < {MIN_REF_SEC_FOR_SYNTH}s, "
                       f"已填静音跳过: {Path(wavfile).name}")
                _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))
                logger.warning(f'[qwen3tts] {msg}')
                return ('skip', None, None, None, None)
            if 0 < ref_dur < MIN_REF_SEC_FOR_REF_TEXT and ref_text:
                # 短 ref 用 ref_text 会被 speaker prompt 带偏语言, 强制 x_vector 模式
                logger.debug(f'[qwen3tts] ref 过短 {ref_dur:.2f}s, 忽略 ref_text 走 x_vector_only_mode')
                ref_text = ''
        return ('clone', wavfile, ref_text or '', filename, text)

    try:
        _len = len(queue_tts)
        i = 0
        while i < _len:
            it = queue_tts[i]
            role = it.get('role')
            resolved = _resolve_clone_item(it)
            kind = resolved[0]

            if kind == 'skip':
                i += 1
                continue

            if kind == 'custom':
                _, speaker, _, filename, text = resolved
                _write_log(logs_file, json.dumps({"type": "logs", "text": f'{i+1}/{_len} {role}'}))
                wavs, sr = CUSTOM_OBJ.generate_custom_voice(
                    text=text, language=language, speaker=speaker, instruct=prompt,
                )
                sf.write(filename, wavs[0], sr)
                _apply_rate_to_file(filename, it.get('rate'))
                i += 1
                continue

            # kind == 'clone': 向后扫描, 收集连续同 (wavfile, ref_text) 的条目, 组 batch
            _, wavfile, ref_text, filename, text = resolved
            batch_filenames = [filename]
            batch_texts = [text]
            batch_src_idx = [i]
            j = i + 1
            while j < _len and len(batch_texts) < MAX_BATCH:
                r = _resolve_clone_item(queue_tts[j])
                if r[0] != 'clone' or r[1] != wavfile or r[2] != ref_text:
                    break
                batch_filenames.append(r[3])
                batch_texts.append(r[4])
                batch_src_idx.append(j)
                j += 1

            prompt_item = _get_prompt_item(wavfile, ref_text)
            tag = f'{batch_src_idx[0]+1}-{batch_src_idx[-1]+1}/{_len}' if len(batch_texts) > 1 \
                else f'{batch_src_idx[0]+1}/{_len}'
            _write_log(logs_file, json.dumps({
                "type": "logs",
                "text": f'{tag} {role} batch={len(batch_texts)} ref={Path(wavfile).name} rate={queue_tts[batch_src_idx[0]].get("rate", "+0%")}'
            }))

            wavs, sr = BASE_OBJ.generate_voice_clone(
                text=batch_texts if len(batch_texts) > 1 else batch_texts[0],
                language=language,
                voice_clone_prompt=[prompt_item] * len(batch_texts),
            )
            # wavs 是 list[np.ndarray], 与 batch_texts 一一对应
            for src_idx, fn, w in zip(batch_src_idx, batch_filenames, wavs):
                sf.write(fn, w, sr)
                _apply_rate_to_file(fn, queue_tts[src_idx].get('rate'))

            i = j
        return True, None
    except Exception:
        msg = traceback.format_exc()
        logger.exception(f'Qwen3-TTS 配音失败:{msg}', exc_info=True)
        return False, msg
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if CUSTOM_OBJ:
                del CUSTOM_OBJ
            if BASE_OBJ:
                del BASE_OBJ
            import gc
            gc.collect()
        except Exception:
            pass
