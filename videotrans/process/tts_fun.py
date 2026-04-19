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
    overall_started = time.perf_counter()
    stats = {
        "rate_calls": 0,
        "rate_sec": 0.0,
        "trim_calls": 0,
        "trim_sec": 0.0,
        "sanitize_calls": 0,
        "sanitize_sec": 0.0,
        "fallback_calls": 0,
        "fallback_sec": 0.0,
        "custom_items": 0,
        "custom_synth_sec": 0.0,
        "clone_batches": 0,
        "clone_items": 0,
        "clone_synth_sec": 0.0,
        "prompt_cache_hits": 0,
        "prompt_build_sec": 0.0,
    }

    def _log_timing(message):
        logger.info(f"[qwen3tts][timing] {message}")
    
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
    _log_timing(f"device={device_map} dtype={dtype} queue={len(queue_tts)}")
    
    BASE_OBJ=None
    CUSTOM_OBJ=None
    

    all_roles={ r.get('role') for r in queue_tts}
    if all_roles & CUSTOM_VOICE or "clone" in all_roles:
        # 存在自定义音色
        load_started = time.perf_counter()
        CUSTOM_OBJ=Qwen3TTSModel.from_pretrained(
            f"{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{model_name}-CustomVoice",
            device_map=device_map,
            dtype=dtype,
            attn_implementation=atten
        )
        _log_timing(f"load_custom_model={time.perf_counter() - load_started:.2f}s")
    if "clone" in all_roles or all_roles-CUSTOM_VOICE:
        # 存在克隆音色
        load_started = time.perf_counter()
        BASE_OBJ=Qwen3TTSModel.from_pretrained(
            f"{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{model_name}-Base",
            device_map=device_map,
            dtype=dtype,
            attn_implementation=atten
        )
        _log_timing(f"load_base_model={time.perf_counter() - load_started:.2f}s")

    # 说话人 prompt 缓存: 同一个 ref_wav 多次合成时, 跳过 speaker-embedding 提取
    # key = (ref_wav_path, ref_text or '', x_vector_only_mode)
    _prompt_cache = {}

    # P1 A.2: 一次性探测 generate_voice_clone 是否接受 instruct kwarg
    # None=未探测, True=支持, False=不支持 (后续静默跳过, 不再重试)
    _clone_instruct_supported = {'v': None}

    def _call_voice_clone(batch_texts, prompts, instruct_str):
        """调用 generate_voice_clone, 动态检测是否支持 instruct 参数。
        不支持时自动回退无 instruct 模式。"""
        text_arg = batch_texts if len(batch_texts) > 1 else batch_texts[0]
        if instruct_str and _clone_instruct_supported['v'] is not False:
            try:
                wavs, sr = BASE_OBJ.generate_voice_clone(
                    text=text_arg,
                    language=language,
                    voice_clone_prompt=prompts,
                    instruct=instruct_str,
                )
                if _clone_instruct_supported['v'] is None:
                    _clone_instruct_supported['v'] = True
                    logger.info('[qwen3tts] generate_voice_clone 支持 instruct 参数, 启用情感透传')
                return wavs, sr
            except TypeError as e:
                if 'instruct' in str(e):
                    _clone_instruct_supported['v'] = False
                    logger.info('[qwen3tts] generate_voice_clone 不支持 instruct, clone 路径不透传情感')
                else:
                    raise
        # 默认 / 已知不支持 / 无 instruct
        return BASE_OBJ.generate_voice_clone(
            text=text_arg, language=language, voice_clone_prompt=prompts,
        )

    def _get_prompt_item(ref_wav, ref_text):
        use_xvec = not bool(ref_text)
        cache_key = (ref_wav, ref_text or '', use_xvec)
        if cache_key in _prompt_cache:
            stats["prompt_cache_hits"] += 1
            return _prompt_cache[cache_key]
        prompt_started = time.perf_counter()
        items = BASE_OBJ.create_voice_clone_prompt(
            ref_audio=ref_wav,
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=use_xvec,
        )
        stats["prompt_build_sec"] += time.perf_counter() - prompt_started
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
            return False
        try:
            factor = 1.0 + float(str(rate_text).replace('%', '')) / 100.0
        except Exception:
            return False
        if factor <= 0 or abs(factor - 1.0) < 0.01:
            return False
        filt = _cascade_atempo(factor)
        if not filt:
            return False
        tmp_out = file_path + '.tempo.wav'
        rate_started = time.perf_counter()
        try:
            tools.runffmpeg([
                '-y', '-i', file_path, '-filter:a', filt, tmp_out
            ], force_cpu=True)
            if Path(tmp_out).exists():
                shutil.move(tmp_out, file_path)
            stats["rate_calls"] += 1
            stats["rate_sec"] += time.perf_counter() - rate_started
            return True
        except Exception as e:
            logger.warning(f'Qwen3-TTS 调整语速失败 {file_path}: {e}')
            try:
                Path(tmp_out).unlink(missing_ok=True)
            except Exception:
                pass
            return False

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
    CLONE_DURATION_RATIO_LIMIT = 3.0
    CLONE_ABS_DURATION_LIMIT_MS = 15000
    CLONE_MIN_VALID_MS = 250
    CLONE_FALLBACK_ROLE = "Vivian"

    def _slot_duration_ms(it):
        try:
            source_ms = int(float(it.get('end_time_source', 0) or 0) - float(it.get('start_time_source', 0) or 0))
        except Exception:
            source_ms = 0
        if source_ms <= 0:
            try:
                source_ms = int(float(it.get('end_time', 0) or 0) - float(it.get('start_time', 0) or 0))
            except Exception:
                source_ms = 0
        return max(1, source_ms)

    def _fallback_custom_voice(item, filename, reason):
        if not CUSTOM_OBJ or CLONE_FALLBACK_ROLE not in CUSTOM_VOICE:
            logger.warning(f"[qwen3tts] fallback skipped: {reason}")
            return False
        fallback_started = time.perf_counter()
        try:
            wavs, sr = CUSTOM_OBJ.generate_custom_voice(
                text=item.get('text', ''),
                language=language,
                speaker=CLONE_FALLBACK_ROLE,
                instruct=prompt,
            )
            sf.write(filename, wavs[0], sr)
            _apply_rate_to_file(filename, item.get('rate'))
            _trim_silence_inplace(filename)
            stats["fallback_calls"] += 1
            stats["fallback_sec"] += time.perf_counter() - fallback_started
            logger.warning(f"[qwen3tts] clone fallback -> {CLONE_FALLBACK_ROLE}: {reason}")
            _write_log(logs_file, json.dumps({
                "type": "logs",
                "text": f"clone fallback -> {CLONE_FALLBACK_ROLE}: {reason}"
            }))
            return True
        except Exception as e:
            logger.warning(f"[qwen3tts] fallback custom voice failed: {e}")
            return False

    def _trim_silence_inplace(file_path):
        trim_started = time.perf_counter()
        try:
            tools.remove_silence_wav(file_path)
        except Exception:
            return False
        stats["trim_calls"] += 1
        stats["trim_sec"] += time.perf_counter() - trim_started
        return True

    def _sanitize_clone_output(item, filename):
        sanitize_started = time.perf_counter()
        try:
            if not tools.vail_file(filename):
                return False
            _trim_silence_inplace(filename)
            actual_ms = int(tools.get_audio_time(filename) or 0)
            slot_ms = _slot_duration_ms(item)
            max_allowed = min(CLONE_ABS_DURATION_LIMIT_MS, int(slot_ms * CLONE_DURATION_RATIO_LIMIT))
            max_allowed = max(max_allowed, slot_ms)
            if actual_ms < CLONE_MIN_VALID_MS:
                if _fallback_custom_voice(item, filename, f"audio too short {actual_ms}ms"):
                    actual_ms = int(tools.get_audio_time(filename) or 0)
            elif actual_ms > max_allowed:
                if _fallback_custom_voice(item, filename, f"audio too long {actual_ms}ms > {max_allowed}ms"):
                    actual_ms = int(tools.get_audio_time(filename) or 0)
                if actual_ms > max_allowed and actual_ms > 0:
                    tools.precise_speed_up_audio(file_path=filename, target_duration_ms=max_allowed)
                    _trim_silence_inplace(filename)
                    actual_ms = int(tools.get_audio_time(filename) or 0)
                    logger.warning(f"[qwen3tts] force-compressed clone audio to {actual_ms}ms (slot={slot_ms}ms)")
            stats["sanitize_calls"] += 1
            stats["sanitize_sec"] += time.perf_counter() - sanitize_started
            return True
        except Exception as e:
            logger.warning(f"[qwen3tts] sanitize clone output failed {filename}: {e}")
            return False

    def _resolve_clone_item(it):
        """返回 (kind, wavfile, ref_text, filename, text, instruct) 或 skip。
        kind: 'skip' | 'custom' | 'clone'
        instruct: P1 A.2 情感指令 (空串表示无)
        """
        text = it.get('text')
        if not text:
            return ('skip', None, None, None, None, '')
        filename = it.get('filename', '') + "-qwen3tts.wav"
        if tools.vail_file(filename):
            return ('skip', None, None, None, None, '')
        role = it.get('role')
        instruct = (it.get('instruct') or '').strip()
        if role in CUSTOM_VOICE and CUSTOM_OBJ:
            return ('custom', role, None, filename, text, instruct)
        if not BASE_OBJ:
            return ('skip', None, None, None, None, '')
        if role == 'clone':
            wavfile = it.get('ref_wav', '')
            ref_text = it.get('ref_text', '')
        else:
            wavfile = f'{ROOT_DIR}/f5-tts/{role}'
            ref_text = roledict.get(role) if roledict else None
        if not wavfile or not Path(wavfile).is_file():
            msg = f"不存在参考音频,无法克隆:{role=},{wavfile=}"
            _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))
            return ('skip', None, None, None, None, '')
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
                return ('skip', None, None, None, None, '')
            if 0 < ref_dur < MIN_REF_SEC_FOR_REF_TEXT and ref_text:
                # 短 ref 用 ref_text 会被 speaker prompt 带偏语言, 强制 x_vector 模式
                logger.debug(f'[qwen3tts] ref 过短 {ref_dur:.2f}s, 忽略 ref_text 走 x_vector_only_mode')
                ref_text = ''
        return ('clone', wavfile, ref_text or '', filename, text, instruct)

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
                _, speaker, _, filename, text, _line_instruct = resolved
                # P1 A.2: 每行情感 instruct 优先于全局 prompt
                _effective_instruct = _line_instruct if _line_instruct else prompt
                _emotion_tag = it.get('emotion', '')
                _log_suffix = f" emotion={_emotion_tag}" if _emotion_tag and _emotion_tag != 'neutral' else ''
                _write_log(logs_file, json.dumps({"type": "logs", "text": f'{i+1}/{_len} {role}{_log_suffix}'}))
                synth_started = time.perf_counter()
                wavs, sr = CUSTOM_OBJ.generate_custom_voice(
                    text=text, language=language, speaker=speaker, instruct=_effective_instruct,
                )
                stats["custom_items"] += 1
                stats["custom_synth_sec"] += time.perf_counter() - synth_started
                sf.write(filename, wavs[0], sr)
                _apply_rate_to_file(filename, it.get('rate'))
                _trim_silence_inplace(filename)
                i += 1
                continue

            # kind == 'clone': 向后扫描, 收集连续同 (wavfile, ref_text, instruct) 的条目, 组 batch
            # P1 A.2: instruct 不同时分 batch, 避免把一句的情感套在另一句上
            _, wavfile, ref_text, filename, text, batch_instruct = resolved
            batch_filenames = [filename]
            batch_texts = [text]
            batch_src_idx = [i]
            j = i + 1
            while j < _len and len(batch_texts) < MAX_BATCH:
                r = _resolve_clone_item(queue_tts[j])
                if r[0] != 'clone' or r[1] != wavfile or r[2] != ref_text or r[5] != batch_instruct:
                    break
                batch_filenames.append(r[3])
                batch_texts.append(r[4])
                batch_src_idx.append(j)
                j += 1

            prompt_item = _get_prompt_item(wavfile, ref_text)
            tag = f'{batch_src_idx[0]+1}-{batch_src_idx[-1]+1}/{_len}' if len(batch_texts) > 1 \
                else f'{batch_src_idx[0]+1}/{_len}'
            _emotion_log = f" instruct={batch_instruct[:16]}..." if batch_instruct else ''
            _write_log(logs_file, json.dumps({
                "type": "logs",
                "text": f'{tag} {role} batch={len(batch_texts)} ref={Path(wavfile).name} rate={queue_tts[batch_src_idx[0]].get("rate", "+0%")}{_emotion_log}'
            }))

            synth_started = time.perf_counter()
            wavs, sr = _call_voice_clone(
                batch_texts,
                [prompt_item] * len(batch_texts),
                batch_instruct,
            )
            stats["clone_batches"] += 1
            stats["clone_items"] += len(batch_texts)
            stats["clone_synth_sec"] += time.perf_counter() - synth_started
            # wavs 是 list[np.ndarray], 与 batch_texts 一一对应
            for src_idx, fn, w in zip(batch_src_idx, batch_filenames, wavs):
                sf.write(fn, w, sr)
                _apply_rate_to_file(fn, queue_tts[src_idx].get('rate'))
                _sanitize_clone_output(queue_tts[src_idx], fn)

            i = j
        _log_timing(
            "summary "
            f"custom_items={stats['custom_items']} custom_synth={stats['custom_synth_sec']:.2f}s "
            f"clone_batches={stats['clone_batches']} clone_items={stats['clone_items']} "
            f"clone_synth={stats['clone_synth_sec']:.2f}s prompt_build={stats['prompt_build_sec']:.2f}s "
            f"prompt_cache_hits={stats['prompt_cache_hits']} rate_calls={stats['rate_calls']} "
            f"rate={stats['rate_sec']:.2f}s trim_calls={stats['trim_calls']} trim={stats['trim_sec']:.2f}s "
            f"sanitize_calls={stats['sanitize_calls']} sanitize={stats['sanitize_sec']:.2f}s "
            f"fallback_calls={stats['fallback_calls']} fallback={stats['fallback_sec']:.2f}s "
            f"total={time.perf_counter() - overall_started:.2f}s"
        )
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
