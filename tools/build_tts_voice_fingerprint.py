#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS 音色指纹库构建器 (离线)

作用: 给指定 TTS 引擎的每个音色合成一段固定样本, 提取说话人 embedding,
      写到 videotrans/voicejson/{engine_key}_emb.json。
      voice_matcher 层 1 会自动读取这份文件做跨引擎的声纹相似度匹配。

设计要点 (Linus):
  - 增量写盘 (每完成 1 个音色立刻落盘), 网络抖死也不丢已完成进度
  - 默认跳过已存在条目 (幂等, 可随时重跑), --force 全量重建
  - 并发=1 (多数云 TTS 有限流, 顺序最稳), --limit N 先跑几条验证链路
  - 不依赖 Qt, 纯 CLI; 但复用 videotrans.tts 里的 BaseTTS 子类, 保证和主程序行为一致

用法:
  python tools/build_tts_voice_fingerprint.py --engine doubao2
  python tools/build_tts_voice_fingerprint.py --engine doubao0 --limit 5      # 先试 5 个
  python tools/build_tts_voice_fingerprint.py --engine edgetts --force        # 全量重建
  python tools/build_tts_voice_fingerprint.py --engine doubao2 --text "你好"   # 自定义文本

支持的 engine:
  edgetts / doubao0 / doubao2 / freeazure / qwen3local / googletts / azure

输出:
  videotrans/voicejson/{engine_key}_emb.json
  {
    "voice_display_name": {
      "voice_id": "BV001_streaming",    # 可选, 部分引擎名字即 id
      "embedding": [...192 floats...],
      "gender": "f" | "m" | "any",      # 来自 voice_tagger 的名字推断 (辅助)
      "sample_sec": 4.3                 # 合成样本时长
    }
  }
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ---- 项目根注入 sys.path, 让 videotrans.* 可直接 import ----
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- 引擎键 → (voicejson 文件, tts_type, TTS 类懒加载, 语言代码) ----
# tts_type 保持和 videotrans.tts.__init__ 里的常量一致
_ENGINE_SPEC = {
    'edgetts':    {'voicejson': 'edge_tts.json',    'tts_type': 0,  'language': 'zh-cn'},
    'qwen3local': {'voicejson': 'qwen3tts.json',    'tts_type': 1,  'language': 'zh-cn'},
    'doubao2':    {'voicejson': 'doubao2.json',     'tts_type': 6,  'language': 'zh-cn'},
    'doubao0':    {'voicejson': 'doubao0.json',     'tts_type': 7,  'language': 'zh-cn'},
    'azure':      {'voicejson': 'azure_voice_list.json', 'tts_type': 18, 'language': 'zh-cn'},
    'freeazure':  {'voicejson': 'azure_voice_list.json', 'tts_type': 28, 'language': 'zh-cn'},
    'googletts':  {'voicejson': '302.json',         'tts_type': 29, 'language': 'zh-cn'},
}

DEFAULT_TEXT = "大家好, 今天天气真不错, 很高兴和你相遇。"


def _load_voices(engine_key: str) -> dict:
    """读 voicejson/{file}, 扁平化成 {display_name: voice_id_or_name}.

    兼容两种常见结构:
      A) {"zh": {"灿灿": "BV700_streaming", ...}, "en": {...}}
      B) ["voice_name_1", "voice_name_2", ...]   (纯列表, display=id)
      C) {"voice_name": {...meta...}, ...}        (对象形, 取 key)
    """
    spec = _ENGINE_SPEC[engine_key]
    path = ROOT / 'videotrans' / 'voicejson' / spec['voicejson']
    if not path.exists():
        raise FileNotFoundError(f'音色列表不存在: {path}')
    raw = json.loads(path.read_text(encoding='utf-8'))

    flat: dict = {}
    if isinstance(raw, list):
        for v in raw:
            if isinstance(v, str) and v:
                flat[v] = v
    elif isinstance(raw, dict):
        # 若顶层有 zh/en 这种语言分组, 合并所有
        lang_keys = [k for k in raw if isinstance(raw.get(k), dict)]
        candidates = [raw[k] for k in lang_keys] if lang_keys else [raw]
        for sub in candidates:
            if not isinstance(sub, dict):
                continue
            for name, val in sub.items():
                if not isinstance(name, str):
                    continue
                if isinstance(val, str):
                    flat[name] = val
                else:
                    flat[name] = name  # 对象形: display == id
    return flat


def _emb_path(engine_key: str) -> Path:
    return ROOT / 'videotrans' / 'voicejson' / f'{engine_key}_emb.json'


def _load_existing(engine_key: str) -> dict:
    p = _emb_path(engine_key)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        # 文件损坏就备份并重来, 别让已写好的东西白丢
        bak = p.with_suffix(p.suffix + f'.bak.{int(time.time())}')
        p.rename(bak)
        print(f'[warn] {p.name} 解析失败, 已备份到 {bak.name}')
        return {}


def _atomic_save(engine_key: str, data: dict) -> None:
    p = _emb_path(engine_key)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(p)


def _synth_one(engine_key: str, display_name: str, voice_id: str, text: str, out_wav: Path) -> bool:
    """用对应引擎合成一句, 产物写到 out_wav. 失败返回 False."""
    spec = _ENGINE_SPEC[engine_key]
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if out_wav.exists():
        out_wav.unlink()

    queue_item = {
        'text': text,
        'filename': str(out_wav),
        'role': display_name,     # 多数引擎用 display; doubao 内部会 get_doubao_rolelist 再映射
        'voice_id': voice_id,     # 兜底字段, 某些引擎直接读这个
        'start_time_source': 0, 'end_time_source': 0, 'startraw': '', 'endraw': '',
        'line': 1, 'time': '00:00:00,000 --> 00:00:04,000',
    }

    try:
        if engine_key == 'edgetts':
            from videotrans.tts._edgetts import EdgeTTS as Cls
        elif engine_key == 'doubao0':
            from videotrans.tts._doubao import DoubaoTTS as Cls
        elif engine_key == 'doubao2':
            from videotrans.tts._doubao2 import Doubao2TTS as Cls
        elif engine_key == 'freeazure':
            from videotrans.tts._freeazure import FreeAzureTTS as Cls
        elif engine_key == 'qwen3local':
            from videotrans.tts._qwenttslocal import QwenttsLocal as Cls
        elif engine_key == 'azure':
            from videotrans.tts._azuretts import AzureTTS as Cls
        else:
            print(f'[skip] 引擎 {engine_key} 暂未接入, 需要你手工补 import')
            return False

        inst = Cls(
            tts_type=spec['tts_type'],
            queue_tts=[queue_item],
            language=spec['language'],
            uuid=f'fp_{int(time.time()*1000)}',
            play=False,
            is_test=True,
        )
        inst.run()
        # 成功判据: 文件存在且 > 2KB
        return out_wav.exists() and out_wav.stat().st_size > 2048
    except Exception as e:
        print(f'[err ] 合成 {display_name} 失败: {e}')
        return False


def _extract_embedding(wav: Path):
    """调用 speaker_embedding.compute_embedding, 返回 list[float] 或 None."""
    try:
        from videotrans.util.speaker_embedding import compute_embedding
        emb = compute_embedding(str(wav))
        if emb and isinstance(emb, (list, tuple)) and len(emb) > 0:
            return list(emb)
        return None
    except Exception as e:
        print(f'[err ] 提 embedding 失败 ({wav.name}): {e}')
        return None


def _wav_seconds(wav: Path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(wav))
        return float(info.frames) / float(info.samplerate or 1)
    except Exception:
        return 0.0


def _tag_gender(name: str) -> str:
    try:
        from videotrans.util.voice_tagger import tag_voice
        return tag_voice(name).get('gender', 'any')
    except Exception:
        return 'any'


def build(engine_key: str, text: str, limit: int, force: bool, keep_wav: bool) -> None:
    if engine_key not in _ENGINE_SPEC:
        raise SystemExit(f'未知引擎 {engine_key}, 支持: {list(_ENGINE_SPEC)}')

    voices = _load_voices(engine_key)
    if not voices:
        raise SystemExit(f'{engine_key} 音色列表为空')

    existing = {} if force else _load_existing(engine_key)
    todo = [(n, v) for n, v in voices.items() if force or n not in existing or not existing[n].get('embedding')]
    total = len(todo)
    if limit > 0:
        todo = todo[:limit]

    print(f'[info] 引擎={engine_key}  音色总数={len(voices)}  已完成={len(existing)}  本次待建={len(todo)}'
          + (f' (limited from {total})' if limit > 0 and limit < total else ''))
    if not todo:
        print('[info] 全部已建, 无需操作 (加 --force 强制重建)')
        return

    tmp_dir = ROOT / 'tmp' / f'fp_{engine_key}'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0
    data = dict(existing)  # 写回时包含旧数据
    t0 = time.time()
    for i, (name, vid) in enumerate(todo, 1):
        out_wav = tmp_dir / f'{i:04d}_{_safe_fname(name)}.wav'
        print(f'[{i:>4}/{len(todo)}] {name} ({vid}) ...', end=' ', flush=True)
        if not _synth_one(engine_key, name, vid, text, out_wav):
            fail += 1
            print('合成 FAIL')
            continue
        emb = _extract_embedding(out_wav)
        if not emb:
            fail += 1
            print('embedding FAIL')
            if not keep_wav:
                try: out_wav.unlink()
                except Exception: pass
            continue

        data[name] = {
            'voice_id': vid,
            'embedding': emb,
            'gender': _tag_gender(name),
            'sample_sec': round(_wav_seconds(out_wav), 2),
        }
        _atomic_save(engine_key, data)   # 增量落盘: 随时可中断
        ok += 1
        if not keep_wav:
            try: out_wav.unlink()
            except Exception: pass
        eta = (time.time() - t0) / i * (len(todo) - i)
        print(f'OK   (ok={ok} fail={fail} ETA={eta:.0f}s)')

    print(f'\n[done] engine={engine_key}  ok={ok}  fail={fail}  写入={_emb_path(engine_key)}')


def _safe_fname(s: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in s)[:40]


def main():
    ap = argparse.ArgumentParser(description='TTS 音色指纹库构建器')
    ap.add_argument('--engine', required=True, choices=list(_ENGINE_SPEC), help='要建库的引擎')
    ap.add_argument('--text', default=DEFAULT_TEXT, help='合成样本文本 (默认一句中文)')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个音色 (0=全部), 用于试跑')
    ap.add_argument('--force', action='store_true', help='强制重建, 无视已有条目')
    ap.add_argument('--keep-wav', action='store_true', help='保留中间 wav 文件 (默认删除)')
    args = ap.parse_args()

    try:
        build(args.engine, args.text, args.limit, args.force, args.keep_wav)
    except KeyboardInterrupt:
        print('\n[abort] 用户中断, 已落盘的条目可下次直接续建')
        sys.exit(130)
    except Exception as e:
        print(f'[fatal] {e}')
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
