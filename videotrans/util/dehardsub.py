# -*- coding: utf-8 -*-
"""
硬字幕去除 (P0)

方案: 调用本地已安装的 video-subtitle-remover (YaoFANGUK/video-subtitle-remover).
我们不重装 VSR, 只复用用户已有的环境 — 普通 Windows 用户用作者官方整合包即可.

集成方式:
- 我们把 VSR 当成一个"黑盒 CLI": 写一个临时 runner 脚本到 cache 目录,
  用 VSR 自带的 python 执行它. runner 内部 import SubtitleRemover, 跑完把产物
  移到我们指定的输出路径.
- 好处: 进程隔离, VSR 崩溃不影响主程序; 无需主程序装 paddle/torch.
- 坏处: 多一次进程启动 (~2s), 完全可接受.

用户侧: 只需在 pyVideoTrans 的 cfg.json 或 UI 里填 `vsr_install_path` (VSR 整合包解压目录).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from videotrans.configure.config import ROOT_DIR, logger


# ---------------- runner 模板 (写到 cache 目录后执行) ----------------
# 这段代码在 VSR 的 Python 环境里跑, 不是在 pyVideoTrans 环境里跑.
# 因此不能 import 任何 pyVideoTrans 模块.
_RUNNER_TEMPLATE = r'''# -*- coding: utf-8 -*-
"""Auto-generated VSR runner by pyVideoTrans. Do not edit."""
import argparse, os, sys, shutil, traceback, glob, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--install', required=True, help='VSR install root (contains backend/main.py)')
    ap.add_argument('--video', required=True, help='input video path')
    ap.add_argument('--output', required=True, help='output video path')
    ap.add_argument('--area', default='', help='y1,y2,x1,x2 or empty for auto')
    ap.add_argument('--mode', default='', help='sttn-auto, sttn-det, lama, propainter, opencv')
    args = ap.parse_args()

    install = os.path.abspath(args.install)
    backend = os.path.join(install, 'backend')
    if not os.path.isdir(backend):
        print(f'[runner] backend dir not found: {backend}', flush=True)
        sys.exit(2)
    # 很多 VSR 版本 import 写的是 from backend.xxx, 要把 install 加到 sys.path
    if install not in sys.path:
        sys.path.insert(0, install)
    if backend not in sys.path:
        sys.path.insert(0, backend)
    # 切到 install 目录, 让 VSR 的相对资源路径 (models/ 等) 正确解析
    os.chdir(install)

    try:
        from backend.main import SubtitleRemover  # type: ignore
    except Exception:
        try:
            from main import SubtitleRemover  # type: ignore
        except Exception:
            print('[runner] failed to import SubtitleRemover', flush=True)
            traceback.print_exc()
            sys.exit(3)

    if args.mode:
        try:
            from backend.config import config as vsr_config  # type: ignore
            from backend.tools.constant import InpaintMode  # type: ignore
            mode = args.mode.strip().lower().replace('_', '-')
            mode_map = {
                'sttn': InpaintMode.STTN_AUTO,
                'auto': InpaintMode.STTN_AUTO,
                'sttn-auto': InpaintMode.STTN_AUTO,
                'sttn-det': InpaintMode.STTN_DET,
                'det': InpaintMode.STTN_DET,
                'lama': InpaintMode.LAMA,
                'propainter': InpaintMode.PROPAINTER,
                'opencv': InpaintMode.OPENCV,
            }
            if mode in mode_map:
                vsr_config.set(vsr_config.inpaintMode, mode_map[mode])
                print(f'[runner] inpaint_mode={mode_map[mode].value}', flush=True)
            else:
                print(f'[runner] unknown inpaint_mode={args.mode}, use VSR default', flush=True)
        except Exception:
            print(f'[runner] failed to set inpaint_mode={args.mode}, use VSR default', flush=True)
            traceback.print_exc()

    sub_area = None
    if args.area:
        try:
            parts = [int(x.strip()) for x in args.area.split(',')]
            if len(parts) == 4:
                sub_area = tuple(parts)  # (y1, y2, x1, x2)
        except Exception:
            print(f'[runner] invalid --area {args.area}, fallback to auto', flush=True)
            sub_area = None

    print(f'[runner] video={args.video}', flush=True)
    print(f'[runner] sub_area={sub_area}', flush=True)
    t0 = time.time()
    try:
        sr = SubtitleRemover(args.video, gui_mode=False)
        if sub_area:
            sr.sub_areas = [sub_area]
        sr.run()
    except TypeError:
        # 旧版 VSR 签名不同, 退化
        sr = SubtitleRemover(args.video)
        if sub_area and hasattr(sr, 'sub_areas'):
            sr.sub_areas = [sub_area]
        sr.run()
    except Exception:
        print('[runner] SubtitleRemover.run() raised', flush=True)
        traceback.print_exc()
        sys.exit(4)

    # 定位 VSR 实际输出: 多数版本在视频同目录生成 {stem}_no_sub.mp4
    produced = getattr(sr, 'video_out_name', None)
    if not produced or not os.path.exists(produced):
        stem = os.path.splitext(os.path.basename(args.video))[0]
        candidates = glob.glob(os.path.join(os.path.dirname(args.video), f'{stem}_no_sub.*'))
        produced = candidates[0] if candidates else None

    if not produced or not os.path.exists(produced):
        print('[runner] cannot locate VSR output file', flush=True)
        sys.exit(5)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if os.path.abspath(produced) != os.path.abspath(args.output):
        shutil.move(produced, args.output)
    print(f'[runner] done in {time.time()-t0:.1f}s -> {args.output}', flush=True)
    sys.exit(0)

if __name__ == '__main__':
    main()
'''


def _probe_python(install_path: Path) -> Optional[str]:
    """按常见整合包 / 源码安装布局探测 VSR 专用 python 路径. 都失败则回退系统 python."""
    is_win = sys.platform.startswith('win')
    py_name = 'python.exe' if is_win else 'python'
    venv_scripts = 'Scripts' if is_win else 'bin'
    sibling_env = install_path.parent / 'video-subtitle-remover-env'
    candidates = [
        sibling_env / py_name,                         # pyVideoTrans 打包内置 portable 布局
        sibling_env / venv_scripts / py_name,          # pyVideoTrans 打包内置 venv/conda 布局
        install_path / 'python' / py_name,            # 作者整合包常见布局
        install_path / 'runtime' / py_name,           # 另一种整合布局
        install_path / 'venv' / venv_scripts / py_name,
        install_path / '.venv' / venv_scripts / py_name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # 最后回退: 当前进程的 python (仅源码运行时可用；打包后 sys.executable 是 pyVideoTrans.exe)
    exe_name = Path(sys.executable).name.lower()
    if exe_name.startswith('python'):
        return sys.executable
    return None


def _subprocess_env(python_bin: str) -> dict:
    env = os.environ.copy()
    py_path = Path(python_bin)
    runtime_dir = py_path.parent.parent if py_path.parent.name.lower() in {'scripts', 'bin'} else py_path.parent
    path_parts = [
        str(py_path.parent),
        str(runtime_dir),
        str(runtime_dir / 'Library' / 'bin'),
        str(runtime_dir / 'DLLs'),
    ]
    env['PATH'] = os.pathsep.join(path_parts + [env.get('PATH', '')])
    return env


def _has_backend(install_path: Path) -> bool:
    return (install_path / 'backend' / 'main.py').exists() or (install_path / 'main.py').exists()


def is_available(install_path: str) -> Tuple[bool, str]:
    """返回 (可用, 原因). 不改任何外部状态, 纯检测."""
    if not install_path:
        return False, '未配置 vsr_install_path'
    p = Path(install_path).expanduser()
    if not p.exists() or not p.is_dir():
        return False, f'VSR 安装目录不存在: {p}'
    if not _has_backend(p):
        return False, f'VSR 目录内未找到 backend/main.py: {p}'
    py = _probe_python(p)
    if not py or not Path(py).exists():
        return False, f'未找到可用 python 解释器: {py}'
    return True, f'OK (python={py})'


def _parse_area(area_str: str, video_info: Optional[dict]) -> str:
    """
    把 vsr_sub_area 规范化成 'y1,y2,x1,x2'.
    支持两种写法:
      - 绝对像素: "850,1000,100,980"
      - 相对比例: "bottom:0.15"  底部 15% 区域 (需要 video_info 提供宽高)
    其它情况返回 ''.
    """
    if not area_str:
        return ''
    s = area_str.strip()
    if re.fullmatch(r'\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*', s):
        return ','.join(x.strip() for x in s.split(','))
    m = re.match(r'(top|bottom)\s*:\s*([0-9.]+)', s, re.I)
    if m and video_info:
        pos, ratio = m.group(1).lower(), float(m.group(2))
        w = int(video_info.get('width') or 0)
        h = int(video_info.get('height') or 0)
        if w > 0 and h > 0 and 0 < ratio < 1:
            band = max(1, int(h * ratio))
            if pos == 'bottom':
                return f'{h - band},{h},0,{w}'
            return f'0,{band},0,{w}'
    return ''


# ---------------- P2: 智能字幕区域检测 (numpy+PIL, 无 OCR) ----------------
def _ffmpeg_bin() -> str:
    """优先用仓库内的 ffmpeg, 其次 PATH."""
    local_win = Path(ROOT_DIR) / 'ffmpeg' / 'ffmpeg.exe'
    local_nix = Path(ROOT_DIR) / 'ffmpeg' / 'ffmpeg'
    if local_win.exists():
        return str(local_win)
    if local_nix.exists():
        return str(local_nix)
    return 'ffmpeg'


def _ffprobe_bin() -> str:
    local_win = Path(ROOT_DIR) / 'ffmpeg' / 'ffprobe.exe'
    local_nix = Path(ROOT_DIR) / 'ffmpeg' / 'ffprobe'
    if local_win.exists():
        return str(local_win)
    if local_nix.exists():
        return str(local_nix)
    return 'ffprobe'


def _has_video_stream(video_path: str, timeout: int = 30) -> bool:
    cmd = [
        _ffprobe_bin(),
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_type,width,height,duration',
        '-of', 'json',
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        if proc.returncode != 0:
            logger.warning(f'[dehardsub] ffprobe failed: {proc.stderr or proc.stdout}')
            return False
        data = json.loads(proc.stdout or '{}')
        streams = data.get('streams') or []
        return any(
            s.get('codec_type') == 'video'
            and int(s.get('width') or 0) > 0
            and int(s.get('height') or 0) > 0
            for s in streams
        )
    except Exception as e:
        logger.warning(f'[dehardsub] ffprobe exception: {e}')
        return False


def _extract_probe_frames(
    video_path: str,
    duration_sec: float,
    probe_dir: Path,
    num_frames: int = 8,
    timeout: int = 60,
) -> List[Path]:
    """用 ffmpeg 均匀抽 num_frames 张 JPG 到 probe_dir. 返回文件列表, 失败返回空列表."""
    probe_dir.mkdir(parents=True, exist_ok=True)
    # 清空旧帧 (重试时避免污染)
    for old in probe_dir.glob('f_*.jpg'):
        try:
            old.unlink()
        except Exception:
            pass
    interval = max(1.0, duration_sec / max(1, num_frames))
    cmd = [
        _ffmpeg_bin(), '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(video_path),
        '-vf', f'fps=1/{interval:.3f}',
        '-frames:v', str(num_frames),
        '-q:v', '3',
        str(probe_dir / 'f_%03d.jpg'),
    ]
    try:
        subprocess.run(cmd, check=False, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.debug(f'[dehardsub] ffmpeg 抽帧失败: {e}')
        return []
    return sorted(probe_dir.glob('f_*.jpg'))


def detect_subtitle_area(
    video_path: str,
    cache_folder: str,
    video_info: Optional[dict] = None,
    num_frames: int = 8,
) -> Optional[str]:
    """
    纯启发式硬字幕区域检测. 返回 'y1,y2,x1,x2' 字符串或 None.

    原理 (针对短剧白字+描边底部字幕):
      1. ffmpeg 均匀抽 8 帧
      2. 每帧转灰度, 算每行的横向边缘密度 (|diff| 均值)
      3. 帧间取中位数, 抗瞬时黑场/片头干扰
      4. 在底部 50% 区域里找显著高于全局阈值的连续带
      5. 合理性校验 (厚度 / 位置 / 最小帧数), 不可信则 None

    任何异常 → None, 上游会回退到 bottom:0.18.
    """
    try:
        import numpy as np
        from PIL import Image
    except Exception as e:
        logger.debug(f'[dehardsub] 检测依赖缺失 (numpy/PIL): {e}')
        return None

    w = int((video_info or {}).get('width') or 0)
    h = int((video_info or {}).get('height') or 0)
    dur_ms = float((video_info or {}).get('time') or 0)
    dur = dur_ms / 1000.0 if dur_ms > 100 else dur_ms  # 兼容毫秒和秒两种表达
    if w < 64 or h < 64 or dur < 1.0:
        logger.debug(f'[dehardsub] 视频尺寸/时长不足以检测 w={w} h={h} dur={dur}')
        return None

    cache_json = Path(cache_folder) / '_vsr_sub_area.json'
    # 幂等: 同一视频二次跑直接读缓存 (用 mtime+size 当指纹)
    try:
        stat = Path(video_path).stat()
        fp = f'{stat.st_size}_{int(stat.st_mtime)}'
        if cache_json.exists():
            data = json.loads(cache_json.read_text(encoding='utf-8'))
            if data.get('fp') == fp and data.get('area'):
                logger.info(f'[dehardsub] 复用字幕区域检测缓存: {data["area"]}')
                return data['area']
    except Exception:
        fp = ''

    probe_dir = Path(cache_folder) / '_vsr_probe'
    frames = _extract_probe_frames(video_path, dur, probe_dir, num_frames=num_frames)
    if len(frames) < 3:
        logger.debug(f'[dehardsub] 抽帧不足 {len(frames)}, 放弃检测')
        return None

    per_row: List = []
    for fp_path in frames:
        try:
            with Image.open(fp_path) as im:
                arr = np.asarray(im.convert('L'), dtype=np.int16)
            if arr.ndim != 2 or arr.shape[0] < 64 or arr.shape[1] < 64:
                continue
            diff = np.abs(np.diff(arr, axis=1))
            per_row.append(diff.mean(axis=1).astype(np.float32))
        except Exception as e:
            logger.debug(f'[dehardsub] 读帧失败 {fp_path}: {e}')
            continue
    if len(per_row) < 3:
        return None

    min_h = min(x.shape[0] for x in per_row)
    stacked = np.stack([x[:min_h] for x in per_row], axis=0)  # (N, H)
    med = np.median(stacked, axis=0)  # (H,)

    H = min_h
    # 只在底部 50% 找字幕 (上部字幕极少见; 如需支持上字幕后续再加)
    bot_start = int(H * 0.5)
    segment = med[bot_start:]
    if segment.size < 10:
        return None
    # 阈值: 全局基线 + 1σ; 对短剧黑场多的片源更稳
    baseline = float(np.median(med))
    sigma = float(np.std(med)) or 1.0
    thr = baseline + max(3.0, 1.0 * sigma)
    hot = segment > thr
    if hot.sum() < 4:
        logger.debug(f'[dehardsub] 底部无显著高边缘带 (thr={thr:.2f})')
        return None

    # 找最长连续带 (允许 5px 空隙, 处理字幕行间留白)
    idx = np.where(hot)[0]
    gaps = np.where(np.diff(idx) > 5)[0]
    groups = np.split(idx, gaps + 1) if len(gaps) else [idx]
    best = max(groups, key=len)
    thickness = int(best[-1] - best[0] + 1)
    # 厚度下限 (太薄是噪声) / 上限 (太厚可能是 logo/片头文字条)
    min_thick = max(8, int(H * 0.015))
    max_thick = max(40, int(H * 0.22))
    if thickness < min_thick or thickness > max_thick:
        logger.debug(f'[dehardsub] 字幕带厚度异常 {thickness} (允许 {min_thick}-{max_thick})')
        return None

    y1 = bot_start + int(best[0])
    y2 = bot_start + int(best[-1]) + 1
    # padding: old subtitles often have outlines and ascenders above the hottest edge band.
    # Expand upward more than downward to cover the whole glyph while avoiding the new subtitle area.
    pad_up = max(18, int(thickness * 0.75))
    pad_down = max(8, int(thickness * 0.30))
    y1 = max(0, y1 - pad_up)
    y2 = min(H, y2 + pad_down)

    # 宽度: 满宽 (水平方向投影法对短字幕不可靠, 且 VSR 对 inpaint 宽度不敏感)
    x1, x2 = 0, w

    area = f'{y1},{y2},{x1},{x2}'
    logger.info(f'[dehardsub] 检测到字幕区域 {area} (厚度={thickness}, thr={thr:.2f}, n={len(per_row)})')
    # 落盘
    try:
        cache_json.write_text(
            json.dumps({'fp': fp, 'area': area, 'thickness': thickness, 'thr': round(thr, 2)},
                       ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
    except Exception:
        pass
    return area


def _resolve_sub_area(
    cfg_area: str,
    video_path: str,
    cache_folder: str,
    video_info: Optional[dict],
) -> Tuple[str, str]:
    """
    统一解析用户配置 → 最终 'y1,y2,x1,x2' + 来源标签.
    cfg 取值:
      ''  or 'auto'     → 智能检测, 失败回退 bottom:0.18
      'bottom:0.15'     → 按比例
      像素              → 原样
      'none'/'full'     → 返回空 (让 VSR 自己扫)
    返回 (area_string, source_label).
    """
    s = (cfg_area or '').strip().lower()
    if s in ('none', 'full'):
        return '', 'full-frame (user)'
    if s in ('', 'auto'):
        detected = detect_subtitle_area(video_path, cache_folder, video_info)
        if detected:
            return detected, 'auto-detected'
        fallback = _parse_area('bottom:0.18', video_info)
        return fallback, 'auto-fallback bottom:0.18'
    parsed = _parse_area(cfg_area, video_info)
    return parsed, f'user-config "{cfg_area}"'


def remove_hardsub(
    *,
    video_path: str,
    output_path: str,
    install_path: str,
    cache_folder: str,
    sub_area_cfg: str = '',
    inpaint_mode: str = '',
    video_info: Optional[dict] = None,
    timeout_sec: int = 3600,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """
    同步调用 VSR 去除硬字幕. 返回 (成功, 说明).
    成功时 output_path 存在且为去字幕后的视频.
    失败一律返回 (False, reason), 不抛异常, 由调用方决定回退策略.
    """
    ok, why = is_available(install_path)
    if not ok:
        return False, why

    src = Path(video_path)
    if not src.exists():
        return False, f'源视频不存在: {src}'

    cache = Path(cache_folder)
    cache.mkdir(parents=True, exist_ok=True)
    runner_path = cache / '_vsr_runner.py'
    runner_path.write_text(_RUNNER_TEMPLATE, encoding='utf-8')

    install = Path(install_path).expanduser()
    py = _probe_python(install)
    area, area_src = _resolve_sub_area(sub_area_cfg or '', str(src), cache_folder, video_info)

    cmd = [
        py,
        str(runner_path),
        '--install', str(install),
        '--video', str(src),
        '--output', str(output_path),
    ]
    if area:
        cmd += ['--area', area]
    if inpaint_mode:
        cmd += ['--mode', inpaint_mode]

    logger.info(f'[dehardsub] 启动 VSR: area={area} ({area_src}) mode={inpaint_mode or "default"} cmd={cmd}')
    if progress_cb:
        try:
            progress_cb(f'字幕区域: {area or "全帧"} ({area_src})')
        except Exception:
            pass
    if progress_cb:
        try:
            progress_cb(f'去除硬字幕中 (VSR): {src.name} ...')
        except Exception:
            pass

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(install),
            env=_subprocess_env(py),
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )
    except Exception as e:
        return False, f'启动 VSR 进程失败: {e}'

    last_line = ''
    deadline = time.time() + timeout_sec
    try:
        while True:
            if time.time() > deadline:
                proc.kill()
                return False, f'VSR 超时 ({timeout_sec}s)'
            line = proc.stdout.readline() if proc.stdout else ''
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            line = line.rstrip()
            if line:
                last_line = line
                logger.debug(f'[dehardsub][vsr] {line}')
                # VSR 进度输出里常包含 "rate" / "%" — 透传给 UI
                if progress_cb and ('%' in line or 'Processing' in line or '[runner]' in line):
                    try:
                        progress_cb(f'VSR: {line[-80:]}')
                    except Exception:
                        pass
    finally:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    dur = time.time() - t0
    rc = proc.returncode
    if rc != 0:
        return False, f'VSR 退出码 {rc}, 耗时 {dur:.1f}s, 末行: {last_line}'

    if not Path(output_path).exists():
        return False, f'VSR 成功退出但未生成输出文件: {output_path}'
    logger.info(f'[dehardsub] 完成, 耗时 {dur:.1f}s -> {output_path}')
    if Path(output_path).stat().st_size <= 1024 or not _has_video_stream(output_path):
        return False, f'VSR generated an invalid video output: {output_path}; last line: {last_line}'
    return True, f'ok in {dur:.1f}s'
