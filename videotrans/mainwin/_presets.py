"""L1 参数预设 (preset) 管理器。

设计原则 (Linus): 扁平 JSON + 一张映射表, 无特殊逻辑。
- 预设 JSON 里每个键直接对应 main_win 的一个控件
- SETTERS 是唯一的映射表, 加/减字段只改这张表
- 未知键自动忽略, 缺失键保持当前值不变 (向后兼容)

入口:
    PresetManager(main_win).install_menu()

使用:
- 菜单栏 "预设" → [内置预设列表]
- 菜单栏 "预设" → 另存为当前配置
- 菜单栏 "预设" → 打开预设目录
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from videotrans.configure import config
from videotrans.configure.config import ROOT_DIR, logger


# 预设文件存放位置: 项目内置 + 用户自定义
BUILTIN_DIR = Path(ROOT_DIR) / 'videotrans' / 'presets'
USER_DIR = Path(ROOT_DIR) / 'presets_user'


# 映射表: (preset_key, widget_attr, kind)
# kind 取值:
#   text    -> combobox.setCurrentText(str(v))
#   index   -> combobox.setCurrentIndex(int(v))
#   checked -> checkbox.setChecked(bool(v))
#   ivalue  -> spin/slider.setValue(int(v))  (v 可能带 '%' 'Hz')
SETTERS: list[tuple[str, str, str]] = [
    ('source_language',   'source_language',   'lang'),
    ('target_language',   'target_language',   'lang'),
    ('translate_type',    'translate_type',    'index'),
    ('tts_type',          'tts_type',          'index'),
    ('recogn_type',       'recogn_type',       'index'),
    ('model_name',        'model_name',        'text'),
    ('subtitle_type',     'subtitle_type',     'index'),
    ('voice_role',        'voice_role',        'text'),
    ('voice_rate',        'voice_rate',        'ivalue'),
    ('volume_rate',       'volume_rate',       'ivalue'),
    ('pitch_rate',        'pitch_rate',        'ivalue'),
    ('voice_autorate',    'voice_autorate',    'checked'),
    ('video_autorate',    'video_autorate',    'checked'),
    ('enable_lipsync',    'enable_lipsync',    'checked'),
    ('remove_hardsub_before_subtitle', 'remove_hardsub_before_subtitle', 'checked'),
    ('is_separate',       'is_separate',       'checked'),
    ('embed_bgm',         'embed_bgm',         'checked'),
    ('is_cuda',           'enable_cuda',       'checked'),
    ('enable_diariz',     'enable_diariz',     'checked'),
    ('nums_diariz',       'nums_diariz',       'index'),
    ('align_sub_audio',   'align_sub_audio',   'checked'),
    ('remove_silent_mid', 'remove_silent_mid', 'checked'),
    ('fix_punc',          'fix_punc',          'checked'),
    ('recogn2pass',       'recogn2pass',       'checked'),
    ('only_out_mp4',      'only_out_mp4',      'checked'),
    ('remove_noise',      'remove_noise',      'checked'),
    ('copysrt_rawvideo',  'copysrt_rawvideo',  'checked'),
    ('clear_cache',       'clear_cache',       'checked'),
    ('rephrase',          'rephrase',          'index'),
]


def _lang_to_display(value) -> str:
    """语言代码 (zh-cn/en) → 当前语种下的显示名 (简体中文/英语)。
    已经是显示名或未知代码时原样返回。"""
    try:
        from videotrans.translator import LANGNAME_DICT
    except Exception:
        return str(value)
    return LANGNAME_DICT.get(str(value), str(value))


def _display_to_lang(value: str) -> str:
    """显示名 → 语言代码; 找不到返回原值。"""
    try:
        from videotrans.translator import LANGNAME_DICT_REV
    except Exception:
        return value
    return LANGNAME_DICT_REV.get(value, value)


def _apply_one(widget, kind: str, value) -> None:
    """单个控件赋值。失败静默 (某些 role/language 在当前 TTS 下可能不存在)。"""
    if kind == 'text':
        widget.setCurrentText(str(value))
    elif kind == 'lang':
        # 预设存的是语言代码 (zh-cn/en); 下拉里是显示名 (简体中文/英语)
        widget.setCurrentText(_lang_to_display(value))
    elif kind == 'index':
        widget.setCurrentIndex(int(value))
    elif kind == 'checked':
        widget.setChecked(bool(value))
    elif kind == 'ivalue':
        if isinstance(value, str):
            v = value.replace('%', '').replace('Hz', '').strip()
            value = int(v) if v else 0
        widget.setValue(int(value))


def _read_one(widget, kind: str):
    if kind == 'text':
        return widget.currentText()
    if kind == 'lang':
        # 保存时反转: 显示名 → 语言代码
        return _display_to_lang(widget.currentText())
    if kind == 'index':
        return widget.currentIndex()
    if kind == 'checked':
        return bool(widget.isChecked())
    if kind == 'ivalue':
        return int(widget.value())
    return None


class PresetManager:
    def __init__(self, main_win):
        self.main = main_win
        self._menu = None
        USER_DIR.mkdir(parents=True, exist_ok=True)

    def _status_bar(self):
        sb = getattr(self.main, 'statusBar', None)
        if callable(sb):
            sb = sb()
        return sb

    # ---------- 菜单安装 ----------
    def install_menu(self) -> None:
        # UI 里 menuBar 被赋为属性 (QMenuBar 实例), 而 QMainWindow.menuBar 是方法;
        # 优先取属性, 拿不到再 fallback 到方法
        mb = getattr(self.main, 'menuBar', None)
        if callable(mb):
            mb = mb()
        self._menu = mb.addMenu('预设')
        self._rebuild()

    def _rebuild(self) -> None:
        if self._menu is None:
            return
        self._menu.clear()
        # 内置
        for p in sorted(BUILTIN_DIR.glob('*.json')):
            self._add_preset_action(p, builtin=True)
        # 用户
        user_files = sorted(USER_DIR.glob('*.json'))
        if user_files:
            self._menu.addSeparator()
            for p in user_files:
                self._add_preset_action(p, builtin=False)
        # 管理操作
        self._menu.addSeparator()
        act_save = QAction('另存当前配置为预设…', self.main)
        act_save.triggered.connect(self._save_as)
        self._menu.addAction(act_save)

        act_open = QAction('打开预设目录', self.main)
        act_open.triggered.connect(self._open_user_dir)
        self._menu.addAction(act_open)

        act_reload = QAction('刷新列表', self.main)
        act_reload.triggered.connect(self._rebuild)
        self._menu.addAction(act_reload)

    def _add_preset_action(self, path: Path, builtin: bool) -> None:
        label = self._label_of(path)
        if builtin:
            label = f'[内置] {label}'
        act = QAction(label, self.main)
        act.triggered.connect(lambda checked=False, p=str(path): self.apply_file(p))
        self._menu.addAction(act)

    @staticmethod
    def _label_of(path: Path) -> str:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            name = data.get('_name') or path.stem
        except Exception:
            name = path.stem
        return name

    # ---------- 加载 ----------
    def apply_file(self, path: str) -> None:
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
        except Exception as e:
            QMessageBox.warning(self.main, '预设加载失败', f'{path}\n\n{e}')
            logger.exception('preset load failed')
            return
        applied, skipped = self.apply_dict(data)
        remark = data.get('_remark', '')
        msg = f'已加载预设: {data.get("_name", Path(path).stem)}\n'
        msg += f'生效 {applied} 项, 跳过 {skipped} 项'
        if remark:
            msg += f'\n\n说明: {remark}'
        self._status_bar().showMessage(msg, 6000)

    def apply_dict(self, data: dict) -> tuple[int, int]:
        applied = 0
        skipped = 0
        for key, attr, kind in SETTERS:
            if key not in data:
                continue
            widget = getattr(self.main, attr, None)
            if widget is None:
                skipped += 1
                continue
            try:
                _apply_one(widget, kind, data[key])
                # 同步到全局 params, 让任务启动时读到
                config.params[key] = data[key]
                applied += 1
            except Exception:
                logger.exception(f'preset apply key={key} failed')
                skipped += 1
        return applied, skipped

    # ---------- 另存为 ----------
    def _save_as(self) -> None:
        name, ok = QInputDialog.getText(self.main, '保存预设', '预设名称 (中文友好):')
        if not ok or not name.strip():
            return
        remark, _ = QInputDialog.getMultiLineText(
            self.main, '保存预设', '备注 (可选):', ''
        )
        data: dict = {'_name': name.strip(), '_remark': remark.strip()}
        for key, attr, kind in SETTERS:
            widget = getattr(self.main, attr, None)
            if widget is None:
                continue
            try:
                data[key] = _read_one(widget, kind)
            except Exception:
                logger.exception(f'preset read key={key} failed')

        safe = ''.join(c for c in name if c.isalnum() or c in '._- ')
        safe = safe.strip().replace(' ', '_') or 'preset'
        target = USER_DIR / f'{safe}.json'
        if target.exists():
            ret = QMessageBox.question(
                self.main, '覆盖?', f'{target.name} 已存在, 覆盖?'
            )
            if ret != QMessageBox.Yes:
                return
        target.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        self._rebuild()
        self._status_bar().showMessage(f'已保存预设: {target}', 5000)

    # ---------- 打开目录 ----------
    def _open_user_dir(self) -> None:
        path = str(USER_DIR)
        try:
            if platform.system() == 'Darwin':
                subprocess.Popen(['open', path])
            elif platform.system() == 'Windows':
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(['xdg-open', path])
        except Exception:
            QMessageBox.information(self.main, '预设目录', path)
