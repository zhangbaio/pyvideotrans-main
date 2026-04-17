import json
import sys
from typing import List, Dict, Optional
from pathlib import Path
import re
import time

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QCheckBox,
    QComboBox, QPushButton, QWidget, QGroupBox, QPlainTextEdit, 
    QMessageBox, QProgressBar, QApplication, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QGridLayout
)
from PySide6.QtGui import QIcon, QDesktopServices, QColor
from PySide6.QtCore import Qt, QTimer, QSize, QUrl

from videotrans.configure.config import ROOT_DIR,tr,app_cfg,settings,params,TEMP_DIR,logger,defaulelang,HOME_DIR
from videotrans.util import tools

class SpeakerAssignmentDialog(QDialog):
    def __init__(
            self,
            parent=None,
            target_sub: str = None,
            all_voices: Optional[List[str]] = None,
            source_sub: str = None,
            cache_folder=None,
            target_language="en",
            tts_type=0,
            video_path: str = ""
    ):
        super().__init__()
        self.parent = parent
        self.target_sub = target_sub
        self.source_srtstring = None
        self.cache_folder = cache_folder
        self.target_language = target_language
        self.tts_type = tts_type
        self.video_path = video_path or ""

        # --- 声音克隆库 (voice_library) 集成 ---
        # drama_dir: 默认用视频父目录名；无 video_path 时退化为 None (完全不启用库)
        self.drama_dir = None
        self.library_characters = []  # [{name, total_sec, episodes}, ...]
        self.speaker_name_edits = {}
        self.speaker_name_combos = {}
        if self.video_path:
            try:
                from videotrans.util.voice_library import get_drama_dir, list_characters
                self.drama_dir = get_drama_dir(self.video_path)
                self.library_characters = list_characters(self.drama_dir)
            except Exception as e:
                logger.warning(f'[voice_library] 初始化失败: {e}')
                self.drama_dir = None

        if source_sub:
            sour_pt = Path(source_sub)
            if sour_pt.as_posix() and not sour_pt.samefile(Path(target_sub)):
                try:
                    self.source_srtstring = sour_pt.read_text(encoding="utf-8")
                except:
                    self.source_srtstring = ""

        self.srt_list_dict = tools.get_subtitle_from_srt(self.target_sub)

        # 说话人数据初始化
        self.speaker_list_sub = []
        self.speakers = {}
        try:
            spk_json_path = Path(f'{self.cache_folder}/speaker.json')
            _list_sub = [] if not spk_json_path.exists() else json.loads(spk_json_path.read_text(encoding='utf-8'))
            _set = set(_list_sub) if _list_sub else None
            if _set and len(_set) > 1:
                self.speaker_list_sub = _list_sub
                self.speakers = {it: None for it in sorted(list(_set))}
        except Exception as e:
            logger.exception(f'获取说话人id失败:{e}', exc_info=True)

        self.all_voices = all_voices or []

        # L2: 预读 spk_to_character.json (若 diariz 阶段已做自动声纹匹配)
        # 用于 _create_speaker_assignment_area 里预填输入框/下拉
        self.auto_matched_spk2name: dict = {}
        try:
            _sm_path = Path(f'{self.cache_folder}/spk_to_character.json')
            if _sm_path.exists():
                _sm_data = json.loads(_sm_path.read_text(encoding='utf-8'))
                self.auto_matched_spk2name = dict(_sm_data.get('mapping', {}) or {})
                # 根据匹配到的角色, 同步回填 fixed_voice (用户打开对话框就看到最终效果)
                if self.drama_dir is not None and self.auto_matched_spk2name:
                    try:
                        from videotrans.util.voice_library import get_fixed_voice
                        for spk_id, ch_name in self.auto_matched_spk2name.items():
                            if spk_id in self.speakers:
                                v = get_fixed_voice(self.drama_dir, ch_name)
                                if v and (not self.all_voices or v in self.all_voices):
                                    self.speakers[spk_id] = v
                    except Exception as e:
                        logger.warning(f'[voice_library] 预填 fixed_voice 失败: {e}')
        except Exception as e:
            logger.warning(f'[voice_library] 读 spk_to_character.json 失败: {e}')

        # 自动为每个说话人轮询分配一个固定音色 (用户仍可在 UI 里手动改)
        # 设计 (Linus): 只有 (a) 检测到多说话人 (b) 音色池非空 时才触发
        # 不偷偷开 diariz, 不替换用户已手动分配的值
        if self.speakers:
            self._auto_assign_speaker_voices()

        self.setWindowTitle(tr("zidonghebingmiaohou"))
        self.setWindowIcon(QIcon(f"{ROOT_DIR}/videotrans/styles/icon.ico"))
        self.setMinimumWidth(int(parent.width*0.95))
        self.setMinimumHeight(int(parent.height*0.95))
        self.setWindowFlags(
        Qt.WindowStaysOnTopHint |       # 2. 始终在最顶层
            Qt.WindowTitleHint |            # 3. 显示标题栏
            Qt.CustomizeWindowHint |        # 4. 允许自定义标题栏按钮（否则OS会强制加关闭按钮）
            Qt.WindowMaximizeButtonHint     # 5. 只加最大化按钮，不加关闭按钮
        )

        main_layout = QVBoxLayout(self)
        
        # --- 顶部：倒计时与提示 ---
        self.count_down = int(float(settings.get('countdown_sec', 1)))
        top_layout = QVBoxLayout()
        hstop = QHBoxLayout()

        self.prompt_label = QLabel(tr("This window will automatically close after the countdown ends"))
        self.prompt_label.setStyleSheet('font-size:14px;color:#aaaaaa')
        self.prompt_label.setWordWrap(True)
        hstop.addWidget(self.prompt_label)

        self.stop_button = QPushButton(f"{tr('Click here to stop the countdown')}({self.count_down})")
        self.stop_button.setStyleSheet("font-size: 16px;color:#ffff00")
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.setMinimumSize(QSize(300, 35))
        self.stop_button.clicked.connect(self.stop_countdown)
        hstop.addWidget(self.stop_button)

        top_layout.addLayout(hstop)
        prompt_label2 = QLabel(tr("If you need to delete a line of subtitles, just clear the text in that line"))
        prompt_label2.setAlignment(Qt.AlignCenter)
        prompt_label2.setStyleSheet("color: #dddddd")
        prompt_label2.setWordWrap(True)
        top_layout.addWidget(prompt_label2)
        main_layout.addLayout(top_layout)

        # --- 查找替换区域 ---
        search_replace_layout = QHBoxLayout()
        search_replace_layout.addStretch()
        self.search_input = QLineEdit()
        self.search_input.setMaximumWidth(200)
        self.search_input.setPlaceholderText(tr("Original text"))
        search_replace_layout.addWidget(self.search_input)
        self.replace_input = QLineEdit()
        self.replace_input.setPlaceholderText(tr("Replace"))
        self.replace_input.setMaximumWidth(200)
        search_replace_layout.addWidget(self.replace_input)
        replace_button = QPushButton(tr("Replace"))
        replace_button.setMinimumWidth(100)
        replace_button.setMaximumWidth(200)
        replace_button.setCursor(Qt.PointingHandCursor)
        replace_button.clicked.connect(self.replace_text)
        search_replace_layout.addWidget(replace_button)
        search_replace_layout.addStretch()
        main_layout.addLayout(search_replace_layout)

        # --- 中间内容区域 ---
        content_layout = QHBoxLayout()
        
        # 左侧：源字幕参考
        if self.source_srtstring:
            left_widget = QWidget()
            left_layout = QVBoxLayout(left_widget)
            self.raw_srt_edit = QPlainTextEdit()
            self.raw_srt_edit.setPlainText(self.source_srtstring)
            self.raw_srt_edit.setReadOnly(True)
            self.raw_srt_edit.setStyleSheet("color: #aaaaaa;")
            tiplabel = QLabel(tr("This is the original language subtitles for comparison reference"))
            tiplabel.setStyleSheet("color:#aaaaaa")
            left_layout.addWidget(tiplabel)
            left_layout.addWidget(self.raw_srt_edit)
            content_layout.addWidget(left_widget, stretch=2)

        # 右侧主区域
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        
        # Loading 区域
        self.loading_widget = QWidget()
        load_layout = QVBoxLayout(self.loading_widget)
        self.loading_label = QLabel(tr('Loading...'), self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        load_layout.addWidget(self.loading_label)
        self.right_layout.addWidget(self.loading_widget)

        # 表格容器
        self.table_container = QWidget()
        self.table_container_layout = QVBoxLayout(self.table_container)
        self.table_container.setVisible(False)
        self.right_layout.addWidget(self.table_container)
        
        # 底部按钮容器
        self.bottom_button_container = QWidget()
        self.bottom_button_container_layout = QHBoxLayout(self.bottom_button_container)
        self.bottom_button_container.setVisible(False)
        self.right_layout.addWidget(self.bottom_button_container)

        content_layout.addWidget(self.right_widget, stretch=7)
        main_layout.addLayout(content_layout, stretch=1)

        # --- 底部按钮 ---
        self.save_button = QPushButton(tr("nextstep"))
        self.save_button.setCursor(Qt.PointingHandCursor)
        self.save_button.setMinimumSize(QSize(300, 35))
        self.save_button.clicked.connect(self.save_and_close)

        self.save_button2 = QPushButton(tr("nosaveandstep"))
        self.save_button2.setCursor(Qt.PointingHandCursor)
        self.save_button2.setToolTip(tr('bubaocunshuoming', self.target_sub))
        self.save_button2.setMinimumSize(QSize(200, 35))
        self.save_button2.clicked.connect(self.save_and_close2)

        self.opendir_button = QPushButton(tr("opendir_button source_sub"))
        self.opendir_button.setCursor(Qt.PointingHandCursor)
        self.opendir_button.setMaximumSize(QSize(150, 30))
        self.opendir_button.clicked.connect(self.opendir_sub)

        cancel_button = QPushButton(tr("Terminate this mission"))
        cancel_button.setCursor(Qt.PointingHandCursor)
        cancel_button.setStyleSheet("background-color:transparent;color:#ff0")
        cancel_button.setMinimumSize(QSize(150, 30))
        cancel_button.clicked.connect(self.cancel_and_close)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.save_button)
        bottom_layout.addWidget(self.save_button2)
        bottom_layout.addWidget(self.opendir_button)
        bottom_layout.addWidget(cancel_button)
        bottom_layout.addStretch()

        main_layout.addLayout(bottom_layout)

        # 延迟加载表格
        QTimer.singleShot(10, self.load_table)


    def load_table(self):
        """极致性能加载表格"""
        if not self.isVisible():
            return

        try:
            # 1. 创建 QTableWidget（比 Model/View 快得多）
            self.table = QTableWidget()
            
            # 2. 【极致性能配置】禁用所有非必要功能
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(["Sel", tr("Line"), tr('Speaker'), tr("Dubbing role"), tr("Time Axis"), tr("Subtitle Text")])
            
            # 禁用所有视觉效果
            self.table.setAlternatingRowColors(False)
            self.table.setShowGrid(False)  # 不显示网格线
            
            # 禁用选择
            self.table.setSelectionMode(QAbstractItemView.NoSelection)
            self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
            
            # 禁用焦点
            self.table.setFocusPolicy(Qt.NoFocus)
            
            # 固定行高，避免动态计算
            self.table.verticalHeader().setDefaultSectionSize(22)
            self.table.verticalHeader().setVisible(False)
            
            # 列宽设置
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Fixed)  # Sel
            header.setSectionResizeMode(1, QHeaderView.Fixed)  # ID
            header.setSectionResizeMode(2, QHeaderView.Fixed)  # Spk
            header.setSectionResizeMode(3, QHeaderView.Fixed)  # Role
            header.setSectionResizeMode(4, QHeaderView.Fixed)  # Time
            header.setSectionResizeMode(5, QHeaderView.Stretch)  # Text
            
            self.table.setColumnWidth(0, 30)
            self.table.setColumnWidth(1, 40)
            self.table.setColumnWidth(2, 50)
            self.table.setColumnWidth(3, 150)
            self.table.setColumnWidth(4, 210)
            
            # 最小样式
            self.table.setStyleSheet("""
                QTableWidget {
                    color: #eeeeee;
                    border: none;
                }
                QHeaderView::section {
                    background-color: #2b2b2b;
                    color: white;
                    padding: 2px;
                    border: none;
                    border-right: 1px solid #3e3e3e;
                }
                QTableWidget::item {
                    padding: 2px;
                }
            """)
            
            # 3. 预计算所有显示数据
            speaker_keys = list(self.speakers.keys()) if self.speakers else []
            default_spk = speaker_keys[0] if speaker_keys else ''
            
            self.display_data = []
            for i, item in enumerate(self.srt_list_dict):
                # Speaker ID
                if self.speakers and i < len(self.speaker_list_sub):
                    spk = self.speaker_list_sub[i]
                else:
                    spk = default_spk if self.speakers else ''
                
                # 时间字符串
                duration = (item['end_time'] - item['start_time']) / 1000.0
                time_str = f"{item['startraw']}->{item['endraw']}({duration:.1f}s)"
                
                self.display_data.append({
                    'line': item['line'],
                    'spk': spk,
                    'time_str': time_str,
                    'text': item['text'],
                    'startraw': item['startraw'],
                    'endraw': item['endraw'],
                    'start_time': item['start_time'],
                    'end_time': item['end_time'],
                    'checked': False,
                    'role': ''
                })
            
            # 4. 设置行数
            total_rows = len(self.display_data)
            self.table.setRowCount(total_rows)
            
            # 5. 【批量填充数据】一次性创建所有单元格
            self._batch_fill_table(0, min(total_rows, 100))  # 先填充前100行
            
            # 6. 添加到布局
            self.table_container_layout.addWidget(self.table)
            
            # 7. 添加底部按钮
            self._setup_bottom_buttons()
            
            # 8. 显示表格
            self.loading_widget.setVisible(False)
            self.table_container.setVisible(True)
            self.bottom_button_container.setVisible(True)
            
            # 9. 延迟加载剩余数据
            if total_rows > 100:
                QTimer.singleShot(0, lambda: self._load_remaining_rows(100))
            
            # 10. 启动倒计时
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_countdown)
            self.timer.start(1000)
            self._active()
            
        except Exception as e:
            print(f"Load table failed: {e}")
            import traceback
            traceback.print_exc()
            self.loading_label.setText(f"Error: {e}")

    def _batch_fill_table(self, start_row, end_row):
        """批量填充表格数据 - 减少重绘"""
        for row in range(start_row, end_row):
            data = self.display_data[row]
            
            # 第0列：复选框
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            chk_item.setCheckState(Qt.Unchecked)
            self.table.setItem(row, 0, chk_item)
            
            # 第1列：ID（只读）
            id_item = QTableWidgetItem(str(data['line']))
            id_item.setFlags(Qt.ItemIsEnabled)  # 只读
            self.table.setItem(row, 1, id_item)
            
            # 第2列：Speaker（只读）
            spk_item = QTableWidgetItem(data['spk'])
            spk_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 2, spk_item)
            
            # 第3列：Role（只读，显示用）
            role_item = QTableWidgetItem(tr('Default Role'))
            role_item.setFlags(Qt.ItemIsEnabled)
            role_item.setForeground(QColor("#ff4d4d"))
            self.table.setItem(row, 3, role_item)
            
            # 第4列：Time（只读）
            time_item = QTableWidgetItem(data['time_str'])
            time_item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 4, time_item)
            
            # 第5列：Text（可编辑）
            text_item = QTableWidgetItem(data['text'])
            text_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
            self.table.setItem(row, 5, text_item)

    def _load_remaining_rows(self, start_row):
        """延迟加载剩余行 - 避免界面冻结"""
        total = len(self.display_data)
        batch_size = 200  # 每批加载200行
        
        end_row = min(start_row + batch_size, total)
        self._batch_fill_table(start_row, end_row)
        
        if end_row < total:
            # 还有数据，继续加载
            QTimer.singleShot(0, lambda: self._load_remaining_rows(end_row))

    def _setup_bottom_buttons(self):
        """设置底部按钮区域"""
        # 如果有说话人，添加说话人分配区域
        if self.speakers:
            speaker_widget = self._create_speaker_assignment_area()
            # 插入到表格容器之前
            self.right_layout.insertWidget(0, speaker_widget)
        
        # 底部按钮
        self.subtitle_combo = QComboBox()
        self.subtitle_combo.addItems(self.all_voices)
        self.bottom_button_container_layout.addWidget(self.subtitle_combo)

        assign_button = QPushButton(tr("Assign roles to selected subtitles"))
        assign_button.setCursor(Qt.PointingHandCursor)
        assign_button.clicked.connect(self.assign_subtitle_roles)
        assign_button.setMinimumSize(QSize(180, 28))
        self.bottom_button_container_layout.addWidget(assign_button)

        self.listen_button = QPushButton(tr("Trial dubbing"))
        self.listen_button.setCursor(Qt.PointingHandCursor)
        self.listen_button.clicked.connect(self.listen_dubbing)
        self.bottom_button_container_layout.addWidget(self.listen_button)
        
        labe_tips=QLabel(tr('If not specified separately'))
        
        self.bottom_button_container_layout.addWidget(labe_tips)
        self.bottom_button_container_layout.addStretch()

    def _create_speaker_assignment_area(self):
        """创建说话人分配区域"""
        group = QGroupBox("")
        group.setStyleSheet("QGroupBox{border:none;}")
        layout = QVBoxLayout(group)
        label_tips = QLabel(tr("Assign a timbre to each speaker"))
        label_tips.setStyleSheet("color:#aaaaaa")
        layout.addWidget(label_tips)

        self.speaker_checks = {}
        self.speaker_labels = {}
        # M2: 每说话人 → 角色名输入 (QLineEdit) + 已有角色下拉 (QComboBox)
        self.speaker_name_edits = {}    # spk_id → QLineEdit
        self.speaker_name_combos = {}   # spk_id → QComboBox

        # 若启用了 voice_library, 在顶部加一条提示
        if self.drama_dir is not None:
            lib_hint = QLabel(
                tr('Drama library') + f': {self.drama_dir.name}  '
                + tr('(Name a speaker to add/reuse cloned voice; leave blank to skip library)')
            )
            lib_hint.setStyleSheet('color:#88ccff;font-size:12px;')
            lib_hint.setWordWrap(True)
            layout.addWidget(lib_hint)

        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 5, 0, 5)
        grid_layout.setHorizontalSpacing(10)
        grid_layout.setVerticalSpacing(5)

        # 每说话人一整行: [Sel] [Speaker x] [固定音色 label] [新名输入框] [已有角色下拉]
        # 列数保持紧凑: drama_dir 为 None 时不显示后两列
        has_lib = self.drama_dir is not None
        existing_names = [''] + [c['name'] for c in self.library_characters]

        for i, spk_id in enumerate(self.speakers):
            check = QCheckBox(f'{tr("Speaker")}{spk_id}')
            check.setStyleSheet("color: #dddddd;")

            # 显示自动分配后的初始音色 (若有)
            initial_voice = self.speakers.get(spk_id) or ''
            label = QLabel(initial_voice)
            label.setMinimumWidth(80)
            label.setStyleSheet("color: #ffcccc;")

            col = 0
            grid_layout.addWidget(check, i, col); col += 1
            grid_layout.addWidget(label, i, col); col += 1

            if has_lib:
                name_edit = QLineEdit()
                name_edit.setPlaceholderText(tr('Character name (blank = skip library)'))
                name_edit.setMinimumWidth(160)

                name_combo = QComboBox()
                name_combo.addItems(existing_names)
                name_combo.setMinimumWidth(140)

                # L2 预填: diariz 阶段声纹已匹配到某角色 → 输入框和下拉同步预选
                auto_name = self.auto_matched_spk2name.get(spk_id, '')
                if auto_name:
                    name_edit.setText(auto_name)
                    if auto_name in existing_names:
                        name_combo.setCurrentText(auto_name)
                    # 加个绿色前缀提示用户"这是自动匹配的"
                    name_edit.setStyleSheet('color:#00cc66;')
                    name_edit.setToolTip(tr('Auto-matched by voice embedding, you can modify'))

                grid_layout.addWidget(name_edit, i, col); col += 1

                # 选中下拉 → 回填输入框 + 按角色绑定的固定音色覆盖当前行 (S2: 跨集音色一致性)
                name_combo.currentTextChanged.connect(
                    lambda text, sid=spk_id, edit=name_edit:
                        self._on_character_selected(sid, text, edit)
                )
                grid_layout.addWidget(name_combo, i, col); col += 1

                self.speaker_name_edits[spk_id] = name_edit
                self.speaker_name_combos[spk_id] = name_combo

            self.speaker_checks[check] = spk_id
            self.speaker_labels[check] = label

        layout.addLayout(grid_layout)

        bottom_row = QHBoxLayout()
        self.speaker_combo = QComboBox()
        self.speaker_combo.addItems(self.all_voices)
        
        lbl = QLabel(tr('Dubbing role'))
        lbl.setStyleSheet("color: #dddddd;")
        bottom_row.addWidget(lbl)
        bottom_row.addWidget(self.speaker_combo)

        assign_button = QPushButton(tr("Assign roles"))
        assign_button.setCursor(Qt.PointingHandCursor)
        assign_button.clicked.connect(self.assign_speaker_roles)
        assign_button.setMinimumSize(QSize(120, 26))
        bottom_row.addWidget(assign_button)

        # 一键重新自动分配 (覆盖当前所有说话人)
        reassign_button = QPushButton(tr("Auto reassign"))
        reassign_button.setCursor(Qt.PointingHandCursor)
        reassign_button.clicked.connect(self._on_reassign_clicked)
        reassign_button.setMinimumSize(QSize(120, 26))
        reassign_button.setToolTip(tr("Reassign each speaker a distinct voice automatically"))
        bottom_row.addWidget(reassign_button)

        bottom_row.addStretch()

        layout.addLayout(bottom_row)
        return group

    # ---------- 自动分配 ----------
    _AUTO_SKIP_VOICES = frozenset({'No', 'clone', ''})

    def _candidate_voices(self):
        """从 all_voices 里剔掉不适合自动分配的条目 (No / clone / 空)"""
        return [v for v in self.all_voices if v not in self._AUTO_SKIP_VOICES]

    def _auto_assign_speaker_voices(self, force: bool = False):
        """给每个说话人分配一个固定音色。

        force=False (默认): 只对当前为 None 的 speaker 赋值, 不覆盖用户手工选择
        force=True: 全部重新分配, 用于"一键自动分配"按钮

        策略 (videotrans.util.voice_matcher 三层降级):
          1. 指纹库存在 → 源 spk embedding vs 音色 embedding 余弦相似
          2. 名字可标签 → F0 性别 + voice_tagger 过滤池, round-robin
          3. 裸 round-robin (兜底)
        任意一层异常都自动降到下一层, 不阻断 UI。
        """
        pool = self._candidate_voices()
        if not pool:
            return

        # 收集 spkN_ref.wav (来自 extract_speaker_refs 的产物)
        speaker_refs: dict = {}
        try:
            refs_json = Path(f'{self.cache_folder}/speaker_refs.json') if self.cache_folder else None
            if refs_json and refs_json.exists():
                _r = json.loads(refs_json.read_text(encoding='utf-8'))
                # 兼容两种结构: {spk: path} 或 {spk: {ref: path, ...}}
                for k, v in (_r or {}).items():
                    if isinstance(v, str):
                        speaker_refs[k] = v
                    elif isinstance(v, dict) and v.get('ref'):
                        speaker_refs[k] = v['ref']
        except Exception as e:
            logger.warning(f'[voice_matcher] 读 speaker_refs.json 失败: {e}')

        # existing: force=False 时把已有值传进去保护, force=True 时全清空重分
        existing = {}
        if not force:
            existing = {sid: v for sid, v in self.speakers.items() if v}

        # 若没有 spkN_ref.wav (diariz 未跑 / 单人场景), 退化到纯 round-robin
        if not speaker_refs:
            for idx, spk_id in enumerate(self.speakers.keys()):
                if force or self.speakers.get(spk_id) is None:
                    self.speakers[spk_id] = pool[idx % len(pool)]
            return

        try:
            from videotrans.util.voice_matcher import match_voices_to_speakers
            # speaker_refs 只保留当前对话框 self.speakers 里存在的 spk
            filtered_refs = {k: v for k, v in speaker_refs.items() if k in self.speakers}
            result = match_voices_to_speakers(
                speaker_refs=filtered_refs,
                all_voices=list(self.all_voices or []),
                tts_type=int(self.tts_type or 0),
                existing=existing,
            )
            for spk_id in self.speakers.keys():
                if spk_id in result:
                    self.speakers[spk_id] = result[spk_id]
        except Exception as e:
            logger.warning(f'[voice_matcher] 匹配失败, 退回 round-robin: {e}')
            for idx, spk_id in enumerate(self.speakers.keys()):
                if force or self.speakers.get(spk_id) is None:
                    self.speakers[spk_id] = pool[idx % len(pool)]

    def _on_character_selected(self, spk_id, character_name, name_edit):
        """S2: 用户在"已有角色"下拉选了一项 → 回填输入框 + 查 drama.json 的 fixed_voice 覆盖当前行音色。

        空字符串 = 用户选了第一项空行, 不做任何操作 (不清空)。
        未绑定固定音色 / 无 drama_dir: 只回填名字, 不改音色。
        """
        if not character_name:
            return
        # 1) 回填输入框
        name_edit.setText(character_name)
        # 2) 跨集音色复用
        if self.drama_dir is None:
            return
        try:
            from videotrans.util.voice_library import get_fixed_voice
            bound_voice = get_fixed_voice(self.drama_dir, character_name)
        except Exception as e:
            logger.warning(f'[voice_library] get_fixed_voice 失败: {e}')
            return
        if not bound_voice:
            return
        # 校验这个音色确实在当前 all_voices 里 (避免老 drama.json 里存的旧音色失效)
        if self.all_voices and bound_voice not in self.all_voices:
            logger.info(f'[voice_library] 角色 "{character_name}" 绑定音色 {bound_voice} 不在当前池中, 忽略')
            return
        self.speakers[spk_id] = bound_voice
        # 刷新上方该 speaker 行的 label + 表格 Role 列
        for check, sid in self.speaker_checks.items():
            if sid == spk_id:
                self.speaker_labels[check].setText(bound_voice)
                break
        self._update_role_column()

    def _on_reassign_clicked(self):
        """一键重新自动分配"""
        self._auto_assign_speaker_voices(force=True)
        # 刷新上方 speaker_labels 显示
        for check, spk_id in self.speaker_checks.items():
            self.speaker_labels[check].setText(self.speakers.get(spk_id) or '')
        # 刷新表格"配音角色"列
        self._update_role_column()

    def assign_speaker_roles(self):
        """分配角色给说话人"""
        selected_role = self.speaker_combo.currentText()
        role_value = None if selected_role == "No" else selected_role

        for check, spk_id in self.speaker_checks.items():
            if check.isChecked():
                self.speakers[spk_id] = role_value
                self.speaker_labels[check].setText(selected_role if role_value else "")
                check.setChecked(False)
        
        # 更新表格中的 Role 列
        self._update_role_column()

    def _update_role_column(self):
        """更新 Role 列显示"""
        for row, data in enumerate(self.display_data):
            # 优先显示行内角色，否则显示说话人对应的全局角色
            role = data.get('role', '')
            if not role and data['spk']:
                role = self.speakers.get(data['spk'], '')
            
            item = self.table.item(row, 3)
            if item:
                item.setText(role if role else tr('Default Role'))

    def assign_subtitle_roles(self):
        """分配角色给选中的行"""
        selected_role = self.subtitle_combo.currentText()
        role_value = None if selected_role == "No" else selected_role

        for row in range(self.table.rowCount()):
            chk_item = self.table.item(row, 0)
            if chk_item and chk_item.checkState() == Qt.Checked:
                self.display_data[row]['role'] = role_value
                chk_item.setCheckState(Qt.Unchecked)
        
        self._update_role_column()

    def replace_text(self):
        """替换文本"""
        search_text = self.search_input.text()
        replace_text = self.replace_input.text()

        if not search_text:
            return

        self.table.setUpdatesEnabled(False)  # 禁用更新，提升性能
        
        for row, data in enumerate(self.display_data):
            if search_text in data['text']:
                new_text = data['text'].replace(search_text, replace_text)
                data['text'] = new_text
                item = self.table.item(row, 5)
                if item:
                    item.setText(new_text)
        
        self.table.setUpdatesEnabled(True)  # 恢复更新

    def listen_dubbing(self):
        """试听配音"""
        selected_role = self.subtitle_combo.currentText()
        role_value = None if selected_role == "No" else selected_role
        if not role_value:
            return

        first_text = self.display_data[0]['text'] if self.display_data else ''
        if not first_text:
            return

        from videotrans.util.ListenVoice import ListenVoice
        
        def feed(d):
            self.listen_button.setText(tr("Trial dubbing"))
            self.listen_button.setDisabled(False)
            if d != "ok":
                tools.show_error(d)

        wk = ListenVoice(parent=self, queue_tts=[{
            "text": first_text,
            "role": role_value,
            "filename": TEMP_DIR + f"/{time.time()}-onlyone_setrole.wav",
            "tts_type": self.tts_type}],
            language=self.target_language,
            tts_type=self.tts_type)
        wk.uito.connect(feed)
        wk.start()
        self.listen_button.setText('Listening...')
        self.listen_button.setDisabled(True)

    def _active(self):
        if self.parent:
            self.parent.activateWindow()

    def cancel_and_close(self):
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()
        self.reject()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.ignore()
        else:
            super().keyPressEvent(event)

    def update_countdown(self):
        self.count_down -= 1
        if self.stop_button and hasattr(self.stop_button, 'setText'):
            self.stop_button.setText(f"{tr('Click here to stop the countdown')}({self.count_down})")
        if self.count_down <= 0:
            self.timer.stop()
            self.save_and_close()

    def stop_countdown(self):
        if hasattr(self, 'timer'):
            self.timer.stop()
        self.stop_button.deleteLater()
        self.prompt_label.deleteLater()

    def save_and_close2(self):
        self.accept()

    def opendir_sub(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(Path(self.target_sub).parent.as_posix()))


    def closeEvent(self, event):
        event.ignore()  # 忽略关闭请求，窗口保持不动
    
    def save_and_close(self):
        self.save_button.setDisabled(True)
        app_cfg.line_roles = {}
        srt_str_list = []

        speaker_keys = list(self.speakers.keys()) if self.speakers else []
        default_spk = speaker_keys[0] if speaker_keys else ''

        for row, data in enumerate(self.display_data):
            # 获取当前文本（从表格中获取最新值）
            text_item = self.table.item(row, 5)
            text = text_item.text().strip() if text_item else data['text'].strip()
            
            srt_str_list.append(f'{data["line"]}\n{data["startraw"]} --> {data["endraw"]}\n{text}')

            # 角色保存逻辑
            role = data.get('role', '')
            if not role and self.speakers and data['spk']:
                role = self.speakers.get(data['spk'], '')

            if role:
                app_cfg.line_roles[str(data["line"])] = role

        try:
            Path(self.target_sub).write_text("\n\n".join(srt_str_list), encoding="utf-8")
        except Exception as e:
            logger.error(f"Save subtitle failed: {e}")
            QMessageBox.critical(self, "Error", f"Save failed: {e}")
            self.save_button.setDisabled(False)
            return

        # M2: 持久化说话人 → 库角色名映射, 供 Step5 读取建库/复用
        # 空名 = 不入库 (决策 4), 完全不写入该 spk_id
        if self.drama_dir is not None and self.speaker_name_edits:
            spk_to_char = {}
            for spk_id, edit in self.speaker_name_edits.items():
                name = (edit.text() or '').strip()
                if name:
                    spk_to_char[spk_id] = name
            try:
                mapping_path = Path(f'{self.cache_folder}/spk_to_character.json')
                mapping_path.write_text(json.dumps({
                    'drama_dir': str(self.drama_dir),
                    'mapping': spk_to_char,
                }, ensure_ascii=False, indent=2), encoding='utf-8')
                logger.info(f'[voice_library] 写入 spk→character 映射 {len(spk_to_char)} 条: {mapping_path}')
            except Exception as e:
                logger.warning(f'[voice_library] 写入 spk_to_character.json 失败: {e}')

            # S3: 把每个有名字的 speaker 的当前固定音色回写 drama.json, 实现跨集音色一致
            # 固定音色模式 (tts_type != clone) 才写; clone 模式音色无意义, 跳过
            try:
                from videotrans.util.voice_library import set_fixed_voice
                # clone-voice tts_type 常量不一, 用启发式: 当前 speaker 的音色是 'clone' 就视为 clone 模式, 不写
                for spk_id, name in spk_to_char.items():
                    voice = self.speakers.get(spk_id) or ''
                    if not voice or voice in self._AUTO_SKIP_VOICES:
                        # 未选有效音色: 不清也不写, 避免误覆盖旧绑定
                        continue
                    set_fixed_voice(self.drama_dir, name, voice)
                logger.info(f'[voice_library] 回写 fixed_voice 完成 ({len(spk_to_char)} 角色)')
            except Exception as e:
                logger.warning(f'[voice_library] 回写 fixed_voice 失败: {e}')

            # L2-S4: 回写 embedding 到 drama.json, 下集批量模式就能自动匹配
            # 读本集 speaker_refs.json 里的 spkN_ref.wav → 提 embedding → 存进角色条目
            # 已有 embedding 的角色不覆盖 (首次录的声纹通常最纯净)
            try:
                from videotrans.util.voice_library import load_drama, set_embedding
                from videotrans.util.speaker_embedding import compute_embedding
                refs_json = Path(f'{self.cache_folder}/speaker_refs.json')
                if refs_json.exists() and spk_to_char:
                    ref_info = json.loads(refs_json.read_text(encoding='utf-8'))
                    drama_data = load_drama(self.drama_dir)
                    existing_chars = drama_data.get('characters', {})
                    written = 0
                    for spk_id, char_name in spk_to_char.items():
                        # 已有 embedding 的角色不重写 (保护首次纯净声纹)
                        if existing_chars.get(char_name, {}).get('embedding'):
                            continue
                        wav = (ref_info.get(spk_id) or {}).get('wav', '')
                        if not wav or not Path(wav).exists():
                            continue
                        emb = compute_embedding(wav)
                        if not emb:
                            continue
                        set_embedding(self.drama_dir, char_name, emb)
                        written += 1
                    if written:
                        logger.info(f'[voice_library] 录入 {written} 个新角色声纹')
            except Exception as e:
                logger.warning(f'[voice_library] 录声纹失败: {e}')

        self.accept()
