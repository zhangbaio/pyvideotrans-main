import re
from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar

from videotrans.configure.config import tr
from videotrans.util import tools


class ClickableProgressBar(QLabel):
    def __init__(self, parent=None):
        super().__init__()
        self.target_dir = None
        self.msg = tr("running")
        self.parent = parent
        self.basename = ""
        self.name = ""
        self.precent = 0
        self.duration = 0
        self.paused = False
        self.ended = False
        self.error = ""

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedHeight(35)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                background-color: transparent;
                border:1px solid #32414B;
                color:#fff;
                height:35px;
                text-align:left;
                border-radius:3px;
            }
            QProgressBar::chunk {
                width: 8px;
                border-radius:0;
            }
            """
        )
        layout = QHBoxLayout(self)
        layout.addWidget(self.progress_bar)

    def _format_seconds(self, value):
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return ""
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins, secs = divmod(int(round(seconds)), 60)
        if mins < 60:
            return f"{mins}m{secs:02d}s"
        hours, mins = divmod(mins, 60)
        return f"{hours}h{mins:02d}m{secs:02d}s"

    def _build_timing_tooltip(self, timing):
        if not isinstance(timing, dict):
            return ""
        lines = []
        total_sec = timing.get("total_sec")
        if total_sec is not None:
            lines.append(f"total: {self._format_seconds(total_sec)}")
        for name, sec in (timing.get("stages") or {}).items():
            lines.append(f"{name}: {self._format_seconds(sec)}")
        return "\n".join(lines)

    def setTarget(self, target_dir=None, name=None):
        self.target_dir = target_dir
        self.name = name
        self.basename = Path(name).name

    def setEnd(self, total_sec=None, timing=None):
        if self.error:
            return
        self.ended = True
        self.precent = 100
        self.progress_bar.setValue(100)
        self.setCursor(Qt.PointingHandCursor)
        total_text = self._format_seconds(total_sec) if total_sec is not None else ""
        suffix = f" | {total_text}" if total_text else ""
        self.progress_bar.setFormat(f" {self.basename}  {tr('endandopen')}{suffix}")
        tooltip = self._build_timing_tooltip(timing)
        if tooltip:
            self.progress_bar.setToolTip(tooltip)
        self.error = ""

    def setPause(self):
        if not self.ended:
            self.paused = True
            self.progress_bar.setFormat(f"  {tr('haspaused')} [{self.precent}%] {self.basename}")

    def setPrecent(self, p):
        self.paused = False
        if p >= 100:
            self.precent = 100
            self.error = ""
            self.setEnd()
        else:
            self.precent = p if p > self.precent else self.precent
            self.progress_bar.setValue(self.precent)
            self.progress_bar.setFormat(f" [{self.precent}%  {self.duration}]  {self.msg} {self.basename}")

    def setError(self, text=""):
        self.error = text
        self.ended = True
        self.progress_bar.setToolTip(tr("Click to view the detailed error report"))
        self.progress_bar.setFormat(f"  [{self.precent}%]  {text[:90]}   {self.basename}")

    def setText(self, text=""):
        if self.progress_bar:
            if self.ended or self.paused:
                return
            if re.match(r"^\d+?$", text):
                self.duration = text
                if self.precent < 60 and int(text) % 5 == 0:
                    self.precent += 0.001
            elif text.strip():
                self.msg = text[:150].replace("\n", "")
            self.progress_bar.setFormat(f" [{self.precent:.2f}%  {self.duration}s]  {self.msg} {self.basename}")

    def mousePressEvent(self, event):
        if self.target_dir and event.button() == Qt.LeftButton:
            if self.error:
                tools.show_error(self.error)
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.target_dir))
