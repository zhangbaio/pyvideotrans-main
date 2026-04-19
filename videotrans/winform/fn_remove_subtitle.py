def openwin():
    import json
    import os
    import subprocess
    from pathlib import Path

    from PySide6.QtCore import QThread, Signal, QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

    from videotrans.configure.config import ROOT_DIR, HOME_DIR, app_cfg, params
    from videotrans.util import contants, tools

    class RemoveSubtitleThread(QThread):
        uito = Signal(str)

        def __init__(self, *, videos, output_dir, area):
            super().__init__()
            self.videos = videos
            self.output_dir = output_dir
            self.area = area

        def post(self, type="logs", text=""):
            self.uito.emit(json.dumps({"type": type, "text": text}, ensure_ascii=False))

        def _python_bin(self):
            candidates = [
                Path(ROOT_DIR) / "tools" / "video-subtitle-remover-env" / "python.exe",
                Path(ROOT_DIR) / "tools" / "video-subtitle-remover-env" / "Scripts" / "python.exe",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return None

        def _installed_app(self):
            candidates = [
                Path(r"D:/Program Files (x86)/Program Files/视频字幕去除器/VideoSubtitleRemover.exe"),
                Path(r"D:/Program Files (x86)/Program Files/Video Subtitle Remover/VideoSubtitleRemover.exe"),
                Path(ROOT_DIR) / "tools" / "video-subtitle-remover" / "VideoSubtitleRemover.exe",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return None

        def _subprocess_env(self, python_bin):
            env = os.environ.copy()
            runtime_dir = python_bin.parent.parent if python_bin.parent.name.lower() == "scripts" else python_bin.parent
            path_parts = [
                str(python_bin.parent),
                str(runtime_dir),
                str(runtime_dir / "Library" / "bin"),
                str(runtime_dir / "DLLs"),
            ]
            env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
            return env

        def run(self):
            python_bin = self._python_bin()
            if not python_bin:
                installed_app = self._installed_app()
                if installed_app:
                    try:
                        subprocess.Popen([str(installed_app)], cwd=str(installed_app.parent))
                        self.post(
                            "manual",
                            f"已打开本地安装版：{installed_app}\n"
                            "当前未检测到该安装版可用的命令行参数，请在软件内手动选择视频处理。"
                        )
                        return
                    except Exception as e:
                        self.post("error", f"打开本地安装版失败：{e}")
                        return
                self.post(
                    "error",
                    "未找到 video-subtitle-remover-env 的 python.exe，也未找到本地安装版 VideoSubtitleRemover.exe。",
                )
                return

            cli = Path(ROOT_DIR) / "tools" / "remove_subtitles_cli.py"
            for index, video in enumerate(self.videos, start=1):
                self.post("logs", f"[{index}/{len(self.videos)}] 开始删除字幕: {Path(video).name}")
                cmd = [
                    str(python_bin),
                    str(cli),
                    "--input",
                    video,
                    "--output-dir",
                    self.output_dir,
                ]
                if self.area:
                    cmd.extend(["--area", self.area])
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=ROOT_DIR,
                        env=self._subprocess_env(python_bin),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    if proc.returncode != 0:
                        self.post("error", (proc.stderr or proc.stdout or "删除字幕失败").strip())
                        return
                    self.post("logs", (proc.stdout or "").strip() or f"{Path(video).name} 完成")
                except Exception as e:
                    self.post("error", str(e))
                    return
            self.post("ok", self.output_dir)

    def select_and_start():
        format_str = " ".join(["*." + f for f in contants.VIDEO_EXTS])
        videos, _ = QFileDialog.getOpenFileNames(
            None,
            "选择需要删除硬字幕的视频",
            params.get("last_opendir", ""),
            f"Video files({format_str})",
        )
        if not videos:
            return
        output_dir = QFileDialog.getExistingDirectory(
            None,
            "选择输出目录",
            f"{HOME_DIR}/remove_subtitle",
        )
        if not output_dir:
            return
        area, ok = QInputDialog.getText(
            None,
            "字幕区域",
            "可选：输入字幕区域 ymin,ymax,xmin,xmax；留空则自动检测/全屏处理",
            text="",
        )
        if not ok:
            return

        msg = QMessageBox()
        msg.setWindowTitle("删除硬字幕")
        msg.setText("任务已开始，完成后会提示。处理时间可能较长。")
        msg.show()

        task = RemoveSubtitleThread(
            videos=[v.replace("\\", "/") for v in videos],
            output_dir=output_dir.replace("\\", "/"),
            area=area.strip(),
        )

        def feed(raw):
            data = json.loads(raw)
            if data["type"] == "error":
                tools.show_error(data["text"])
            elif data["type"] == "manual":
                QMessageBox.information(None, "删除硬字幕", data["text"])
            elif data["type"] == "ok":
                QMessageBox.information(None, "删除硬字幕", "全部处理完成")
                QDesktopServices.openUrl(QUrl.fromLocalFile(data["text"]))

        task.uito.connect(feed)
        task.start()
        app_cfg.child_forms["fn_remove_subtitle_task"] = task

    select_and_start()
