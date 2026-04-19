# Video Subtitle Remover 集成说明

当前项目通过 `tools/remove_subtitles_cli.py` 调用 `tools/video-subtitle-remover-src`
中的 `backend.main.SubtitleRemover`，并使用独立 Python 环境
`tools/video-subtitle-remover-env` 运行。这样可以避免 VSR 的 Paddle、Torch、OpenCV、
PySide6 等重依赖污染 pyVideoTrans 主程序环境。

## 推荐发布形态

给普通用户发布时，不要求用户安装 Python、Conda、Paddle、Torch 或编译工具。发布包里应该已经包含：

- `pyVideoTrans.exe`
- `ffmpeg/`
- `tools/remove_subtitles_cli.py`
- `tools/video-subtitle-remover-src/`
- `tools/video-subtitle-remover-env/`
- `models/`、`output/`、`logs/`、`tmp/` 等运行目录

`build_win.bat` 已经在 PyInstaller 打包完成后复制这些 VSR 目录到
`dist/pyVideoTrans/tools/`。最终把 `dist/pyVideoTrans` 整个目录压缩或做成安装包即可。

## 构建机准备

构建机需要提前准备好 VSR 独立运行时：

1. 将 VSR 源码放到 `tools/video-subtitle-remover-src/`。
2. 将可移植的 VSR Python 环境放到 `tools/video-subtitle-remover-env/`。
3. 确认存在 `tools/video-subtitle-remover-env/Scripts/python.exe`。
4. 确认 `tools/remove_subtitles_cli.py` 可以在本机调用 VSR 处理测试视频。

这些目录体积很大，不建议提交到 Git；它们只作为本地构建资源和最终发布包资源。

注意：普通 `python -m venv` 创建的环境通常不是完整可移植环境，它可能依赖创建机器上的
`C:\Users\...\miniconda3` 或系统 Python。给新电脑发布前，建议使用官方 VSR release 的
Windows 预构建包，或用 `conda-pack`/便携 Python 重新整理成真正可移动的运行时。一个基本判断是：
`video-subtitle-remover-env` 内应带有 Python 标准库、Python DLL 和运行依赖 DLL，而不是只带
`Scripts/python.exe` 与 `Lib/site-packages`。

## 用户侧体验

用户只需要打开 `pyVideoTrans.exe`，在主流程里勾选“写新字幕前去旧硬字幕”，再正常开始翻译/配音任务。
程序会在最终写入新字幕之前，先对无声视频底片执行 VSR 去硬字幕，然后再把新字幕和新音频合成进去。
这样不会影响原视频的音频提取、识别和配音流程。

工具菜单里的“删除硬字幕/去除硬字幕”仍可作为独立批处理入口使用。

如果新电脑没有 NVIDIA 驱动，CUDA 版环境可能无法工作。面向不懂编程的用户，建议优先发布 CPU 版或 DirectML 版；
如果要发布 CUDA 版，安装包说明里应写明需要安装对应显卡驱动。

## 验证清单

构建完成后，在一台没有 Python 开发环境的新 Windows 电脑或干净虚拟机上验证：

1. 解压或安装发布包。
2. 运行 `pyVideoTrans.exe`。
3. 主界面选择一段短视频，选择一种字幕输出方式，勾选“写新字幕前去旧硬字幕”。
4. 输出目录中生成旧硬字幕已去除、并写入新字幕的视频。
5. 日志中没有 `python.exe not found`、`No module named paddle`、`No module named torch` 等错误。
