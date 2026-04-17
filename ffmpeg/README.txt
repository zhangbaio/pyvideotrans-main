ffmpeg 二进制按平台放置, 不入 git 仓库。

Windows 打包前:
  1. 到 https://www.gyan.dev/ffmpeg/builds/ 下载 ffmpeg-release-essentials.zip
  2. 解压, 把 bin/ 里的 ffmpeg.exe、ffprobe.exe、ffplay.exe 复制到本目录
  3. 可选: 也放 ytwin32.exe (yt-dlp.exe 重命名, 用于在线视频下载)

macOS/Linux 本地开发:
  brew install ffmpeg   (macOS)
  apt install ffmpeg    (Ubuntu)
  或把 ffmpeg 可执行文件放本目录, 文件名为 ffmpeg (无扩展)
