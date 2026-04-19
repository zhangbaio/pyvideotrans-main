import argparse
import os
import shutil
import sys
from pathlib import Path


def _parse_area(value):
    if not value:
        return None
    parts = [int(float(p.strip())) for p in value.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("area must be ymin,ymax,xmin,xmax")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description="Remove hard subtitles using bundled VSR")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--area", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache_home = root / "tmp" / "video-subtitle-remover-cache"
    cache_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(cache_home)
    os.environ["USERPROFILE"] = str(cache_home)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_home / "paddlex")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    src_dir = root / "tools" / "video-subtitle-remover-src"
    if not src_dir.exists():
        raise FileNotFoundError(f"VSR source not found: {src_dir}")

    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(src_dir / "backend"))

    from backend.main import SubtitleRemover

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    remover = SubtitleRemover(str(input_path), sub_area=_parse_area(args.area), gui_mode=False)
    remover.run()

    generated = Path(remover.video_out_name)
    if not generated.exists():
        raise RuntimeError(f"VSR did not generate output: {generated}")

    output_path = output_dir / generated.name
    if generated.resolve() != output_path.resolve():
        shutil.move(str(generated), str(output_path))
    print(f"[Finished] output={output_path}")


if __name__ == "__main__":
    main()
