import argparse
import re
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}

def find_videos(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

def extract_frames(video_path: Path, output_dir: Path, interval: float, start_index: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Không mở được: {video_path.name}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0 or frame_count <= 0:
        print(f"Không đọc được thông tin video: {video_path.name}")
        cap.release()
        return 0

    duration = frame_count / fps
    saved = 0
    timestamp = 0.0

    while timestamp <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        out = output_dir / f"anh_{start_index + saved + 1}.jpg"
        cv2.imwrite(str(out), frame)
        saved += 1
        timestamp += interval

    cap.release()
    print(f"[+] {video_path.name} → {saved} ảnh")
    return saved

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default="Video_Normal")
    parser.add_argument("--output",   default="Normal")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise SystemExit(f"Không tìm thấy thư mục: {input_dir}")
    if args.interval <= 0:
        raise SystemExit("Interval phải > 0")

    videos = find_videos(input_dir)
    if not videos:
        print(f"Không có video nào trong: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for v in videos:
        total += extract_frames(v, output_dir, args.interval, start_index=total)

    print(f"\n{total} ảnh → '{output_dir.resolve()}'")

if __name__ == "__main__":
    main()