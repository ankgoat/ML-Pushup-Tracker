# extract_frames.py
import cv2
from pathlib import Path

# 1️⃣ Where your split videos live (with train/val/test and good/bad)
INPUT_DIR  = Path("raw_pushup_videos")
# 2️⃣ Where to dump extracted frames:
OUTPUT_DIR = Path("all_frames")
# 3️⃣ How many frames per second to grab:
FPS = 10  # every 0.1s

def extract_from_clip(video_path: Path):
    clip_name = f"{video_path.parent.name}_{video_path.stem}"
    out_folder = OUTPUT_DIR/clip_name
    out_folder.mkdir(parents=True, exist_ok=True)

    vid = cv2.VideoCapture(str(video_path))
    orig_fps = vid.get(cv2.CAP_PROP_FPS) or FPS
    frame_interval = max(1, int(orig_fps // FPS))

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            filename = out_folder / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved += 1
        frame_idx += 1
    vid.release()
    print(f"→ {video_path.parent.name}/{video_path.name}: saved {saved} frames")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        for label in ("good", "bad"):
            folder = INPUT_DIR / split / label
            for video in folder.glob("*.mp4"):
                extract_from_clip(video)

if __name__ == "__main__":
    main()
