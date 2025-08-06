# extract_landmarks.py
import json
import cv2
from pathlib import Path
import mediapipe as mp

# 1️⃣ Folders
INPUT_DIR  = Path("all_frames")            # where your JPG frames are
OUTPUT_DIR = Path("landmark_data_big")     # where JSONs will go

# 2️⃣ Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def process_frame(frame_path: Path):
    img = cv2.imread(str(frame_path))
    if img is None:
        return

    # Convert to RGB and run pose detection
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Serialize landmarks
    lm_list = None
    if results.pose_landmarks:
        lm_list = [
            {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
            for lm in results.pose_landmarks.landmark
        ]

    # Build output path that mirrors input structure
    rel_path = frame_path.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / rel_path.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump({"image_path": str(frame_path), "landmarks": lm_list}, f)

    status = "found" if lm_list else "missing"
    print(f"→ {rel_path}: landmarks {status}")


def main():
    for frame_path in INPUT_DIR.rglob("*.jpg"):
        process_frame(frame_path)

if __name__ == "__main__":
    main()
