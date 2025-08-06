import cv2
import os
import json
import time
import mediapipe as mp

# ——— 1. Setup Mediapipe Pose and Output Folder ———
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing = mp.solutions.drawing_utils

output_dir = "landmark_data"
os.makedirs(output_dir, exist_ok=True)

# ——— 2. Open Webcam and Initialize Counter ———
cap = cv2.VideoCapture(0)  # 0 = default webcam
frame_idx = 0

# ——— 3. Main Loop: Capture, Process, Save ———
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⛔ Could not read frame. Exiting...")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # a) Serialize JSON data
        timestamp = time.time()
        data = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "landmarks": [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in results.pose_landmarks.landmark
            ]
        }

        # b) Save JSON
        json_filename = f"frame_{frame_idx:05d}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(data, f)

        # c) Save image frame for later overlay
        img_filename = f"frame_{frame_idx:05d}.jpg"
        img_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(img_path, frame)

    else:
        print(f"[Warning] No landmarks in frame {frame_idx}")

    # d) Draw landmarks
    drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    # e) Display
    cv2.imshow("Pose Capture", frame)

    # f) Increment frame index and allow exit
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exit requested. Stopping capture.")
        break

# ——— 4. Cleanup ———
cap.release()
cv2.destroyAllWindows()
pose.close()
print("✅ Done. JSON + JPG saved in folder:", output_dir)