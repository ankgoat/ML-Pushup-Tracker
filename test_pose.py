import cv2
import os
import json
import time
import mediapipe as mp

# ——— 1. Setup Mediapipe Pose and Output Folder ———
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

    # a) Convert to RGB for Mediapipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # b) Run pose detection
    results = pose.process(img_rgb)

    # c) If landmarks detected, serialize & save
    if results.pose_landmarks:
        data = {
            "frame_idx": frame_idx,
            "timestamp": time.time(),
            "landmarks": [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in results.pose_landmarks.landmark
            ]
        }
        filename = f"frame_{frame_idx:05d}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f)
    else:
        # Optional: log when no person is found
        print(f"[Warning] No landmarks in frame {frame_idx}")

    # d) Draw pose overlay on the frame for your reference
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    # e) Show the frame
    cv2.imshow("Pose Capture", frame)

    # f) Increment counter and handle quit key
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exit requested. Stopping capture.")
        break

# ——— 4. Cleanup ———
cap.release()
cv2.destroyAllWindows()
pose.close()
print("✅ Done. Landmarks saved in folder:", output_dir)
