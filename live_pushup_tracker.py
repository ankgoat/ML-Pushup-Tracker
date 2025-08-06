# live_pushup_tracker.py

import cv2
import mediapipe as mp
import joblib
from collections import deque
from feature_engineer import extract_features

# ── 1) Load artifacts from models/ ──
scaler = joblib.load("models/scaler.joblib")
clf    = joblib.load("models/logistic_model.joblib")

# ── 2) Smoothing & rep-counting setup ──
SMOOTHING_WINDOW = 5
pred_queue = deque(maxlen=SMOOTHING_WINDOW)
rep_count  = 0
down       = False  # flag to detect downward phase

# ── 3) MediaPipe Pose setup ──
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def main():
    global rep_count, down

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror feed so it feels more natural
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(rgb)

        if not results.pose_landmarks:
            cv2.putText(frame, "No person detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )

            # Extract landmarks as dicts for your feature function
            landmarks = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z}
                for lm in results.pose_landmarks.landmark
            ]

            # Compute features, scale, predict
            X = extract_features(landmarks)
            Xs = scaler.transform([X])
            y  = clf.predict(Xs)[0]  # 1 = good, 0 = bad

            # Smoothing: majority vote over last SMOOTHING_WINDOW frames
            pred_queue.append(y)
            vote = 1 if sum(pred_queue) > len(pred_queue) / 2 else 0

            # Overlay “Good Form” / “Bad Form”
            label = "Good Form" if vote == 1 else "Bad Form"
            color = (0,255,0) if vote == 1 else (0,0,255)
            cv2.putText(frame, label, (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Rep counting: detect when wrist_y moves down then up
            wrist_y = X[-1]  # last feature is wrist_y
            THR_HIGH = 0.6
            THR_LOW  = 0.4

            if not down and wrist_y > THR_HIGH:
                down = True
            if down and wrist_y < THR_LOW:
                rep_count += 1
                down = False

        # Always display rep count
        cv2.putText(frame, f"Reps: {rep_count}", (10,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        cv2.imshow("MVP Push-Up Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
