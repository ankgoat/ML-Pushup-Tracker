# live_pushup_tracker.py

import cv2
import mediapipe as mp
import joblib
from feature_engineer import extract_features, compute_features

# ── 1) Load model artifacts ──
scaler = joblib.load("models/scaler.joblib")
clf    = joblib.load("models/logistic_model.joblib")

# ── 2) Rep‐counting state & thresholds ──
rep_count        = 0
down              = False
ELBOW_DOWN_ANGLE  = 90   # below this => consider "down"
ELBOW_UP_ANGLE    = 160  # above this => count rep

# ── 3) MediaPipe Pose setup ──
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# ── 4) DrawingSpecs for coloring ──
GOOD_LND  = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4)
GOOD_CONN = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
BAD_LND   = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4)
BAD_CONN  = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)

# ── 5) Main loop ──
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

        # Mirror & color convert
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(rgb)

        if results.pose_landmarks:
            # Convert landmarks to dicts
            lm_dicts = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z}
                for lm in results.pose_landmarks.landmark
            ]

            # 1) Rep counting based on elbow angle
            feat_dict  = compute_features(lm_dicts)
            elbow_mean = feat_dict.get('elbow_mean', 180.0) or 180.0
            if not down and elbow_mean < ELBOW_DOWN_ANGLE:
                down = True
            elif down and elbow_mean > ELBOW_UP_ANGLE:
                rep_count += 1
                down = False

            # 2) Classification of form
            X   = extract_features(lm_dicts)
            Xs  = scaler.transform([X])
            y   = clf.predict(Xs)[0]  # 1 = good form, 0 = bad form

            # 3) Select color specs
            if y == 1:
                lnd_spec, conn_spec = GOOD_LND, GOOD_CONN
                text, text_color   = "Good Form", (0,255,0)
            else:
                lnd_spec, conn_spec = BAD_LND, BAD_CONN
                text, text_color   = "Bad Form",  (0,0,255)

            # 4) Draw skeleton in live color
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                POSE_CONNECTIONS,
                landmark_drawing_spec   = lnd_spec,
                connection_drawing_spec = conn_spec
            )

            # 5) Overlay form label
            cv2.putText(frame, text, (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
        else:
            # Show red default skeleton? cannot draw without landmarks
            cv2.putText(frame, "No person detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Always show rep count
        cv2.putText(frame, f"Reps: {rep_count}", (10,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        cv2.imshow("Push-Up Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
