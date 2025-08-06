import pandas as pd
import json
import argparse
from pathlib import Path
import numpy as np

def angle(a, b, c):
    v1 = np.array([a['x'] - b['x'], a['y'] - b['y']])
    v2 = np.array([c['x'] - b['x'], c['y'] - b['y']])
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-6:
        return np.nan
    cos_ = np.dot(v1, v2) / norm
    return float(np.degrees(np.arccos(np.clip(cos_, -1, 1))))

def compute_features(landmarks):
    if landmarks is None:
        return {
            'elbow_mean': np.nan, 'elbow_std': np.nan,
            'shoulder_mean': np.nan, 'shoulder_std': np.nan,
            'hip_mean': np.nan, 'hip_std': np.nan,
            'wrist_y': np.nan
        }

    # key joints
    L_SH, L_EL, L_WR = landmarks[11], landmarks[13], landmarks[15]
    R_SH, R_EL, R_WR = landmarks[12], landmarks[14], landmarks[16]
    L_HI, L_KN       = landmarks[23], landmarks[25]
    R_HI, R_KN       = landmarks[24], landmarks[26]

    elbow_angles    = [angle(L_SH, L_EL, L_WR), angle(R_SH, R_EL, R_WR)]
    shoulder_angles = [angle(L_HI, L_SH, L_EL), angle(R_HI, R_SH, R_EL)]
    hip_angles      = [angle(L_KN, L_HI, L_SH), angle(R_KN, R_HI, R_SH)]
    wrist_y         = float((L_WR['y'] + R_WR['y']) / 2)

    return {
        'elbow_mean':     np.mean(elbow_angles),
        'elbow_std':      np.std(elbow_angles),
        'shoulder_mean':  np.mean(shoulder_angles),
        'shoulder_std':   np.std(shoulder_angles),
        'hip_mean':       np.mean(hip_angles),
        'hip_std':        np.std(hip_angles),
        'wrist_y':        wrist_y
    }

def extract_features(landmarks):
    feat_dict = compute_features(landmarks)
    keys = [
        'elbow_mean', 'elbow_std',
        'shoulder_mean', 'shoulder_std',
        'hip_mean', 'hip_std',
        'wrist_y'
    ]
    return [feat_dict[k] for k in keys]

def main():
    parser = argparse.ArgumentParser(description="Compute features from landmark JSONs.")
    parser.add_argument("--in-csv",  required=True, help="(unused, for compatibility)")
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    args = parser.parse_args()

    labels_df = pd.read_csv("landmark_data_big/labels_clean.csv")
    all_feats = []
    for _, row in labels_df.iterrows():
        json_path = Path(row['frame_path'])
        try:
            data = json.loads(json_path.read_text())
            landmarks = data.get('landmarks')
        except Exception:
            landmarks = None

        label_value = str(row['label']).strip().lower()
        feats = compute_features(landmarks)
        feats['label'] = 1 if label_value == 'good' else 0
        all_feats.append(feats)

    # —— FIX: create df first, then fillna —— 
    df = pd.DataFrame(all_feats)
    df.fillna(df.median(), inplace=True)

    df.to_csv(args.out_csv, index=False)

if __name__ == '__main__':
    main()
