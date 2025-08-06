ML Push‑Up Form Tracker

A complete end‑to‑end pipeline for extracting pose landmarks from webcam video, engineering features, training a push‑up form classifier, and running a live real‑time tracker with rep counting and colored skeleton feedback.

🚀 Features

Real‑time Pose Extraction: Uses MediaPipe to capture 33 body landmarks from your webcam stream.

On‑the‑fly Feature Engineering: Computes joint angles and positional metrics each frame.

Trainable Classifier: Logistic Regression (or replaceable) that learns Good Form vs. Bad Form from labeled data.

Live Inference & Visualization:

Overlays skeleton in green (good) or red (bad) form.

Displays live rep counter using elbow‑angle state machine.

Modular Scripts: Separate stages for feature extraction, dataset splitting, training, evaluation, and live demo.

📂 Project Structure

ML_FormCorrectionProject/
├── landmark_data_big/labels_clean.csv    # CSV mapped to landmark JSONs
├── feature_engineer.py                   # `compute_features` + `extract_features`
├── split_dataset.py                      # train/val/test split CLI
├── train_model.py                        # fits scaler + classifier → models/
├── evaluate_baseline.py                  # prints accuracy on validation set
├── live_pushup_tracker.py                # real‑time demo script
├── models/
│   ├── logistic_model.joblib             # trained classifier
│   └── scaler.joblib                     # trained StandardScaler
├── requirements.txt                      # python dependencies
└── .gitignore                            # ignores venv, data, large files

🛠️ Setup & Installation

Clone the repo:

git clone https://github.com/ankgoat/ML-Pushup-Tracker.git
cd ML-Pushup-Tracker

Create & activate a virtual environment (Windows example):

python -m venv venv
.\venv\Scripts\Activate.ps1

Install dependencies:

pip install --upgrade pip
pip install -r requirements.txt

⬇️ Download the Kaggle Dataset

This project uses a pre‑annotated push‑up landmark dataset on Kaggle (mohamadashrafsalama/pushup). To download it:

Install the Kaggle CLI and place your API token in the config folder:

pip install kaggle
mkdir -p ~/.kaggle
# Copy `kaggle.json` into ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

Download directly into landmark_data_big/ and unzip:

kaggle datasets download -d mohamadashrafsalama/pushup \
  -p landmark_data_big --unzip

Confirm the folder now contains your JSON landmark files and labels_clean.csv.

📊 Data Pipeline

1. Feature Extraction

Extract joint angles and positional features from labeled landmark JSONs:

python feature_engineer.py \
  --in-csv  landmark_data_big/labels_clean.csv \
  --out-csv features.csv

Reads frame_path JSONs from labels_clean.csv.

Outputs features.csv with 7 numeric features + label (0/1).

2. Train/Val/Test Split

Split features into train, validation, and test sets:

python split_dataset.py \
  --in-csv    features.csv \
  --train-out train.csv \
  --val-out   val.csv \
  --test-out  test.csv

🚂 Model Training & Evaluation

3. Train Model

Fit a logistic regression and save both model and scaler:

python train_model.py \
  --train-csv  train.csv \
  --val-csv    val.csv \
  --model-out  models/logistic_model.joblib \
  --scaler-out models/scaler.joblib

4. Evaluate Baseline

Check validation accuracy:

python evaluate_baseline.py \
  --train-csv train.csv \
  --val-csv   val.csv \
  --model     models/logistic_model.joblib \
  --scaler    models/scaler.joblib

🎥 Live Real‑Time Tracking

Launch the live push‑up form tracker:

python live_pushup_tracker.py

Green skeleton when form is classified as Good Form.

Red skeleton when classified as Bad Form.

Reps counter increments once per full down→up cycle using elbow angle thresholds.

Configurable Parameters

Elbow thresholds in live_pushup_tracker.py:

ELBOW_DOWN_ANGLE = 90   # degrees below which "down" is detected
ELBOW_UP_ANGLE   = 160  # degrees above which rep is counted

🔧 Customization & Extension

Swap in a more powerful classifier (RandomForest, XGBoost).

Add new features: torso angle, joint velocities, symmetry metrics.

Replace OpenCV UI with a Streamlit/Flask web front‑end.

Package as a desktop app via PyInstaller or as a pip‑installable library.

🤝 Contributing

Fork the repo and create a feature branch.

Open a PR with your changes and a clear description.

Ensure all existing scripts run without errors.

📜 License

MIT License © Ankith Goswami

