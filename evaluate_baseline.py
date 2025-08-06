import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a validation set.")
    parser.add_argument("--val-csv", required=True, help="Path to validation CSV")
    parser.add_argument("--model", required=True, help="Trained model file (.joblib)")
    parser.add_argument("--scaler", required=False, help="Optional: Scaler file (.joblib)")
    args = parser.parse_args()

    clf = joblib.load(args.model)
    val = pd.read_csv(args.val_csv)
    features = [c for c in val.columns if c != 'label']
    X_val, y_val = val[features], val['label']

    if args.scaler:
        scaler = joblib.load(args.scaler)
        X_val = scaler.transform(X_val)

    y_pred = clf.predict(X_val)
    print("=== Classification Report ===")
    print(classification_report(y_val, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_val, y_pred))

if __name__ == "__main__":
    main()
