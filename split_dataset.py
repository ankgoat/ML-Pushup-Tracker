# split_dataset.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Split a features CSV into train/val/test sets
def main():
    parser = argparse.ArgumentParser(description="Split features CSV into train/val/test sets.")
    parser.add_argument(
        "--in-csv", required=True,
        help="Input features CSV (e.g., big_dataset_features.csv)"
    )
    parser.add_argument(
        "--train-out", default="train.csv",
        help="Output path for the train split"
    )
    parser.add_argument(
        "--val-out", default="val.csv",
        help="Output path for the validation split"
    )
    parser.add_argument(
        "--test-out", default="test.csv",
        help="Output path for the test split"
    )
    args = parser.parse_args()

    # Load the CSV specified on the command line
    df = pd.read_csv(args.in_csv)

    # 70% train, 30% holdout
    train_df, holdout_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df.label,
        random_state=42
    )

    # Split holdout equally into validation and test (each ~15% of total)
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=0.5,
        stratify=holdout_df.label,
        random_state=42
    )

    # Save output files
    train_df.to_csv(args.train_out, index=False)
    val_df.to_csv(args.val_out,     index=False)
    test_df.to_csv(args.test_out,   index=False)

    print(f"✔️ Splits: {train_df.shape} (train), {val_df.shape} (val), {test_df.shape} (test)")

if __name__ == "__main__":
    main()