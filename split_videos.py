# split_videos.py
import random, shutil
from pathlib import Path

# 1) Source folder (your extracted Kaggle data)
SRC = Path("raw_pushup_videos_all")

# 2) Destination root for splits
DST = Path("raw_pushup_videos")

# 3) Map the Kaggle subfolder names → our labels
LABELS = {"correct": "good", "incorrect": "bad"}

# 4) Fractions for train/val/test
FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}

def main():
    # a) Make the dst directories
    for split in FRACTIONS:
        for lab in LABELS.values():
            (DST/split/lab).mkdir(parents=True, exist_ok=True)

    # b) Shuffle & move each class
    for src_lab, dst_lab in LABELS.items():
        vids = list((SRC/src_lab).glob("*.mp4"))
        random.shuffle(vids)
        n = len(vids)
        i1 = int(n * FRACTIONS["train"])
        i2 = i1 + int(n * FRACTIONS["val"])

        chunks = [
            (vids[:i1], "train"),
            (vids[i1:i2], "val"),
            (vids[i2:], "test"),
        ]
        for subset, split in chunks:
            for v in subset:
                target = DST/split/dst_lab/v.name
                shutil.move(v, target)
                print(f"Moved {v.name} → {split}/{dst_lab}")

if __name__ == "__main__":
    main()
