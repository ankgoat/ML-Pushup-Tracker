# make_labels_big.py
import csv
from pathlib import Path

# Directory containing all frame JSONs with structure: landmark_data_big/train_good_clipX/frame_0000.json
BIG = Path("landmark_data_big")
out = BIG / "labels.csv"

with open(out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_path", "label"])

    # Iterate all JSONs
    for json_file in BIG.rglob("*.json"):
        # e.g. rel.parts = ("train_good_push up 177", "frame_0000.json")
        rel = json_file.relative_to(BIG)
        folder_name = rel.parts[0]  # first-level folder
        # Split folder_name by '_' into [split, label, rest]
        parts = folder_name.split("_", 2)
        if len(parts) < 2:
            continue
        split, label = parts[0], parts[1]
        # Write full path and label ("good"/"bad")
        writer.writerow([str(json_file), label])

print(f"Wrote labels to {out}")
