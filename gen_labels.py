import os, json
import csv

json_dir   = "landmark_data/json"
out_csv    = "landmark_data/labels.csv"

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file","label"])
    for fname in sorted(os.listdir(json_dir)):
        path = os.path.join(json_dir, fname)
        data = json.load(open(path))
        # data['label'] is 1 for Correct, 0 for Wrong
        writer.writerow([fname, data["label"]])