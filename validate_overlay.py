import cv2, json, os

# 1. Point to your data folder & sample frame
data_dir = "landmark_data"
idx = 750
json_path = os.path.join(data_dir, f"frame_{idx:05d}.json")
img_path  = os.path.join(data_dir, f"frame_{idx:05d}.jpg")

# 2. Load JSON
with open(json_path, "r") as f:
    sample = json.load(f)

print("Frame:", sample["frame_idx"], "| Num landmarks:", len(sample["landmarks"]))

# 3. Load image & get size
img = cv2.imread(img_path)
h, w = img.shape[:2]

# 4. Overlay landmarks
for lm in sample["landmarks"]:
    x_px = int(lm["x"] * w)
    y_px = int(lm["y"] * h)
    cv2.circle(img, (x_px, y_px), 4, (0,255,0), -1)

# 5. Show result
cv2.imshow("Overlay Check", img)
cv2.waitKey(0)
cv2.destroyAllWindows()