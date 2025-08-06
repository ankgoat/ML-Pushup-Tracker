import os, json, cv2, csv

DATA_DIR   = "landmark_data"
LABEL_FILE = "labels.csv"

# 1. Gather the JSON files you want to label
files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json"))

# 2. Open CSV for writing/appending
writer = csv.writer(open(LABEL_FILE, "a", newline=""))
if os.stat(LABEL_FILE).st_size == 0:
    writer.writerow(["filename","label"])  # header

# 3. Loop through each file
for fn in files:
    base = fn[:-5]
    img = cv2.imread(os.path.join(DATA_DIR, base+".jpg"))
    data = json.load(open(os.path.join(DATA_DIR, fn)))
    h, w = img.shape[:2]

    # Overlay landmarks
    for lm in data["landmarks"]:
        x, y = int(lm["x"]*w), int(lm["y"]*h)
        cv2.circle(img, (x,y), 4, (0,255,0), -1)

    cv2.imshow("Label Frame", img)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    # 4. Read user input
    if key == ord('g'):
        label = "good"
    elif key == ord('b'):
        label = "bad"
    else:
        print("Skipping", fn)
        continue

    # 5. Write label
    writer.writerow([base, label])
    print(f"Labeled {base} as {label}\nPress any key for next…")