import os
import json
import numpy as np
import cv2

IMG_DIR = r"segmentation_dataset/images"
MASK_DIR = "segmentation_dataset/masks"
LOG_DIR = "segmentation_dataset/logs"
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def polygon_to_mask(poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

missing_json = []

for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    base = os.path.splitext(fname)[0]
    img_path = os.path.join(IMG_DIR, fname)
    json_path = os.path.join(IMG_DIR, base + ".json")

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    if not os.path.exists(json_path):
        missing_json.append(fname)
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    full_mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data.get("shapes", []):
        if shape.get("label", "") != "object":
            continue
        points = shape.get("points", [])
        m = polygon_to_mask(points, h, w)
        full_mask = cv2.bitwise_or(full_mask, m)

    out_path = os.path.join(MASK_DIR, base + ".png")
    cv2.imwrite(out_path, full_mask)

log_path = os.path.join(LOG_DIR, "missing_json.txt")
with open(log_path, "w", encoding="utf-8") as f:
    for x in missing_json:
        f.write(x + "\n")

print("Da tao mask vao:", MASK_DIR)
print("So anh thieu json:", len(missing_json))
print("Danh sach:", log_path)