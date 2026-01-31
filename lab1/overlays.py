import os
import cv2

IMG_DIR = "segmentation_dataset/images"
MASK_DIR = "segmentation_dataset/masks"
OUT_DIR = "segmentation_dataset/overlays"
os.makedirs(OUT_DIR, exist_ok=True)

for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    base = os.path.splitext(img_name)[0]
    img_path = os.path.join(IMG_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, base + ".png")

    img = cv2.imread(img_path)
    if img is None or not os.path.exists(mask_path):
        continue

    mask = cv2.imread(mask_path, 0)
    color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    cv2.imwrite(os.path.join(OUT_DIR, base + ".jpg"), overlay)

print("Da tao overlays:", OUT_DIR)
