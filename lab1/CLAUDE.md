# CLAUDE.md — Dự án Face/Person Segmentation

## Vai trò
Bạn là senior ML engineer chuyên về Computer Vision và TensorFlow.
Hỗ trợ sinh viên xây dựng pipeline segmentation từ thu thập dữ liệu đến train model.

---

## Stack công nghệ
- Python 3.9+
- TensorFlow / Keras
- OpenCV (cv2)
- NumPy, Matplotlib
- Labelme (gán nhãn)

---

## Cấu trúc thư mục dự án

```
lab1/
├── segmentation_dataset/
│   ├── images/        # ảnh gốc (.jpg/.png)
│   ├── masks/         # mask nhị phân (.png, 0/255)
│   ├── overlays/      # ảnh kiểm tra overlay
│   └── logs/
├── dataset_split/
│   ├── images/train|val|test/
│   └── masks/train|val|test/
├── checkpoints/
│   └── unet_best.keras
├── scripts/
│   ├── 01_labelme_json_to_mask.py
│   ├── 02_resize_images_and_masks.py
│   ├── 03_make_overlays.py
│   ├── 04_dataset_stats.py
│   └── 01_split_dataset.py
├── training.py
└── unet_person_segmentation.keras
```

---

## Quy ước dữ liệu
- Tên file: `img_001.jpg` ↔ `img_001.png`
- Mask: background = 0, object = 255 (nhị phân PNG)
- Kích thước chuẩn: 256×256
- Resize mask dùng `INTER_NEAREST`
- Tọa độ polygon lấy từ JSON Labelme (imageWidth/imageHeight), KHÔNG từ ảnh đã resize

---

## Config training hiện tại
```python
DATA_DIR   = "dataset_split"
IMG_SIZE   = 256
BATCH_SIZE = 8
EPOCHS     = 50
SEED       = 42
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.25
TEST_RATIO  = 0.15
```

---

## Model: UNet (from scratch)
- Encoder: Conv(32) → Pool → Conv(64) → Pool
- Bottleneck: Conv(128) + Dropout(0.3) + L2(1e-4)
- Decoder: UpSample + Skip connection × 2
- Output: Conv(1, sigmoid)
- Loss: BCE + Dice
- Metric: Soft IoU

---

## Kết quả hiện tại
| Dataset | Ảnh | Best val_iou | Best epoch |
|---------|-----|-------------|------------|
| 49 ảnh  | -   | ~0.00 (mask lỗi) | - |
| 129 ảnh | ✅  | ~0.289 | 3 |

---

## Vấn đề đã gặp & cách fix
1. **Mask toàn đen** → tọa độ polygon lệch do resize ảnh trước khi convert JSON → dùng `imageHeight/imageWidth` từ JSON
2. **val_iou = 2e-11** → iou_metric dùng threshold cứng 0.5 → đổi sang soft IoU
3. **Overfit sớm (epoch 3)** → dataset nhỏ, thêm Dropout + L2 + augmentation
4. **`mask / 255.0` lỗi dtype** → thêm `tf.cast(mask, tf.float32)` trước

---

## Lệnh hay dùng
```bash
# Gán nhãn
labelme

# Convert mask
python scripts/01_labelme_json_to_mask.py

# Chia dataset
python scripts/01_split_dataset.py

# Train
python training.py

# Kiểm tra mask lỗi
python scripts/04_dataset_stats.py
```
