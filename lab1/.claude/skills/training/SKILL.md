# SKILL: Training UNet Segmentation

## Vai trò khi kích hoạt
Senior ML engineer chuyên TensorFlow, hiểu sâu về segmentation loss, overfitting, và data pipeline.

## Kích hoạt khi
"train", "huấn luyện", "epoch", "val_loss", "iou", "overfit", "loss không giảm"

## Checklist trước khi train
- [ ] Mask không bị đen (object ratio > 1%)
- [ ] Số ảnh train/val đủ (train ≥ 80, val ≥ 20)
- [ ] IMG_SIZE đồng nhất giữa load_pair và unet()
- [ ] mask dùng `tf.cast(..., tf.float32)` trước khi chia 255

## Đọc log training

| Dấu hiệu | Ý nghĩa | Giải pháp |
|----------|---------|-----------|
| val_loss tăng từ epoch sớm | Overfit | Tăng Dropout, L2, augmentation |
| val_iou = 2e-11 cố định | Model đoán toàn 0 | Kiểm tra mask, dùng soft IoU |
| val_accuracy đứng im | Val set toàn background | Mask val bị lỗi |
| val_iou nhảy loạn | Val set quá nhỏ | Tăng VAL_RATIO |

## Ngưỡng val_iou tham chiếu
| Dataset | Kỳ vọng |
|---------|---------|
| < 100 ảnh | 0.2 – 0.35 |
| 100–200 ảnh | 0.3 – 0.5 |
| > 200 ảnh | 0.5 – 0.7 |

## Cấu hình khuyến nghị theo số ảnh
```python
# 100–150 ảnh
BATCH_SIZE  = 8
EPOCHS      = 50
VAL_RATIO   = 0.25
Dropout     = 0.3 (bottleneck), 0.1 (encoder)
L2          = 1e-4
```
