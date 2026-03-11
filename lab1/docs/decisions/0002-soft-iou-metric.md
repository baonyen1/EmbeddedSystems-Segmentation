# 0002 – Dùng Soft IoU thay vì Hard IoU (threshold=0.5)

## Trạng thái: Đã chấp nhận

## Bối cảnh
Hard IoU dùng `tf.cast(y_pred > 0.5, tf.float32)` → những epoch đầu model output
giá trị thấp (0.1~0.3), chưa vượt ngưỡng 0.5 → IoU luôn = 0 (hoặc 2e-11).
Không theo dõi được learning progress thực sự.

## Quyết định
Dùng Soft IoU: tính trực tiếp trên giá trị sigmoid, không threshold.
```python
intersection = tf.reduce_sum(y_true * y_pred, axis=1)
union = tf.reduce_sum(y_true + y_pred - y_true * y_pred, axis=1)
iou = (intersection + smooth) / (union + smooth)
```

## Hậu quả
- Thấy được progress từ epoch 1
- Giá trị Soft IoU ≠ Hard IoU (thường cao hơn một chút)
- Khi báo cáo kết quả cuối cần dùng Hard IoU (threshold 0.5)
