# SKILL: Chuẩn bị dữ liệu Segmentation

## Vai trò khi kích hoạt
Chuyên gia data pipeline cho Computer Vision, thành thạo OpenCV và Labelme.

## Kích hoạt khi
"mask đen", "json", "labelme", "gán nhãn", "convert mask", "resize", "dataset"

## Quy trình chuẩn

```
1. Chụp/download ảnh → images/
2. Gán nhãn Labelme  → JSON cạnh ảnh (label = "object")
3. Convert JSON→mask → python 01_labelme_json_to_mask.py
4. Kiểm tra mask     → python 04_dataset_stats.py
5. Chia dataset      → python 01_split_dataset.py
```

## Lỗi hay gặp

### Mask toàn đen
**Nguyên nhân:** Resize ảnh trước khi convert → tọa độ polygon lệch
**Fix:** Lấy h, w từ JSON thay vì từ ảnh:
```python
h = data["imageHeight"]
w = data["imageWidth"]
# Vẽ mask với kích thước gốc, sau đó resize về 256x256
mask_resized = cv2.resize(full_mask, (256,256), interpolation=cv2.INTER_NEAREST)
```

### Label không phải "object"
**Kiểm tra:**
```python
with open(json_path) as f:
    data = json.load(f)
print([s["label"] for s in data["shapes"]])
```

## Kiểm tra chất lượng mask
```python
# Mask tốt: object ratio 10%–60%
# Mask lỗi: ratio < 1% hoặc > 95%
ratio = (mask > 0).sum() / mask.size
```
