# SKILL: Debug TensorFlow Segmentation

## Vai trò khi kích hoạt
Senior TensorFlow engineer, chuyên debug data pipeline và training issues.

## Kích hoạt khi
"lỗi", "error", "traceback", "không chạy được", "crash"

## Lỗi hay gặp

### TypeError: x and y must have the same dtype
```python
# Fix: cast trước khi chia
mask = tf.cast(mask, tf.float32) / 255.0
```

### ValueError: size must be a 1-D Tensor of 2 elements
```python
# Fix: truyền list thay vì số nguyên
tf.image.resize(img, [IMG_SIZE, IMG_SIZE])  # ✅
tf.image.resize(img, IMG_SIZE)              # ❌
```

### unet() nhận tuple thay vì số
```python
IMG_SIZE = 256          # ✅ số nguyên
IMG_SIZE = (256, 256)   # ❌ nếu dùng trong unet(input_size=(IMG_SIZE, IMG_SIZE, 3))
```

### Hàm chết sau return
```python
def make_datasets():
    return ds  # ← code sau đây không bao giờ chạy
    def build_ds():  # ❌ dead code
        ...
```

### shutil.copy2 lỗi
Nguyên nhân: tên ảnh và mask không khớp
```python
# Kiểm tra trước khi copy
if not os.path.exists(mask_path):
    print(f"Thiếu mask: {base}.png")
```
