# 0001 – Dùng imageHeight/imageWidth từ JSON thay vì từ ảnh

## Trạng thái: Đã chấp nhận

## Bối cảnh
Script `01_labelme_json_to_mask.py` ban đầu lấy `h, w` từ `cv2.imread(img_path).shape`.
Sau khi ảnh bị resize về 256×256, tọa độ polygon trong JSON vẫn là tọa độ gốc (ví dụ x=945 trong ảnh 946px).
Vẽ polygon tại tọa độ vượt ngoài mask 256×256 → mask toàn đen.

## Quyết định
Lấy `h = data["imageHeight"]`, `w = data["imageWidth"]` từ JSON.
Vẽ mask với kích thước gốc, sau đó resize về 256×256 bằng `INTER_NEAREST`.

## Hậu quả
- Mask chính xác dù ảnh đã bị resize trước
- Phải chạy lại toàn bộ script convert sau khi áp dụng fix
