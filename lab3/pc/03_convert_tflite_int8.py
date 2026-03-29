import os
import cv2
import numpy as np
import tensorflow as tf

# --- CẤU HÌNH ---
KERAS_MODEL = r"C:\Users\nguye\Desktop\workspace\EmbeddedSystems-Segmentation\lab1\checkpoints\unet_best.keras"
REP_DIR     = r"C:\Users\nguye\Desktop\workspace\EmbeddedSystems-Segmentation\lab3\rep_images"
REP_SAMPLES = 100 # Để 100 là đủ đại diện rồi, 500 hơi lâu
IMG_SIZE    = (256, 256)
OUT_TFLITE  = r"C:\Users\nguye\Desktop\workspace\EmbeddedSystems-Segmentation\lab3\models\unet_int8.tflite"

def rep_data_gen():
    """Hàm tạo dữ liệu đại diện để tính toán range quantization (scale/zero_point)"""
    paths = [os.path.join(REP_DIR, f) for f in os.listdir(REP_DIR)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    # Lấy mẫu ngẫu nhiên hoặc subset
    paths = paths[:REP_SAMPLES]
    
    for p in paths:
        img = cv2.imread(p)
        if img is None: continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
        
        # Chuẩn hóa y hệt lúc train
        x = img.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        yield [x]

def main():
    print("--- Đang load model gốc ---")
    # Load thẳng model (Giả định model đã là float32)
    model = tf.keras.models.load_model(KERAS_MODEL, compile=False)

    print("--- Bắt đầu Convert sang TFLite INT8 ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Thiết lập tối ưu hóa định dạng tĩnh (Integer Quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    
    # Ép toàn bộ các node phải là INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Ép kiểu dữ liệu đầu vào/đầu ra là uint8 (tiết kiệm bandwidth cho vi xử lý)
    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8

    try:
        tflite_int8 = converter.convert()
        
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(OUT_TFLITE), exist_ok=True)
        
        with open(OUT_TFLITE, "wb") as f:
            f.write(tflite_int8)
        print(f"✅ Đã lưu model tại: {OUT_TFLITE}")

    except Exception as e:
        print(f"❌ Lỗi Convert: {e}")
        return

    # --- KIỂM TRA MODEL SAU KHI CONVERT ---
    print("\n--- Thông số Model Quantized ---")
    itp = tf.lite.Interpreter(model_path=OUT_TFLITE)
    itp.allocate_tensors()
    
    input_details = itp.get_input_details()[0]
    output_details = itp.get_output_details()[0]
    
    print(f"Input  - Dtype: {input_details['dtype']}, Quantization: {input_details['quantization']}")
    print(f"Output - Dtype: {output_details['dtype']}, Quantization: {output_details['quantization']}")

if __name__ == "__main__":
    main()