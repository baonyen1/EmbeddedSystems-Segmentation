import os
import cv2
import numpy as np
import tensorflow as tf

KERAS_MODEL = "C:\\Users\\nguye\\Desktop\\workspace\\EmbeddedSystems-Segmentation\\lab4\\rep_images\\models\\unet_pruned_075x.keras"  # đổi sang 075x nếu cần
REP_DIR = "C:\\Users\\nguye\\Desktop\\workspace\\EmbeddedSystems-Segmentation\\lab4\\rep_images"
REP_SAMPLES = 500
IMG_SIZE = (256, 256)
OUT_TFLITE = "C:\\Users\\nguye\\Desktop\\workspace\\EmbeddedSystems-Segmentation\\lab4\\models_out\\unet_pruned_075x_int8.tflite"
os.makedirs("C:\\Users\\nguye\\Desktop\\workspace\\EmbeddedSystems-Segmentation\\lab4\\models_out", exist_ok=True)

def rep_data_gen():
    paths = [os.path.join(REP_DIR, f) for f in os.listdir(REP_DIR) if f.lower().endswith((".jpg",".png",".jpeg"))]
    paths = paths[:REP_SAMPLES]
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
        x = img.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        yield [x]

model = tf.keras.models.load_model(KERAS_MODEL, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

try:
    tflite_int8 = converter.convert()
except Exception as e:
    print("Full integer output failed -> fallback mixed output float32. Error:", e)
    converter.inference_output_type = tf.float32
    tflite_int8 = converter.convert()

with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_int8)

print("Saved:", OUT_TFLITE, "Size(bytes):", len(tflite_int8))

itp = tf.lite.Interpreter(model_path=OUT_TFLITE)
itp.allocate_tensors()
print("Input:", itp.get_input_details()[0]["dtype"], itp.get_input_details()[0]["quantization"])
print("Output:", itp.get_output_details()[0]["dtype"], itp.get_output_details()[0]["quantization"])
