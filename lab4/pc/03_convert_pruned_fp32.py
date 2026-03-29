import os
import tensorflow as tf

KERAS_MODEL = "C:\\Users\\nguye\\Desktop\\workspace\\EmbeddedSystems-Segmentation\\lab4\\rep_images\\models\\unet_pruned_075x.keras"  # đổi sang 075x nếu cần
OUT_TFLITE  = "C:\\Users\\nguye\\Desktop\\workspace\\EmbeddedSystems-Segmentation\\lab4\\models_out\\unet_pruned_075x_fp32.tflite"
os.makedirs("C:\\Users\\nguye\\Desktop\\workspace\\EmbeddedSystems-Segmentation\\lab4\\models_out", exist_ok=True)

model = tf.keras.models.load_model(KERAS_MODEL, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_model)

print("Saved:", OUT_TFLITE, "Size(bytes):", len(tflite_model))