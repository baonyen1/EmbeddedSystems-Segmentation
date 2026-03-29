import tensorflow as tf

KERAS_MODEL = r"C:\Users\nguye\Desktop\workspace\EmbeddedSystems-Segmentation\lab1\checkpoints\unet_best.keras"
OUT_TFLITE  = r"C:\Users\nguye\Desktop\workspace\EmbeddedSystems-Segmentation\lab3\models\unet_fp32.tflite"

# Load without compiling
model = tf.keras.models.load_model(KERAS_MODEL, compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable TF Select ops to support Conv2D, Relu, MaxPool, etc.
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_model)

print("Saved:", OUT_TFLITE)