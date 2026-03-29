import tensorflow as tf

MODEL_PATH = "/run/media/bao/Windows/Users/nguye/Desktop/workspace/EmbeddedSystems-Segmentation/lab1/checkpoints/unet_best.keras"

def main():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.summary()
    print("Params:", model.count_params())
    print("Input:", model.inputs[0].shape)
    print("Output:", model.outputs[0].shape)

if __name__ == "__main__":
    main()