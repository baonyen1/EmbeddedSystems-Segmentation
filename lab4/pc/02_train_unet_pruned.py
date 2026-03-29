import os
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = (256, 256)
BATCH = 4
EPOCHS = 100
LR = 1e-4
SEED = 42

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset_split")
MODEL_DIR = os.path.join(ROOT_DIR, "rep_images", "models")

IMG_DIR_TR = os.path.join(DATASET_DIR, "images", "train")
MSK_DIR_TR = os.path.join(DATASET_DIR, "masks", "train")
IMG_DIR_VA = os.path.join(DATASET_DIR, "images", "val")
MSK_DIR_VA = os.path.join(DATASET_DIR, "masks", "val")

def load_pairs(img_dir, msk_dir):
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not os.path.isdir(msk_dir):
        raise FileNotFoundError(f"Mask dir not found: {msk_dir}")

    imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".png",".jpeg"))])
    pairs = []
    for f in imgs:
        base = os.path.splitext(f)[0]
        # giả sử mask cùng base tên (base.png). Nếu lab gốc dùng quy ước khác, chỉnh lại.
        m = base + ".png"
        ip = os.path.join(img_dir, f)
        mp = os.path.join(msk_dir, m)
        if os.path.exists(mp):
            pairs.append((ip, mp))
    return pairs

def read_img_mask(ip, mp):
    img = tf.io.read_file(ip)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0

    msk = tf.io.read_file(mp)
    msk = tf.image.decode_image(msk, channels=1, expand_animations=False)
    msk = tf.image.resize(msk, IMG_SIZE, method="nearest")
    # mask: 0 hoặc 255 -> chuẩn hóa 0/1
    msk = tf.cast(msk > 127, tf.float32)
    return img, msk

def make_ds(pairs, shuffle=True):
    if not pairs:
        raise ValueError("No valid image/mask pairs found.")

    ips = [p[0] for p in pairs]
    mps = [p[1] for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((ips, mps))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(read_img_mask, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

def conv_block(x, f):
    x = keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    return x

def build_unet(base_filters=32):
    inp = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # Encoder
    c1 = conv_block(inp, base_filters)
    p1 = keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, base_filters*2)
    p2 = keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, base_filters*4)
    p3 = keras.layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, base_filters*8)
    p4 = keras.layers.MaxPooling2D()(c4)

    # Bottleneck
    bn = conv_block(p4, base_filters*16)

    # Decoder
    u4 = keras.layers.UpSampling2D()(bn)
    u4 = keras.layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, base_filters*8)

    u3 = keras.layers.UpSampling2D()(c5)
    u3 = keras.layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, base_filters*4)

    u2 = keras.layers.UpSampling2D()(c6)
    u2 = keras.layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, base_filters*2)

    u1 = keras.layers.UpSampling2D()(c7)
    u1 = keras.layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, base_filters)

    out = keras.layers.Conv2D(1, 1, activation="sigmoid")(c8)
    return keras.Model(inp, out, name=f"unet_slim_{base_filters}")

def dice_coef(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    return (2.0 * inter + eps) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + eps)

def dice_coef_hard(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    return (2.0 * inter + eps) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + eps)

def build_callbacks(best_model_path):
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_dice_coef",
        save_best_only=True,
        mode="max"
    )
    es = keras.callbacks.EarlyStopping(
        monitor="val_dice_coef",
        patience=15,
        restore_best_weights=True,
        mode="max"
    )
    rlr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
    return [ckpt, es, rlr]

def main():
    tf.keras.utils.set_random_seed(SEED)

    tr_pairs = load_pairs(IMG_DIR_TR, MSK_DIR_TR)
    va_pairs = load_pairs(IMG_DIR_VA, MSK_DIR_VA)
    tr_ds = make_ds(tr_pairs, shuffle=True)
    va_ds = make_ds(va_pairs, shuffle=False)

    print(f"Train pairs: {len(tr_pairs)}")
    print(f"Val pairs: {len(va_pairs)}")

    runs = [
        (24, os.path.join(MODEL_DIR, "unet_pruned_075x.keras")),
        (16, os.path.join(MODEL_DIR, "unet_pruned_050x.keras")),
    ]
    for base_filters, out_path in runs:
        model = build_unet(base_filters=base_filters)
        model.compile(
            optimizer=keras.optimizers.Adam(LR),
            loss="binary_crossentropy",
            metrics=[dice_coef, dice_coef_hard]
        )

        callbacks = build_callbacks(out_path)

        model.summary()
        model.fit(tr_ds, validation_data=va_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
        print("Best model saved:", out_path)

if __name__ == "__main__":
    main()