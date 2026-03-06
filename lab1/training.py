import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# =====================
# CONFIG
# =====================
IMG_SIZE    = 256
BATCH_SIZE  = 8
EPOCHS      = 50
VAL_SPLIT   = 0.2   # 20% dữ liệu dùng để validation
SEED        = 42

IMG_DIR  = r"C:\Users\nguye\Documents\3RD YEAR\TT_nhung\lab1\segmentation_dataset\images"
MASK_DIR = r"C:\Users\nguye\Documents\3RD YEAR\TT_nhung\lab1\segmentation_dataset\masks"


# =====================
# LOSS & METRICS
# =====================
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss tính per-sample rồi lấy mean."""
    y_true = tf.cast(tf.reshape(y_true, [tf.shape(y_true)[0], -1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]), tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=1)
    union        = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(1.0 - dice)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """IoU (Intersection over Union) — phù hợp hơn accuracy cho segmentation."""
    y_true = tf.cast(y_true > threshold, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union        = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


# =====================
# DATA LOADING
# =====================
def load_pair(img_path, mask_path):
    # Dùng decode_jpeg/decode_png thay vì decode_image để đảm bảo shape rõ ràng
    img_raw = tf.io.read_file(img_path)
    img = tf.cond(
        tf.strings.regex_full_match(img_path, r".*\.(jpg|jpeg|JPG|JPEG)"),
        lambda: tf.image.decode_jpeg(img_raw, channels=3),
        lambda: tf.image.decode_png(img_raw, channels=3)
    )
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    mask_raw = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_raw, channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.0
    # Đảm bảo mask là nhị phân 0/1
    mask = tf.round(mask)

    return img, mask

def make_datasets():
    image_paths, mask_paths = [], []

    for f in sorted(os.listdir(IMG_DIR)):
        base, ext = os.path.splitext(f)
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        mask_path = os.path.join(MASK_DIR, base + ".png")
        if os.path.exists(mask_path):
            image_paths.append(os.path.join(IMG_DIR, f))
            mask_paths.append(mask_path)

    n = len(image_paths)
    print(f"Tổng số cặp img-mask hợp lệ: {n}")
    if n == 0:
        raise ValueError("Không tìm thấy dữ liệu! Kiểm tra lại IMG_DIR và MASK_DIR.")

    # Shuffle cố định rồi chia train/val
    import random
    random.seed(SEED)
    indices = list(range(n))
    random.shuffle(indices)

    n_val   = max(1, int(n * VAL_SPLIT))
    val_idx = indices[:n_val]
    trn_idx = indices[n_val:]

    def build_ds(idx_list, shuffle=False):
        imgs  = [image_paths[i] for i in idx_list]
        masks = [mask_paths[i]  for i in idx_list]
        ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
        ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(len(idx_list), seed=SEED)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = build_ds(trn_idx, shuffle=True)
    val_ds   = build_ds(val_idx, shuffle=False)

    print(f"  Train: {len(trn_idx)} ảnh | Val: {len(val_idx)} ảnh")
    return train_ds, val_ds


# =====================
# U-NET MODEL
# =====================
def conv_block(x, filters, dropout_rate=0.0):
    """2x Conv + BatchNorm + optional Dropout."""
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x

def unet(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck (dropout giúp chống overfit với dataset nhỏ)
    b = conv_block(p2, 128, dropout_rate=0.3)

    # Decoder
    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = conv_block(u1, 64)

    u2 = layers.UpSampling2D()(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = conv_block(u2, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c4)

    return models.Model(inputs, outputs, name="UNet")


# =====================
# CALLBACKS
# =====================
def get_callbacks():
    os.makedirs("checkpoints", exist_ok=True)
    return [
        # Lưu model có val_loss thấp nhất
        callbacks.ModelCheckpoint(
            filepath="checkpoints/unet_best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # Dừng sớm nếu val_loss không cải thiện sau 10 epoch
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Giảm learning rate khi val_loss plateau
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard log (chạy: tensorboard --logdir logs/)
        callbacks.TensorBoard(log_dir="logs", histogram_freq=1),
    ]


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    train_ds, val_ds = make_datasets()

    model = unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=bce_dice_loss,
        metrics=["accuracy", iou_metric]
    )
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks()
    )

    model.save("unet_person_segmentation.keras")
    print("✅ Model đã lưu: unet_person_segmentation.keras")
    print("✅ Model tốt nhất: checkpoints/unet_best.keras")