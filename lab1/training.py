import os
import random
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# =====================
# CONFIG
# =====================
DATA_DIR = "dataset_split"
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50
VAL_SPLIT   = 0.15
SEED        = 42

# Transfer learning: dùng MobileNetV2 pretrained
# NOTE: Tam tat transfer learning vi shape mismatch - dung Deeper U-Net
USE_TRANSFER_LEARNING = False
TRANSFER_LEARNING_MODEL = "mobilenetv2"  # hoặc "efficientnetb0"

# Dice loss weight - tăng weight cho dice loss
DICE_WEIGHT = 0.7
BCE_WEIGHT = 0.3

# Test-Time Augmentation
USE_TTA = True
TTA_SHIFTS = [(0, 0), (0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]
TTA_FLIPS = [False, True]

# Augmentation config (cho training)
AUG_ROTATION_RANGE = 25.0
AUG_ZOOM_RANGE = 0.2
AUG_SHEAR_RANGE = 0.15

print(tf.config.list_physical_devices('GPU'))

# =====================
# DATA LOADING
# =====================
def load_pair(img_path, mask_path):
    # Ảnh
    img_raw = tf.io.read_file(img_path)
    img = tf.cond(
        tf.strings.regex_full_match(img_path, r".*\.(jpg|jpeg|JPG|JPEG)"),
        lambda: tf.image.decode_jpeg(img_raw, channels=3),
        lambda: tf.image.decode_png(img_raw,  channels=3)
    )
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    # Mask
    mask_raw = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_raw, channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.round(mask)  # đảm bảo nhị phân 0/1

    return img, mask

def make_datasets():
    def build_ds(split, shuffle=False, augment_data=False):
        img_dir  = os.path.join(DATA_DIR, "images", split)
        mask_dir = os.path.join(DATA_DIR, "masks",  split)

        imgs = sorted([f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        img_paths  = [os.path.join(img_dir,  f) for f in imgs]
        mask_paths = [os.path.join(mask_dir, os.path.splitext(f)[0] + ".png") for f in imgs]

        ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
        ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        if augment_data:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(len(imgs), seed=SEED)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), len(imgs)

    train_ds, n_train = build_ds("train", shuffle=True,  augment_data=True)
    val_ds,   n_val   = build_ds("val",   shuffle=False, augment_data=False)

    print(f"Train: {n_train} images | Val: {n_val} images")
    return train_ds, val_ds
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
    """Weighted BCE + Dice loss."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return BCE_WEIGHT * bce + DICE_WEIGHT * dice

def iou_metric(y_true, y_pred, smooth=1e-6):
    """Soft IoU — không dùng threshold cứng, thấy được learning progress."""
    y_true = tf.cast(tf.reshape(y_true, [tf.shape(y_true)[0], -1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]), tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=1)
    union        = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))


# =====================
# AUGMENTATION
# =====================
def augment(img, mask):
    """Augmentation mạnh cho dataset nhỏ: rotation, zoom, flip + brightness/contrast."""
    # === Rotation (dùng tf.image.rot90) ===
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img  = tf.image.rot90(img, k=k)
    mask = tf.image.rot90(mask, k=k)

    # === Zoom/Scale ===
    if tf.random.uniform(()) > 0.5:
        zoom_x = tf.random.uniform((), 1 - AUG_ZOOM_RANGE, 1 + AUG_ZOOM_RANGE)
        zoom_y = tf.random.uniform((), 1 - AUG_ZOOM_RANGE, 1 + AUG_ZOOM_RANGE)
        crop_h = tf.cast(tf.cast(IMG_SIZE, tf.float32) / zoom_y, tf.int32)
        crop_w = tf.cast(tf.cast(IMG_SIZE, tf.float32) / zoom_x, tf.int32)
        img = tf.image.resize_with_crop_or_pad(img, crop_h, crop_w)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method='bilinear')
        mask = tf.image.resize_with_crop_or_pad(mask, crop_h, crop_w)
        mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method='nearest')

    # === Flip ngang ===
    if tf.random.uniform(()) < 0.5:
        img  = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    # === Flip dọc ===
    if tf.random.uniform(()) < 0.3:
        img  = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    # === Brightness / contrast (chỉ áp dụng cho ảnh, KHÔNG cho mask) ===
    img = tf.image.random_brightness(img, 0.3)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    img = tf.clip_by_value(img, 0.0, 1.0)

    # === Thêm Gaussian noise (chỉ cho ảnh) ===
    if tf.random.uniform(()) > 0.7:
        noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.05)
        img = tf.clip_by_value(img + noise, 0.0, 1.0)

    return img, mask


# =====================
# U-NET MODEL (Deeper + Transfer Learning)
# =====================
def conv_block(x, filters, dropout_rate=0.0):
    """2x Conv + BatchNorm + L2 + optional Dropout."""
    from tensorflow.keras import regularizers
    reg = regularizers.l2(1e-5) if USE_TRANSFER_LEARNING else regularizers.l2(1e-4)

    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    return x

def unet(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    """Deeper U-Net với 256, 512 filters ở bottleneck."""
    if USE_TRANSFER_LEARNING:
        # Transfer learning với MobileNetV2 pretrained
        if TRANSFER_LEARNING_MODEL == "mobilenetv2":
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=input_size,
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False  # Freeze ban đầu, unfreeze sau
            inputs = layers.Input(input_size)
            # Lấy features ở các layer khác nhau cho skip connections
            c1 = base_model.get_layer('block_1_expand_relu').output  # 32 filters
            c2 = base_model.get_layer('block_3_expand_relu').output  # 64 filters
            b = base_model.get_layer('block_13_expand_relu').output  # 128 filters
            # Decoder
            u1 = layers.UpSampling2D()(b)
            u1 = conv_block(u1, 64, dropout_rate=0.2)
            u1 = layers.Concatenate()([u1, c2])
            u1 = conv_block(u1, 64, dropout_rate=0.2)
            u2 = layers.UpSampling2D()(u1)
            u2 = conv_block(u2, 32, dropout_rate=0.1)
            u2 = layers.Concatenate()([u2, c1])
            u2 = conv_block(u2, 32, dropout_rate=0.1)
            outputs = layers.Conv2D(1, 1, activation="sigmoid")(u2)
            return models.Model(inputs, outputs, name="UNet_MobileNetV2")

        elif TRANSFER_LEARNING_MODEL == "efficientnetb0":
            base_model = tf.keras.applications.EfficientNetB0(
                input_shape=input_size,
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            inputs = layers.Input(input_size)
            x = base_model(inputs, training=False)
            # Lấy features cho decoder
            block1 = base_model.get_layer('block2a_expand_activation').output
            block2 = base_model.get_layer('block3a_expand_activation').output
            bottleneck = base_model.get_layer('top_activation').output
            # Decoder
            u1 = layers.UpSampling2D(size=(2, 2))(bottleneck)
            u1 = conv_block(u1, 64, dropout_rate=0.2)
            u1 = layers.Concatenate()([u1, block2])
            u2 = layers.UpSampling2D(size=(2, 2))(u1)
            u2 = conv_block(u2, 32, dropout_rate=0.1)
            u2 = layers.Concatenate()([u2, block1])
            outputs = layers.Conv2D(1, 1, activation="sigmoid")(u2)
            return models.Model(inputs, outputs, name="UNet_EfficientNetB0")

    # U-Net từ đầu (không transfer learning) - deeper version
    inputs = layers.Input(input_size)

    # Encoder - thêm lớp 256 filters
    c1 = conv_block(inputs, 32, dropout_rate=0.05)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64, dropout_rate=0.1)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128, dropout_rate=0.2)
    p3 = layers.MaxPooling2D()(c3)

    # Bottleneck - 256 filters
    b = conv_block(p3, 256, dropout_rate=0.3)

    # Decoder
    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c3])
    c4 = conv_block(u1, 128, dropout_rate=0.2)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, 64, dropout_rate=0.1)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c1])
    c6 = conv_block(u3, 32, dropout_rate=0.05)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c6)

    return models.Model(inputs, outputs, name="UNet_Deep")


# =====================
# CALLBACKS
# =====================
def get_callbacks():
    os.makedirs("checkpoints", exist_ok=True)
    return [
        callbacks.ModelCheckpoint(
            filepath="checkpoints/unet_best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.TensorBoard(log_dir="logs", histogram_freq=1),
    ]


# =====================
# TEST-TIME AUGMENTATION (TTA)
# =====================
def predict_with_tta(model, img, n_shifts=5):
    """
    Test-Time Augmentation: average predictions từ nhiều augmented versions.
    img: single image [IMG_SIZE, IMG_SIZE, 3]
    """
    if not USE_TTA:
        return model.predict(img[None, ...])[0]

    predictions = []
    img = img[0] if len(img.shape) == 4 else img

    # Original prediction
    pred_orig = model.predict(img[None, ...], verbose=0)[0]
    predictions.append(pred_orig)

    # Flipped prediction
    img_flip = tf.image.flip_left_right(img)
    pred_flip = model.predict(img_flip[None, ...], verbose=0)[0]
    pred_flip = tf.image.flip_left_right(pred_flip)
    predictions.append(pred_flip)

    # Small shifts
    for dx, dy in TTA_SHIFTS[1:]:  # skip (0,0)
        img_shifted = tf.roll(img, shift=int(dx*IMG_SIZE), axis=0)
        img_shifted = tf.roll(img_shifted, shift=int(dy*IMG_SIZE), axis=1)
        pred_shifted = model.predict(img_shifted[None, ...], verbose=0)[0]
        pred_shifted = tf.roll(pred_shifted, shift=-int(dx*IMG_SIZE), axis=0)
        pred_shifted = tf.roll(pred_shifted, shift=-int(dy*IMG_SIZE), axis=1)
        predictions.append(pred_shifted)

    # Average all predictions
    return tf.reduce_mean(predictions, axis=0)


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    train_ds, val_ds = make_datasets()

    model = unet()

    # Nếu dùng transfer learning, unfreeze một phần sau vài epochs
    if USE_TRANSFER_LEARNING:
        print("\n=== Transfer Learning Mode ===")
        print(f"Using {TRANSFER_LEARNING_MODEL} pretrained encoder")
        print("Phase 1: Training decoder with frozen encoder (10 epochs)...")

        # Compile với learning rate thấp hơn cho transfer learning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=bce_dice_loss,
            metrics=["accuracy", iou_metric]
        )
        model.summary()

        # Train phase 1: frozen encoder
        history_phase1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=get_callbacks()
        )

        # Unfreeze một số layer cuối của encoder
        print("\nPhase 2: Unfreezing encoder layers...")
        if TRANSFER_LEARNING_MODEL == "mobilenetv2":
            base_model = model.layers[1]  # MobileNetV2 base
        elif TRANSFER_LEARNING_MODEL == "efficientnetb0":
            base_model = model.layers[1]  # EfficientNetB0 base
        else:
            base_model = None

        if base_model:
            for layer in base_model.layers[-40:]:  # Unfreeze 40 layers cuối
                layer.trainable = True

        # Recompile với learning rate rất thấp
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=bce_dice_loss,
            metrics=["accuracy", iou_metric]
        )

        # Train phase 2: fine-tuning
        print("Training with fine-tuning...")
        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS - 10,
            callbacks=get_callbacks()
        )

        history = {
            'history': {k: history_phase1.history.get(k, []) + history_phase2.history.get(k, [])
                       for k in ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'iou_metric', 'val_iou_metric']}
        }
    else:
        print("\n=== Training U-Net from scratch (Deeper version) ===")
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
    print("\n✅ Model đã lưu: unet_person_segmentation.keras")
    print("✅ Model tốt nhất: checkpoints/unet_best.keras")

    # TTA evaluation (nếu muốn test với TTA)
    if USE_TTA:
        print("\n✅ TTA (Test-Time Augmentation) đã được enable")
        print("   Dùng predict_with_tta(model, img) để predict với TTA")