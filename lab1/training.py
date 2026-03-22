import os
import importlib.util
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# =====================
# CONFIG
# =====================
DATA_DIR    = "dataset_split"
IMG_SIZE    = 256
BATCH_SIZE  = 4        # tăng từ 4 → 8: batch lớn hơn giúp BatchNorm ổn định hơn
EPOCHS      = 200      # tăng epochs, EarlyStopping sẽ dừng đúng lúc
SEED        = 42

# Loss weights
DICE_WEIGHT = 0.7
BCE_WEIGHT  = 0.3      # tăng BCE weight một chút để giữ gradient ổn định

# Augmentation — nhẹ hơn trước vì dataset nhỏ (300 ảnh)
AUG_ZOOM_RANGE      = 0.15   # giảm từ 0.2 → 0.15
AUG_ROTATION_RANGE  = 15.0   # giảm từ 25 → 15 độ (người không xoay nhiều)

print("GPUs:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# NOTE: BỎ mixed_float16 — gây numerical instability với Dice loss
# mixed_precision.set_global_policy('mixed_float16')


# =====================
# DATA LOADING
# =====================
def load_pair(img_path, mask_path):
    img_raw = tf.io.read_file(img_path)
    img = tf.cond(
        tf.strings.regex_full_match(img_path, r".*\.(jpg|jpeg|JPG|JPEG)"),
        lambda: tf.image.decode_jpeg(img_raw, channels=3),
        lambda: tf.image.decode_png(img_raw,  channels=3)
    )
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    mask_raw = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_raw, channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.round(mask)
    return img, mask


def augment(img, mask):
    """
    Augmentation nhẹ hơn, phù hợp với segmentation người:
    - Bỏ flip_up_down (người lộn ngược không tự nhiên)
    - Bỏ rot90 (người xoay 90° bất thường)
    - Giảm zoom, rotation range
    - Thêm cutout để tăng robustness
    """
    # Flip ngang (tự nhiên với người)
    if tf.random.uniform(()) < 0.5:
        img  = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    # Rotation nhỏ (±15 độ) — dùng tf.keras.layers thay vì rot90
    angle = tf.random.uniform((), -AUG_ROTATION_RANGE, AUG_ROTATION_RANGE) * (3.14159 / 180.0)
    img  = tfa_rotate(img,  angle)
    mask = tfa_rotate(mask, angle, is_mask=True)

    # Zoom/Crop nhẹ
    if tf.random.uniform(()) > 0.5:
        scale  = tf.random.uniform((), 1.0 - AUG_ZOOM_RANGE, 1.0 + AUG_ZOOM_RANGE)
        new_sz = tf.cast(tf.cast(IMG_SIZE, tf.float32) / scale, tf.int32)
        new_sz = tf.clip_by_value(new_sz, IMG_SIZE // 2, int(IMG_SIZE * 1.5))
        img  = tf.image.resize_with_crop_or_pad(img,  new_sz, new_sz)
        img  = tf.image.resize(img,  [IMG_SIZE, IMG_SIZE], method='bilinear')
        mask = tf.image.resize_with_crop_or_pad(mask, new_sz, new_sz)
        mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method='nearest')

    # Shift (translate)
    if tf.random.uniform(()) > 0.5:
        pad = IMG_SIZE // 8
        img  = tf.pad(img,  [[pad, pad], [pad, pad], [0, 0]])
        mask = tf.pad(mask, [[pad, pad], [pad, pad], [0, 0]])
        offset_h = tf.random.uniform((), 0, 2*pad, dtype=tf.int32)
        offset_w = tf.random.uniform((), 0, 2*pad, dtype=tf.int32)
        img  = tf.image.crop_to_bounding_box(img,  offset_h, offset_w, IMG_SIZE, IMG_SIZE)
        mask = tf.image.crop_to_bounding_box(mask, offset_h, offset_w, IMG_SIZE, IMG_SIZE)

    # Color jitter (chỉ image)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)  # thêm saturation
    img = tf.image.random_hue(img, 0.05)              # nhẹ thôi
    img = tf.clip_by_value(img, 0.0, 1.0)

    # Gaussian noise nhẹ (chỉ image)
    if tf.random.uniform(()) > 0.7:
        noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.03)  # giảm stddev
        img = tf.clip_by_value(img + noise, 0.0, 1.0)

    # Cutout (che một vùng nhỏ) — tăng robustness
    if tf.random.uniform(()) > 0.7:
        cut_size = IMG_SIZE // 8
        cx = tf.random.uniform((), cut_size, IMG_SIZE - cut_size, dtype=tf.int32)
        cy = tf.random.uniform((), cut_size, IMG_SIZE - cut_size, dtype=tf.int32)
        mask_cut = tf.ones([IMG_SIZE, IMG_SIZE, 1], dtype=tf.float32)
        # Tạo vùng 0 tại (cx, cy)
        indices_h = tf.range(IMG_SIZE)
        indices_w = tf.range(IMG_SIZE)
        hh, ww = tf.meshgrid(indices_h, indices_w, indexing='ij')
        in_box = tf.cast(
            (tf.abs(hh - cx) < cut_size // 2) & (tf.abs(ww - cy) < cut_size // 2),
            tf.float32
        )
        in_box = tf.expand_dims(in_box, axis=-1)
        img = img * (1.0 - in_box)

    mask = tf.round(tf.clip_by_value(mask, 0.0, 1.0))
    return img, mask


def tfa_rotate(tensor, angle, is_mask=False):
    """
    Rotate image/mask bằng affine transform thủ công (không cần TFA).
    angle: radian
    """
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    h = w = IMG_SIZE
    cx = tf.cast(w, tf.float32) / 2.0
    cy = tf.cast(h, tf.float32) / 2.0

    xs = tf.cast(tf.range(w), tf.float32) - cx
    ys = tf.cast(tf.range(h), tf.float32) - cy
    xx, yy = tf.meshgrid(xs, ys)
    src_x = cos_a * xx + sin_a * yy + cx
    src_y = -sin_a * xx + cos_a * yy + cy

    src_x = tf.clip_by_value(src_x, 0.0, tf.cast(w - 1, tf.float32))
    src_y = tf.clip_by_value(src_y, 0.0, tf.cast(h - 1, tf.float32))

    src_x0 = tf.cast(tf.floor(src_x), tf.int32)
    src_y0 = tf.cast(tf.floor(src_y), tf.int32)
    src_x1 = tf.minimum(src_x0 + 1, w - 1)
    src_y1 = tf.minimum(src_y0 + 1, h - 1)

    if is_mask:
        # Nearest neighbor cho mask
        src_xr = tf.cast(tf.round(src_x), tf.int32)
        src_yr = tf.cast(tf.round(src_y), tf.int32)
        indices = tf.stack([src_yr, src_xr], axis=-1)
        result = tf.gather_nd(tensor, indices)
        return tf.round(tf.clip_by_value(result, 0.0, 1.0))
    else:
        # Bilinear interpolation cho image
        fx = src_x - tf.cast(src_x0, tf.float32)
        fy = src_y - tf.cast(src_y0, tf.float32)
        fx = tf.expand_dims(fx, -1)
        fy = tf.expand_dims(fy, -1)

        i00 = tf.gather_nd(tensor, tf.stack([src_y0, src_x0], axis=-1))
        i01 = tf.gather_nd(tensor, tf.stack([src_y0, src_x1], axis=-1))
        i10 = tf.gather_nd(tensor, tf.stack([src_y1, src_x0], axis=-1))
        i11 = tf.gather_nd(tensor, tf.stack([src_y1, src_x1], axis=-1))

        result = (i00 * (1 - fx) * (1 - fy) +
                  i01 * fx       * (1 - fy) +
                  i10 * (1 - fx) * fy       +
                  i11 * fx       * fy)
        return result


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
    y_true = tf.cast(tf.reshape(y_true, [tf.shape(y_true)[0], -1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=1)
    union        = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(1.0 - dice)

def bce_dice_loss(y_true, y_pred):
    bce  = tf.keras.losses.binary_crossentropy(
        tf.cast(y_true, tf.float32),
        tf.cast(y_pred, tf.float32)
    )
    dice = dice_loss(y_true, y_pred)
    return BCE_WEIGHT * tf.reduce_mean(bce) + DICE_WEIGHT * dice

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(tf.reshape(y_true, [tf.shape(y_true)[0], -1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=1)
    union = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))

def dice_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(tf.reshape(y_true, [tf.shape(y_true)[0], -1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=1)
    union = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1)
    return tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth))


# =====================
# MODEL: U-Net + Attention Gate + Conv2DTranspose
# =====================
def conv_block(x, filters, dropout_rate=0.0):
    """Double Conv + BN + ReLU + optional SpatialDropout."""
    reg = tf.keras.regularizers.l2(1e-4)
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    return x


def attention_gate(x, g, filters):
    """
    Attention Gate: tập trung vào vùng quan trọng (người).
    x: skip connection từ encoder
    g: gating signal từ decoder
    """
    theta_x = layers.Conv2D(filters, 1, padding="same")(x)
    phi_g   = layers.Conv2D(filters, 1, padding="same")(g)
    add     = layers.Add()([theta_x, phi_g])
    relu    = layers.Activation("relu")(add)
    psi     = layers.Conv2D(1, 1, padding="same")(relu)
    sigmoid = layers.Activation("sigmoid")(psi)
    return layers.Multiply()([x, sigmoid])


def unet_attention(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    """
    U-Net với:
    - Attention Gates ở mỗi skip connection
    - Conv2DTranspose thay UpSampling2D (học được upsampling)
    - Deeper: 32→64→128→256→512
    - Deep supervision (auxiliary loss ở bottleneck)
    """
    inputs = layers.Input(input_size)

    # ---- Encoder ----
    c1 = conv_block(inputs, 32,  dropout_rate=0.05)
    p1 = layers.MaxPooling2D()(c1)                       # 128x128

    c2 = conv_block(p1,  64,  dropout_rate=0.1)
    p2 = layers.MaxPooling2D()(c2)                       # 64x64

    c3 = conv_block(p2,  128, dropout_rate=0.2)
    p3 = layers.MaxPooling2D()(c3)                       # 32x32

    c4 = conv_block(p3,  256, dropout_rate=0.3)
    p4 = layers.MaxPooling2D()(c4)                       # 16x16

    # ---- Bottleneck ----
    b = conv_block(p4, 512, dropout_rate=0.4)            # 16x16

    # ---- Decoder với Attention Gate + Conv2DTranspose ----
    # Block 1: 16→32
    u1 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(b)   # 32x32
    c4_att = attention_gate(c4, u1, 256)
    u1 = layers.Concatenate()([u1, c4_att])
    c5 = conv_block(u1, 256, dropout_rate=0.3)

    # Block 2: 32→64
    u2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c5)  # 64x64
    c3_att = attention_gate(c3, u2, 128)
    u2 = layers.Concatenate()([u2, c3_att])
    c6 = conv_block(u2, 128, dropout_rate=0.2)

    # Block 3: 64→128
    u3 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c6)   # 128x128
    c2_att = attention_gate(c2, u3, 64)
    u3 = layers.Concatenate()([u3, c2_att])
    c7 = conv_block(u3, 64,  dropout_rate=0.1)

    # Block 4: 128→256
    u4 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c7)   # 256x256
    c1_att = attention_gate(c1, u4, 32)
    u4 = layers.Concatenate()([u4, c1_att])
    c8 = conv_block(u4, 32,  dropout_rate=0.05)

    # ---- Output ----
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output")(c8)

    return models.Model(inputs, outputs, name="UNet_Attention")


# =====================
# CALLBACKS
# =====================
def get_callbacks():
    os.makedirs("checkpoints", exist_ok=True)
    return [
        callbacks.ModelCheckpoint(
            filepath="checkpoints/unet_best.keras",
            monitor="val_iou_metric",   # monitor IoU thay vì loss
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_iou_metric",
            mode="max",
            patience=50,                # tăng patience
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_iou_metric",
            mode="max",
            factor=0.5,
            patience=25,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(log_dir="logs", histogram_freq=0),
    ]


# =====================
# WARMUP + COSINE DECAY LR SCHEDULE
# =====================
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup rồi cosine decay — ổn định hơn fixed LR."""
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr      = base_lr
        self.total_steps  = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = step / tf.cast(self.warmup_steps, tf.float32)
        cosine = 0.5 * (1.0 + tf.math.cos(
            3.14159 * (step - self.warmup_steps) /
            tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        ))
        return tf.where(step < self.warmup_steps,
                        warmup * self.base_lr,
                        cosine * self.base_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps
        }


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    train_ds, val_ds = make_datasets()

    model = unet_attention()
    model.summary()

    # Tính steps để dùng LR schedule
    # Ước tính n_train ~ 80% * 300 = 240
    N_TRAIN     = 240
    STEPS_EPOCH = N_TRAIN // BATCH_SIZE
    TOTAL_STEPS = EPOCHS * STEPS_EPOCH
    WARMUP      = 5 * STEPS_EPOCH  # warmup 5 epochs

    lr_schedule = WarmupCosineDecay(
        base_lr=3e-4,       # giảm từ 1e-3 → 3e-4
        total_steps=TOTAL_STEPS,
        warmup_steps=WARMUP
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=bce_dice_loss,
        metrics=["accuracy", iou_metric, dice_metric]
    )

    print("\n=== Training U-Net + Attention Gate ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks()
    )

    # ---- Final Eval ----
    final_eval = model.evaluate(val_ds, verbose=0, return_dict=True)
    print("\n=== Final Validation Metrics ===")
    print(f"Val IoU : {final_eval.get('iou_metric', float('nan')):.4f}")
    print(f"Val Dice: {final_eval.get('dice_metric', float('nan')):.4f}")

    model.save("unet_person_segmentation.keras")
    print("\n✅ Model saved: unet_person_segmentation.keras")
    print("✅ Best checkpoint: checkpoints/unet_best.keras")