import os
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "C:\\Users\\nguye\\Desktop\\workspace\\EmbeddedSystems-Segmentation\\lab4\\models_out\\unet_pruned_075x_fp32.tflite"
IMG_DIR = "C:\\Users\\nguye\\Desktop\\workspace\\EmbeddedSystems-Segmentation\\lab4\\dataset_split\\images\\test"
MASK_DIR = "C:\\Users\\nguye\\Desktop\\workspace\\EmbeddedSystems-Segmentation\\lab4\\dataset_split\\masks\\test"
IMG_SIZE = (256, 256)
THRESH = 0.5
NUM_SAMPLES = 50  # giới hạn để chạy nhanh

def preprocess_quant(img_bgr, in_det):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    x = img.astype(np.float32) / 255.0
    scale, zero = in_det["quantization"]
    if in_det["dtype"] == np.uint8:
        xq = (x / scale + zero).astype(np.uint8)
    else:
        xq = (x / scale + zero).astype(np.int8)
    return np.expand_dims(xq, axis=0)

def dequant_output(yq, out_det):
    scale, zero = out_det["quantization"]
    if scale == 0:
        return yq.astype(np.float32)
    return (yq.astype(np.float32) - zero) * scale

def dice_iou(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(dice), float(iou)

def main():
    itp = Interpreter(model_path=MODEL_PATH, num_threads=4)
    itp.allocate_tensors()
    in_det = itp.get_input_details()[0]
    out_det = itp.get_output_details()[0]

    files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg",".png",".jpeg"))]
    files = sorted(files)[:NUM_SAMPLES]

    dices, ious = [], []
    for f in files:
        base = os.path.splitext(f)[0]
        img = cv2.imread(os.path.join(IMG_DIR, f))
        msk = cv2.imread(os.path.join(MASK_DIR, base + ".png"), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue

        xq = preprocess_quant(img, in_det)
        itp.set_tensor(in_det["index"], xq)
        itp.invoke()
        yq = itp.get_tensor(out_det["index"])
        y = dequant_output(yq, out_det)
        if y.ndim == 4:
            y = y[0]
        if y.shape[-1] == 1:
            y = y[..., 0]

        pred = (y >= THRESH)
        gt = cv2.resize(msk, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST) >= 128

        d, i = dice_iou(pred, gt)
        dices.append(d); ious.append(i)

    print("Samples:", len(dices))
    print("Dice mean:", float(np.mean(dices)))
    print("IoU mean:", float(np.mean(ious)))

if __name__ == "__main__":
    main()
