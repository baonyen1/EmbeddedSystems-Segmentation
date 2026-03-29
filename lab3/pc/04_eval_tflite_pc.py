import argparse
import os

import cv2
import numpy as np

try:
	from tflite_runtime.interpreter import Interpreter
except ModuleNotFoundError:
	import tensorflow as tf

	Interpreter = tf.lite.Interpreter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MODEL_INT8 = os.path.join(ROOT_DIR, "lab3", "models", "unet_int8.tflite")
MODEL_FP32 = os.path.join(ROOT_DIR, "lab3", "models", "unet_fp32.tflite")
IMG_DIR = os.path.join(ROOT_DIR, "lab1", "dataset_split", "images", "test")
MASK_DIR = os.path.join(ROOT_DIR, "lab1", "dataset_split", "masks", "test")
OUT_DIR = os.path.join(ROOT_DIR, "lab3", "outputs", "eval_demo_5")

IMG_SIZE = (256, 256)
THRESH = 0.5


def build_interpreter(model_path, num_threads=4):
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Missing model: {model_path}")
	itp = Interpreter(model_path=model_path, num_threads=num_threads)
	itp.allocate_tensors()
	return itp


def preprocess_for_input(img_bgr, input_detail):
	img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
	x = img.astype(np.float32) / 255.0

	if input_detail["dtype"] == np.float32:
		return np.expand_dims(x, axis=0).astype(np.float32)

	scale, zero = input_detail["quantization"]
	if scale == 0:
		raise ValueError("Input quantization scale is 0, cannot quantize input")

	xq = np.round(x / scale + zero)
	qinfo = np.iinfo(input_detail["dtype"])
	xq = np.clip(xq, qinfo.min, qinfo.max)
	return np.expand_dims(xq.astype(input_detail["dtype"]), axis=0)


def dequantize_output(output, output_detail):
	if output_detail["dtype"] == np.float32:
		return output.astype(np.float32)

	scale, zero = output_detail["quantization"]
	if scale == 0:
		return output.astype(np.float32)
	return (output.astype(np.float32) - zero) * scale


def run_inference(interpreter, img_bgr):
	in_det = interpreter.get_input_details()[0]
	out_det = interpreter.get_output_details()[0]

	x = preprocess_for_input(img_bgr, in_det)
	interpreter.set_tensor(in_det["index"], x)
	interpreter.invoke()
	y = interpreter.get_tensor(out_det["index"])
	y = dequantize_output(y, out_det)

	if y.ndim == 4:
		y = y[0]
	if y.ndim == 3 and y.shape[-1] == 1:
		y = y[..., 0]
	return y


def dice_iou(pred, gt):
	pred = pred.astype(np.bool_)
	gt = gt.astype(np.bool_)
	inter = np.logical_and(pred, gt).sum()
	union = np.logical_or(pred, gt).sum()
	dice = (2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
	iou = (inter + 1e-6) / (union + 1e-6)
	return float(dice), float(iou)


def list_test_files(img_dir):
	exts = (".jpg", ".jpeg", ".png")
	files = [f for f in os.listdir(img_dir) if f.lower().endswith(exts)]
	return sorted(files)


def to_color_mask(mask_bool):
	mask_u8 = (mask_bool.astype(np.uint8) * 255)
	return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def add_title(img_bgr, title):
	canvas = img_bgr.copy()
	cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
	return canvas


def save_visualization(out_path, img_bgr, gt, pred_int8):
	img = cv2.resize(img_bgr, (IMG_SIZE[1], IMG_SIZE[0]))
	gt_color = to_color_mask(gt)
	i8_color = to_color_mask(pred_int8)
	overlay = cv2.addWeighted(img, 0.65, i8_color, 0.35, 0.0)

	t1 = cv2.hconcat([
		add_title(img, "Input"),
		add_title(gt_color, "Ground truth"),
	])
	t2 = cv2.hconcat([
		add_title(i8_color, "Mask detect"),
		add_title(overlay, "Overlay"),
	])

	grid = cv2.vconcat([t1, t2])
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	cv2.imwrite(out_path, grid)


def main():
	parser = argparse.ArgumentParser(description="Evaluate baseline UNet int8 TFLite")
	parser.add_argument("--num-samples", type=int, default=5, help="Number of test images")
	parser.add_argument("--threads", type=int, default=4, help="TFLite interpreter threads")
	parser.add_argument("--out-dir", type=str, default=OUT_DIR, help="Folder to save demo visualizations")
	args = parser.parse_args()

	int8_itp = build_interpreter(MODEL_INT8, num_threads=args.threads)
	fp32_itp = build_interpreter(MODEL_FP32, num_threads=args.threads)

	files = list_test_files(IMG_DIR)[: args.num_samples]
	if not files:
		raise RuntimeError(f"No test images found in: {IMG_DIR}")

	dices_int8, ious_int8 = [], []
	dices_fp32, ious_fp32 = [], []
	mae_probs = []

	print("=== Demo eval baseline UNet int8 (with fp32 comparison) ===")
	print(f"Model int8 : {MODEL_INT8}")
	print(f"Model fp32 : {MODEL_FP32}")
	print(f"Samples    : {len(files)}")
	print(f"Save viz   : {args.out_dir}")
	print("-" * 92)
	print(f"{'file':40s} {'Dice_i8':>8s} {'IoU_i8':>8s} {'Dice_f32':>9s} {'IoU_f32':>8s} {'MAE':>8s}")
	print("-" * 92)

	for f in files:
		stem, _ = os.path.splitext(f)
		img_path = os.path.join(IMG_DIR, f)
		mask_path = os.path.join(MASK_DIR, stem + ".png")

		img = cv2.imread(img_path)
		msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		if img is None or msk is None:
			print(f"Skip {f}: missing image/mask")
			continue

		gt = cv2.resize(msk, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST) >= 128

		prob_int8 = run_inference(int8_itp, img)
		prob_fp32 = run_inference(fp32_itp, img)

		pred_int8 = prob_int8 >= THRESH
		pred_fp32 = prob_fp32 >= THRESH

		d_i8, i_i8 = dice_iou(pred_int8, gt)
		d_f32, i_f32 = dice_iou(pred_fp32, gt)
		mae = float(np.mean(np.abs(prob_int8 - prob_fp32)))

		dices_int8.append(d_i8)
		ious_int8.append(i_i8)
		dices_fp32.append(d_f32)
		ious_fp32.append(i_f32)
		mae_probs.append(mae)

		viz_path = os.path.join(args.out_dir, f"{stem}_demo.png")
		save_visualization(viz_path, img, gt, pred_int8)

		short_name = f if len(f) <= 40 else f[:37] + "..."
		print(f"{short_name:40s} {d_i8:8.4f} {i_i8:8.4f} {d_f32:9.4f} {i_f32:8.4f} {mae:8.5f}")

	if not dices_int8:
		raise RuntimeError("No valid samples evaluated")

	print("-" * 92)
	print(f"{'MEAN':40s} {np.mean(dices_int8):8.4f} {np.mean(ious_int8):8.4f} {np.mean(dices_fp32):9.4f} {np.mean(ious_fp32):8.4f} {np.mean(mae_probs):8.5f}")
	print(
		f"Delta (int8 - fp32): Dice={np.mean(dices_int8) - np.mean(dices_fp32):+.4f}, "
		f"IoU={np.mean(ious_int8) - np.mean(ious_fp32):+.4f}"
	)
	print(f"Saved {len(dices_int8)} visualization images to: {args.out_dir}")


if __name__ == "__main__":
	main()
