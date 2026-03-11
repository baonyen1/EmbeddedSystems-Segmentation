import cv2, numpy as np, os

MASK_DIR = r"C:\Users\nguye\Documents\3RD YEAR\TT_nhung\lab1\segmentation_dataset\masks"

ratios = []
bad = []

for f in sorted(os.listdir(MASK_DIR)):
    if not f.endswith(".png"): continue
    mask = cv2.imread(os.path.join(MASK_DIR, f), 0)
    if mask is None: continue
    ratio = (mask > 0).sum() / mask.size
    ratios.append((f, ratio))
    if ratio < 0.01 or ratio > 0.95:  # quá ít hoặc quá nhiều object
        bad.append((f, ratio))

print(f"Tổng: {len(ratios)} mask")
print(f"\n⚠️  Mask bất thường ({len(bad)} file):")
for f, r in bad:
    print(f"  {f}: {r:.1%}")

ratios_only = [r for _, r in ratios]
print(f"\nObject ratio - Min: {min(ratios_only):.1%} | Max: {max(ratios_only):.1%} | Mean: {np.mean(ratios_only):.1%}")