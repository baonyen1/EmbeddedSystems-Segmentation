import os, random, shutil

SRC_DIR = r"/home/bao/Documents/workspace/EmbeddedSystems-Segmentation/lab1/dataset_split/images/train"   # có thể đổi sang images/val
OUT_DIR = "rep_images"
N = 500
SEED = 42

def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    imgs = [f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg",".png",".jpeg"))]
    random.shuffle(imgs)
    imgs = imgs[:N]

    for i, f in enumerate(imgs):
        src = os.path.join(SRC_DIR, f)
        dst = os.path.join(OUT_DIR, f"{i:05d}_" + f)
        shutil.copy2(src, dst)

    print("Saved rep images:", len(imgs), "to", OUT_DIR)

if __name__ == "__main__":
    main()