import os, cv2, random, argparse
import numpy as np
from glob import glob

IMG_EXTS = ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG","*.bmp","*.webp")

def augment(img):
    # brightness/contrast
    alpha = random.uniform(0.80, 1.20)
    beta = random.randint(-25, 25)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # gamma (low-light / over-exposure simulation)
    if random.random() < 0.5:
        gamma = random.uniform(0.7, 1.5)
        inv = 1.0 / gamma
        table = (np.arange(256) / 255.0) ** inv * 255.0
        img = cv2.LUT(img, table.astype(np.uint8))

    # blur
    if random.random() < 0.35:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # jpeg artifact
    if random.random() < 0.25:
        q = random.randint(35, 80)
        enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])[1]
        img = cv2.imdecode(enc, 1)

    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_img = os.path.join(args.out, "images")
    out_lbl = os.path.join(args.out, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    img_files = []
    for e in IMG_EXTS:
        img_files += glob(os.path.join(args.images, e))
    img_files = sorted(img_files)

    saved = 0
    for img_path in img_files:
        name = os.path.basename(img_path)
        stem = os.path.splitext(name)[0]

        img = cv2.imread(img_path)
        if img is None:
            continue

        aug = augment(img)
        cv2.imwrite(os.path.join(out_img, name), aug)

        label_src = os.path.join(args.labels, stem + ".txt")
        label_dst = os.path.join(out_lbl, stem + ".txt")
        if os.path.exists(label_src):
            with open(label_src, "r", encoding="utf-8") as r, open(label_dst, "w", encoding="utf-8") as w:
                w.write(r.read())
            saved += 1

    print(f"[Photometric] generated={saved} out={args.out}")

if __name__ == "__main__":
    main()
