import os, cv2, random, argparse
import numpy as np
from glob import glob

IMG_EXTS = ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG","*.bmp","*.webp")

def cutout(img, n_holes=(1,3), hole_ratio=(0.08, 0.22)):
    h, w = img.shape[:2]
    nh = random.randint(n_holes[0], n_holes[1])
    for _ in range(nh):
        rr = random.uniform(hole_ratio[0], hole_ratio[1])
        ch = int(h * rr)
        cw = int(w * rr)
        y1 = random.randint(0, max(0, h - ch - 1))
        x1 = random.randint(0, max(0, w - cw - 1))
        y2 = min(h, y1 + ch)
        x2 = min(w, x1 + cw)

        if random.random() < 0.5:
            img[y1:y2, x1:x2] = 0
        else:
            img[y1:y2, x1:x2] = np.random.randint(0, 255, (y2-y1, x2-x1, 3), dtype=np.uint8)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", type=float, default=0.6)
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
    for pth in img_files:
        name = os.path.basename(pth)
        stem = os.path.splitext(name)[0]

        img = cv2.imread(pth)
        if img is None:
            continue

        if random.random() < args.p:
            img = cutout(img)

        cv2.imwrite(os.path.join(out_img, name), img)

        ls = os.path.join(args.labels, stem + ".txt")
        ld = os.path.join(out_lbl, stem + ".txt")
        if os.path.exists(ls):
            with open(ls, "r", encoding="utf-8") as r, open(ld, "w", encoding="utf-8") as w:
                w.write(r.read())
            saved += 1

    print(f"[Occlusion] generated={saved} out={args.out}")

if __name__ == "__main__":
    main()
