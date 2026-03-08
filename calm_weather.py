import os, cv2, random, argparse
import numpy as np
from glob import glob

IMG_EXTS = ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG","*.bmp","*.webp")

def add_haze(img, strength=(0.08, 0.22)):
    a = random.uniform(*strength)
    haze = np.full_like(img, 255, dtype=np.uint8)
    out = cv2.addWeighted(img, 1 - a, haze, a, 0)
    return out

def add_shadow(img, alpha=(0.5, 0.8)):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1 = random.randint(0, w//2), random.randint(0, h)
    x2, y2 = random.randint(w//2, w), random.randint(0, h)
    x3, y3 = random.randint(w//2, w), random.randint(0, h)
    pts = np.array([[x1,y1],[x2,y2],[x3,y3]], np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    a = random.uniform(*alpha)
    shadow = img.copy()
    shadow[mask==255] = (shadow[mask==255] * a).astype(np.uint8)
    return shadow

def color_shift(img):
    # mild temperature shift
    b = random.randint(-10, 10)
    g = random.randint(-10, 10)
    r = random.randint(-10, 10)
    out = img.astype(np.int16)
    out[:,:,0] = np.clip(out[:,:,0] + b, 0, 255)
    out[:,:,1] = np.clip(out[:,:,1] + g, 0, 255)
    out[:,:,2] = np.clip(out[:,:,2] + r, 0, 255)
    return out.astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p_haze", type=float, default=0.35)
    ap.add_argument("--p_shadow", type=float, default=0.35)
    ap.add_argument("--p_shift", type=float, default=0.45)
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

        if random.random() < args.p_haze:
            img = add_haze(img)
        if random.random() < args.p_shadow:
            img = add_shadow(img)
        if random.random() < args.p_shift:
            img = color_shift(img)

        cv2.imwrite(os.path.join(out_img, name), img)

        ls = os.path.join(args.labels, stem + ".txt")
        ld = os.path.join(out_lbl, stem + ".txt")
        if os.path.exists(ls):
            with open(ls, "r", encoding="utf-8") as r, open(ld, "w", encoding="utf-8") as w:
                w.write(r.read())
            saved += 1

    print(f"[Weather] generated={saved} out={args.out}")

if __name__ == "__main__":
    main()
