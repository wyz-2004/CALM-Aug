import os, json, cv2, random, argparse
from glob import glob

IMG_EXTS = ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG","*.bmp","*.webp")

def read_label(path):
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 5:
                boxes.append([float(x) for x in p])  # cls x y w h
    return boxes

def save_label(path, boxes):
    with open(path, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

def yolo_to_xyxy(box, W, H):
    _, x, y, bw, bh = box
    x1 = int((x - bw/2) * W); y1 = int((y - bh/2) * H)
    x2 = int((x + bw/2) * W); y2 = int((y + bh/2) * H)
    x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def xyxy_to_yolo(cls, x1, y1, x2, y2, W, H):
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    # clamp to [0,1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return [float(cls), cx, cy, bw, bh]

def paste_patch(src_img, src_box, dst_img):
    H, W = src_img.shape[:2]
    xyxy = yolo_to_xyxy(src_box, W, H)
    if xyxy is None:
        return None, None
    x1,y1,x2,y2 = xyxy
    patch = src_img[y1:y2, x1:x2]
    if patch is None or patch.size == 0:
        return None, None

    ph, pw = patch.shape[:2]
    Hd, Wd = dst_img.shape[:2]
    if ph >= Hd or pw >= Wd:
        return None, None

    px = random.randint(0, Wd - pw - 1)
    py = random.randint(0, Hd - ph - 1)

    dst_img[py:py+ph, px:px+pw] = patch
    return dst_img, xyxy_to_yolo(int(src_box[0]), px, py, px+pw, py+ph, Wd, Hd)

def build_repeat_map(counts, extreme_th=10, tail_th=50, mid_th=100,
                     r_extreme=8, r_tail=4, r_mid=2, r_head=0):
    rep = {}
    for c, n in enumerate(counts):
        if n < extreme_th:
            rep[c] = r_extreme
        elif n < tail_th:
            rep[c] = r_tail
        elif n < mid_th:
            rep[c] = r_mid
        else:
            rep[c] = r_head
    return rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--stats", required=True, help="class_stats.json from stat_yolo_classes.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)

    # thresholds
    ap.add_argument("--extreme_th", type=int, default=10)
    ap.add_argument("--tail_th", type=int, default=50)
    ap.add_argument("--mid_th", type=int, default=100)

    # repeats (how many augmented images to generate per instance occurrence)
    ap.add_argument("--r_extreme", type=int, default=8)
    ap.add_argument("--r_tail", type=int, default=4)
    ap.add_argument("--r_mid", type=int, default=2)
    ap.add_argument("--r_head", type=int, default=0)

    # per augmented image, how many paste operations
    ap.add_argument("--max_paste", type=int, default=2)

    # cap to prevent explosion
    ap.add_argument("--max_gen_total", type=int, default=12000)

    args = ap.parse_args()
    random.seed(args.seed)

    with open(args.stats, "r", encoding="utf-8") as f:
        stats = json.load(f)
    counts = stats["instances_per_class"]
    nc = stats["nc"]

    rep_map = build_repeat_map(
        counts,
        extreme_th=args.extreme_th, tail_th=args.tail_th, mid_th=args.mid_th,
        r_extreme=args.r_extreme, r_tail=args.r_tail, r_mid=args.r_mid, r_head=args.r_head
    )

    tail_classes = [c for c, r in rep_map.items() if r > 0]
    print("[ClassAware] nc=", nc)
    print("[ClassAware] tail classes:", tail_classes[:20], "..." if len(tail_classes) > 20 else "")
    print("[ClassAware] repeat map (first 30):", {k: rep_map[k] for k in range(min(30, nc))})

    out_img = os.path.join(args.out, "images")
    out_lbl = os.path.join(args.out, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    img_files = []
    for e in IMG_EXTS:
        img_files += glob(os.path.join(args.images, e))
    img_files = sorted(img_files)

    uid = 0
    used_src = 0
    saved = 0

    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(args.labels, stem + ".txt")
        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        boxes = read_label(label_path)
        if not boxes:
            continue

        # find tail boxes present in this image
        tail_boxes = [b for b in boxes if int(b[0]) in rep_map and rep_map[int(b[0])] > 0]
        if not tail_boxes:
            continue

        used_src += 1

        # compute how many aug images to generate for this image:
        # sum of repeats for tail classes present (capped)
        repeats = 0
        for b in tail_boxes:
            repeats += rep_map[int(b[0])]
        repeats = max(1, min(repeats, 20))  # per-image cap to avoid too many from one image

        for _ in range(repeats):
            if saved >= args.max_gen_total:
                print(f"[ClassAware] reach max_gen_total={args.max_gen_total}, stop.")
                print(f"[ClassAware] used_src_images={used_src}, generated={saved}")
                return

            dst = img.copy()
            new_boxes = [b[:] for b in boxes]

            # do max_paste times, each time choose a tail box (biased to rarer via repeats)
            for _k in range(args.max_paste):
                b = random.choice(tail_boxes)
                dst2, nb = paste_patch(img, b, dst)
                if nb is not None:
                    dst = dst2
                    new_boxes.append(nb)

            out_name = f"ca_{uid:07d}.jpg"
            out_lbl_name = f"ca_{uid:07d}.txt"
            cv2.imwrite(os.path.join(out_img, out_name), dst)
            save_label(os.path.join(out_lbl, out_lbl_name), new_boxes)

            uid += 1
            saved += 1

    print(f"[ClassAware] used_src_images={used_src} generated={saved} out={args.out}")

if __name__ == "__main__":
    main()
