import os, argparse, shutil
from glob import glob
from ultralytics import YOLO

IMG_EXTS = ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG","*.bmp","*.webp")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.teacher)

    out_img = os.path.join(args.out, "images")
    out_lbl = os.path.join(args.out, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    img_files = []
    for e in IMG_EXTS:
        img_files += glob(os.path.join(args.images, e))
    img_files = sorted(img_files)

    kept = 0
    total = 0
    for p in img_files:
        total += 1
        name = os.path.basename(p)
        stem = os.path.splitext(name)[0]
        lab = os.path.join(args.labels, stem + ".txt")
        if not os.path.exists(lab):
            continue

        r = model.predict(p, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        ok = (r.boxes is not None) and (len(r.boxes) > 0)

        if ok:
            shutil.copy2(p, os.path.join(out_img, name))
            shutil.copy2(lab, os.path.join(out_lbl, stem + ".txt"))
            kept += 1

    print(f"[TeacherFilter] kept={kept}/{total} (conf>={args.conf}) out={args.out}")

if __name__ == "__main__":
    main()
