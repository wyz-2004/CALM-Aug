import os, json, argparse
from glob import glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="YOLO labels dir")
    ap.add_argument("--nc", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    counts = [0] * args.nc
    files = sorted(glob(os.path.join(args.labels, "*.txt")))
    for f in files:
        with open(f, "r", encoding="utf-8") as r:
            for line in r:
                s = line.strip()
                if not s:
                    continue
                c = int(s.split()[0])
                if 0 <= c < args.nc:
                    counts[c] += 1

    stats = {
        "nc": args.nc,
        "instances_per_class": counts,
        "min": int(min(counts)) if counts else 0,
        "max": int(max(counts)) if counts else 0,
        "sum": int(sum(counts)) if counts else 0,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        json.dump(stats, w, indent=2)

    pairs = sorted(list(enumerate(counts)), key=lambda x: x[1])
    print("Class instance counts (sorted asc):")
    for i, c in pairs[: args.topk]:
        print(f"  class {i:02d}: {c}")
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
