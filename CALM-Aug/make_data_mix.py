import os, argparse, yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original data.yaml")
    ap.add_argument("--train_images", required=True, help="absolute path to train images dir")
    ap.add_argument("--out", required=True, help="output yaml path")
    args = ap.parse_args()

    with open(args.src, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data["train"] = args.train_images  # absolute path works
    # keep val/test unchanged
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print("Saved:", args.out)
    print("train:", data["train"])
    print("val  :", data.get("val"))
    print("test :", data.get("test"))

if __name__ == "__main__":
    main()
