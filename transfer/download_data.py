import argparse
import os
from typing import Tuple

from datasets import load_dataset


def to_folder(src: str, tgt: str) -> Tuple[str, str]:
    # Our convention is data/{tgt}_{src}/
    return f"data/{tgt}_{src}", f"{tgt}_{src}"


def save_split(ds, split: str, out_dir: str) -> None:
    path = os.path.join(out_dir, f"{split}.tsv")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for eg in ds:
            src = eg["translation"][args.src]
            tgt = eg["translation"][args.tgt]
            if src is None or tgt is None:
                continue
            f.write(src.replace("\n", " ") + "\t" + tgt.replace("\n", " ") + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="opus100")
    p.add_argument("--src", type=str, required=True)
    p.add_argument("--tgt", type=str, required=True)
    p.add_argument("--max_train", type=int, default=0, help="limit train samples (0 for all)")
    p.add_argument("--max_valid", type=int, default=2000, help="limit valid samples")
    p.add_argument("--max_test", type=int, default=2000, help="limit test samples")
    return p.parse_args()


def main() -> None:
    global args
    args = parse_args()
    if args.dataset.lower() in {"opus100", "opus-100", "opus_100"}:
        name = f"{args.src}-{args.tgt}"
        if args.src > args.tgt:
            name = f"{args.tgt}-{args.src}"
        ds = load_dataset("opus100", name)
    else:
        raise ValueError("Unsupported dataset")

    out_dir, folder = to_folder(args.src, args.tgt)

    train = ds.get("train")
    valid = ds.get("validation") or ds.get("dev") or ds.get("validation1")
    test = ds.get("test")

    if train is not None:
        if args.max_train and len(train) > args.max_train:
            train = train.select(range(args.max_train))
        save_split(train, "train", out_dir)
    if valid is not None:
        if args.max_valid and len(valid) > args.max_valid:
            valid = valid.select(range(args.max_valid))
        save_split(valid, "valid", out_dir)
    if test is not None:
        if args.max_test and len(test) > args.max_test:
            test = test.select(range(args.max_test))
        save_split(test, "test", out_dir)

    print(f"Saved splits to {out_dir}")


if __name__ == "__main__":
    main()

