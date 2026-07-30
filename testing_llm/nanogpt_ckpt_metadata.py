"""Print metadata from a nanoGPT-style checkpoint (.pt) file.

Loads a checkpoint saved by train.py and reports everything except the
large weight tensors: model_args, iter_num, best_val_loss, config, and a
summary of the model/optimizer state dicts (shapes + parameter count).

Usage:
    python testing_llm/nanogpt_ckpt_metadata.py path/to/ckpt.pt
    python testing_llm/nanogpt_ckpt_metadata.py path/to/ckpt.pt --weights
"""
import argparse
import os

import torch


def human(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n}"
        n /= 1000
    return f"{n:.1f}T"


def summarize_state_dict(name, sd, list_all=False):
    if not isinstance(sd, dict):
        print(f"  {name}: (not a state dict: {type(sd).__name__})")
        return
    total = 0
    tensors = 0
    for k, v in sd.items():
        if torch.is_tensor(v):
            tensors += 1
            total += v.numel()
            if list_all:
                print(f"    {k:55s} {tuple(v.shape)}  {v.dtype}")
    print(f"  {name}: {tensors} tensors, {human(total)} params")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt", help="path to checkpoint .pt file")
    ap.add_argument("--weights", action="store_true",
                    help="list every tensor in the model state dict")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"file not found: {args.ckpt}")

    size_mb = os.path.getsize(args.ckpt) / (1024 * 1024)
    print(f"file: {args.ckpt}  ({size_mb:.1f} MB)\n")

    # weights_only=False because the checkpoint stores a config object too
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        print(f"checkpoint is a {type(ckpt).__name__}, not a dict")
        return

    print("top-level keys:", list(ckpt.keys()), "\n")

    if "iter_num" in ckpt:
        print(f"iter_num:      {ckpt['iter_num']}")
    if "best_val_loss" in ckpt:
        bvl = ckpt["best_val_loss"]
        bvl = bvl.item() if torch.is_tensor(bvl) else bvl
        print(f"best_val_loss: {bvl}")
    print()

    if "model_args" in ckpt:
        print("model_args:")
        for k, v in ckpt["model_args"].items():
            print(f"  {k:15s} = {v}")
        print()

    if "config" in ckpt:
        print("config:")
        for k, v in sorted(ckpt["config"].items()):
            print(f"  {k:25s} = {v}")
        print()

    if "model" in ckpt:
        summarize_state_dict("model", ckpt["model"], list_all=args.weights)
    if "optimizer" in ckpt:
        summarize_state_dict("optimizer state", ckpt["optimizer"].get("state", {}))


if __name__ == "__main__":
    main()
