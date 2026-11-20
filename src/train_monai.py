"""Train a 3D MONAI network (patch-based) on the same data and split as the 2D U-Net.

Usage:
  python -m src.train_monai --config configs/monai3d.yaml --overfit    # sanity check on one batch
  python -m src.train_monai --config configs/monai3d.yaml              # full training
  python -m src.train_monai --config configs/monai3d.yaml --resume
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from tqdm import tqdm

from . import metrics
from .monai_utils import build_monai_model, make_items, predict_volume_3d, train_transforms
from .train import plot_curves
from .utils import add_common_args, get_device, load_config, read_json, set_seed, write_json


def validate(model, data_dir, case_ids, cfg, device):
    """Mean 3D Dice over validation patients using sliding-window inference on the full volumes."""
    scores = []
    for cid in case_ids:
        vol = np.load(Path(data_dir) / cid / "image.npy", mmap_mode="r")
        gt = np.load(Path(data_dir) / cid / "mask.npy").astype(bool)
        prob = predict_volume_3d(model, vol, device, cfg["data"]["hu_window"], cfg["train"]["patch_size"],
                                 cfg["infer"]["sw_batch_size"], cfg["infer"]["overlap"], cfg["train"]["amp"])
        scores.append(metrics.dice(prob > cfg["infer"]["threshold"], gt))
    return float(np.mean(scores))


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--overfit", action="store_true", help="train on one batch of patches to test the pipeline")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    tcfg = cfg["train"]
    set_seed(cfg["seed"])
    device = get_device()
    use_amp = tcfg["amp"] and device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    data_dir = cfg["paths"]["processed_dir"]
    splits = read_json(cfg["paths"]["splits_file"])
    print({k: len(v) for k, v in splits.items()})

    model = build_monai_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MONAI {cfg['model']['name']} parameters: {n_params / 1e6:.2f} M | patch {tcfg['patch_size']}")
    loss_fn = DiceCELoss(sigmoid=True)  # soft Dice + binary cross-entropy, same recipe as the 2D model

    items = make_items(data_dir, splits["train"])
    reps = 1 if args.overfit else tcfg["volumes_repeat"]
    ds = Dataset(data=items * reps, transform=train_transforms(cfg))
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True,
                        num_workers=0 if args.overfit else tcfg["num_workers"],
                        pin_memory=device.type == "cuda",
                        persistent_workers=(not args.overfit) and tcfg["num_workers"] > 0)
    print(f"Training volumes: {len(items)} | steps per epoch: {len(loader)} | "
          f"patches per step: {tcfg['batch_size'] * tcfg['patches_per_volume']}")

    # ------------------------------------------------------------------ overfit sanity test
    if args.overfit:
        batch = next(iter(loader))
        x, y = batch["image"].to(device), batch["label"].to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for it in range(1, 301):
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward(); opt.step()
            if it % 50 == 0 or it == 1:
                pred = (torch.sigmoid(logits.detach()) > 0.5).float()
                d = (2 * (pred * y).sum() / (pred.sum() + y.sum()).clamp(min=1)).item()
                print(f"iter {it:3d} | loss {loss.item():.4f} | dice {d:.4f}")
        print("PASS if Dice approaches 1.0. If not, there is a bug in data, model or loss.")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"], eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = Path(cfg["paths"]["runs_dir"]) / tcfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg, run_dir / "config_used.json")
    log_csv = run_dir / "log.csv"
    log, start_epoch, best = [], 1, -1.0
    if args.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        log = pd.read_csv(log_csv).to_dict("records") if log_csv.exists() else []
        print(f"Resumed from epoch {ck['epoch']} (best val dice so far {best:.4f})")

    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        model.train()
        t0, losses = time.time(), []
        for batch in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            x, y = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()

        val_dice = float("nan")
        if epoch % tcfg["val_every"] == 0 or epoch == tcfg["epochs"]:
            val_dice = validate(model, data_dir, splits["val"], cfg, device)
            if val_dice > best:
                best = val_dice
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_dice": val_dice},
                           run_dir / "best.pt")
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_dice": val_dice,
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        log.append(row)
        pd.DataFrame(log).to_csv(log_csv, index=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "cfg": cfg}, run_dir / "last.pt")
        print(f"epoch {epoch:3d}/{tcfg['epochs']} | loss {row['train_loss']:.4f} | val dice {val_dice:.4f} "
              f"| best {best:.4f} | {row['seconds']}s")

    plot_curves(log_csv, run_dir / "curves.png")
    print(f"Done. Best validation Dice {best:.4f}. Checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
