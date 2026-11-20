"""Train the 2D U-Net.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from . import metrics
from .dataset import FemurSliceDataset
from .inference import predict_volume
from .losses import DiceBCELoss
from .model import build_model, count_parameters
from .utils import add_common_args, get_device, load_config, read_json, set_seed, write_json


def batch_dice(logits, target, thr=0.5):
    pred = (torch.sigmoid(logits.float()) > thr).float()
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()
    return float((2 * inter / denom).item()) if denom > 0 else 1.0


def validate(model, data_dir, case_ids, cfg, device):
    """Mean 3D Dice over validation patients (full volumes, no post-processing)."""
    scores = []
    for cid in case_ids:
        vol = np.load(Path(data_dir) / cid / "image.npy", mmap_mode="r")
        gt = np.load(Path(data_dir) / cid / "mask.npy").astype(bool)
        prob = predict_volume(model, vol, device, cfg["data"]["hu_window"],
                              batch_size=cfg["infer"]["batch_size"], amp=cfg["train"]["amp"])
        scores.append(metrics.dice(prob > 0.5, gt))
    return float(np.mean(scores))


def plot_curves(log_csv, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(log_csv)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(df["epoch"], df["train_loss"]); ax[0].set_title("Train loss (Dice + BCE)"); ax[0].set_xlabel("epoch")
    v = df.dropna(subset=["val_dice"])
    ax[1].plot(v["epoch"], v["val_dice"], marker="o"); ax[1].set_title("Validation 3D Dice"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--overfit", action="store_true", help="train on a single batch to test the pipeline")
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

    model = build_model(cfg["model"]).to(device)
    print(f"U-Net parameters: {count_parameters(model) / 1e6:.2f} M")
    criterion = DiceBCELoss()

    # ------------------------------------------------------------------ overfit-one-batch sanity test
    if args.overfit:
        ds = FemurSliceDataset(data_dir, splits["train"], cfg["data"]["hu_window"], augment=False,
                               neg_ratio=0.0, seed=cfg["seed"])
        loader = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True, num_workers=0)
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for it in range(1, 301):
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward(); opt.step()
            if it % 25 == 0 or it == 1:
                with torch.no_grad():
                    d = batch_dice(model(x), y)
                print(f"iter {it:3d} | loss {loss.item():.4f} | dice {d:.4f}")
        print("PASS if Dice is close to 1.0 (> 0.95). If not, there is a bug in data, model or loss.")
        return

    # ------------------------------------------------------------------ normal training
    train_ds = FemurSliceDataset(data_dir, splits["train"], cfg["data"]["hu_window"], augment=True,
                                 aug_cfg=cfg["augment"], neg_ratio=tcfg["neg_ratio"], seed=cfg["seed"])
    n_samples = min(tcfg["samples_per_epoch"], len(train_ds))
    sampler = RandomSampler(train_ds, replacement=False, num_samples=n_samples)
    loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], sampler=sampler,
                        num_workers=tcfg["num_workers"], pin_memory=device.type == "cuda",
                        drop_last=True, persistent_workers=tcfg["num_workers"] > 0)
    print(f"Training slices available: {len(train_ds)} | per epoch: {n_samples}")

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
        for x, y in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = criterion(logits, y)
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
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "cfg": cfg},
                   run_dir / "last.pt")
        print(f"epoch {epoch:3d}/{tcfg['epochs']} | loss {row['train_loss']:.4f} | val dice {val_dice:.4f} "
              f"| best {best:.4f} | {row['seconds']}s")

    plot_curves(log_csv, run_dir / "curves.png")
    print(f"Done. Best validation Dice {best:.4f}. Checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
