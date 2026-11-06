"""Train ResNet18 (transfer learning) and keep the best model by validation macro-F1.

Usage:
    python train.py --epochs 10
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from common import OCTDataset, build_model, get_device, make_transforms, set_seed


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise evaluates."""
    train = optimizer is not None
    model.train(train)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds += out.argmax(1).cpu().tolist()
            targets += y.cpu().tolist()
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    df = pd.read_csv(args.manifest)
    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]

    train_tfm, eval_tfm = make_transforms(args.img_size)
    train_dl = DataLoader(OCTDataset(train_df, class_to_idx, train_tfm), batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(OCTDataset(val_df, class_to_idx, eval_tfm), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # Class weights inversely proportional to class frequency (handles imbalance)
    counts = train_df["label"].value_counts().reindex(classes).fillna(1).values
    weights = torch.tensor(len(train_df) / (len(classes) * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_model(len(classes), pretrained=True, freeze=True).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_f1 = [], -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_f1 = run_epoch(model, train_dl, device, criterion, optimizer)
        va_loss, va_f1 = run_epoch(model, val_dl, device, criterion)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_f1=tr_f1,
                            val_loss=va_loss, val_f1=va_f1))
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.3f} F1 {tr_f1:.3f} | "
              f"val loss {va_loss:.3f} F1 {va_f1:.3f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "img_size": args.img_size}, out_dir / "best.pt")

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"\nBest validation macro-F1: {best_f1:.3f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
