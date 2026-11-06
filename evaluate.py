"""Evaluate the best model on the test split and generate Grad-CAM figures.

Outputs in outputs/: test_report.txt, confusion_matrix.png, gradcam.png

Usage:
    python evaluate.py
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from torch.utils.data import DataLoader

from common import OCTDataset, build_model, get_device, grad_cam, make_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cam-per-class", type=int, default=2)
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes, img_size = ckpt["classes"], ckpt["img_size"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    model = build_model(len(classes), pretrained=False, freeze=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    df = pd.read_csv(args.manifest)
    test_df = df[df.split == "test"].reset_index(drop=True)
    _, eval_tfm = make_transforms(img_size)
    dl = DataLoader(OCTDataset(test_df, class_to_idx, eval_tfm), batch_size=args.batch_size)

    # --- Test-set predictions ---
    probs, y_true = [], []
    with torch.no_grad():
        for x, y in dl:
            probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
            y_true += y.tolist()
    probs = np.concatenate(probs)
    y_true = np.array(y_true)
    y_pred = probs.argmax(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(classes)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                   digits=3, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        auc = float("nan")  # a class is missing from the test set
    summary = f"{report}\nMacro F1: {macro_f1:.3f}\nMacro AUC (OvR): {auc:.3f}\n"
    print(summary)
    (out_dir / "test_report.txt").write_text(summary)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- Grad-CAM: a few test images per class ---
    n = args.cam_per_class
    fig, axes = plt.subplots(len(classes), n, figsize=(3.2 * n, 3.2 * len(classes)), squeeze=False)
    for ci, cname in enumerate(classes):
        subset = test_df[test_df.label == cname].head(n)
        for j in range(n):
            ax = axes[ci][j]
            ax.axis("off")
            if j >= len(subset):
                continue
            pil = Image.open(subset.iloc[j]["path"]).convert("RGB")
            x = eval_tfm(pil).unsqueeze(0).to(device)
            cam, pred_idx = grad_cam(model, x, model.layer4)
            ax.imshow(pil.resize((img_size, img_size)))
            ax.imshow(cam, cmap="jet", alpha=0.4, extent=(0, img_size, img_size, 0),
                      interpolation="bilinear")
            ax.set_title(f"true: {cname} | pred: {classes[pred_idx]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gradcam.png", dpi=150)
    plt.close(fig)
    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
