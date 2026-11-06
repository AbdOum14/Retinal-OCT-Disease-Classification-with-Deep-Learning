"""Build data/manifest.csv with image path, class, patient_id and split (train/val/test).

The split is done BY PATIENT (no patient appears in more than one split) and is
stratified by class as far as possible.

Usage:
    python prepare_data.py --data-dir PATH/TO/OCTDL --csv PATH/TO/metadata.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def find_col(df: pd.DataFrame, keyword: str) -> str:
    for c in df.columns:
        if keyword in c.lower():
            return c
    raise SystemExit(
        f"No column containing '{keyword}' found in the CSV. "
        f"Available columns: {list(df.columns)}"
    )


def index_images(data_dir: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for img in class_dir.rglob("*"):
            if img.suffix.lower() in IMG_EXT:
                rows.append({"path": str(img.resolve()), "label": class_dir.name, "stem": img.stem})
    if not rows:
        raise SystemExit(f"No images found in subfolders of {data_dir}")
    return pd.DataFrame(rows)


def split_by_patient(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """~60% train / 20% val / 20% test, grouped by patient and stratified by class."""
    df = df.reset_index(drop=True)
    df["split"] = "train"

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(sgkf.split(df, df["label"], df["patient_id"]))
    df.loc[test_idx, "split"] = "test"

    rest = df.loc[rest_idx]
    sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    tr_idx, val_idx = next(sgkf2.split(rest, rest["label"], rest["patient_id"]))
    df.loc[rest.index[val_idx], "split"] = "val"
    return df


def build_manifest(data_dir, csv_path, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with path, label, patient_id and split (patient-level)."""
    imgs = index_images(Path(data_dir))
    meta = pd.read_csv(csv_path)

    file_col = find_col(meta, "file")
    pid_col = find_col(meta, "patient")
    meta["stem"] = meta[file_col].astype(str).map(lambda x: Path(x).stem)
    meta = meta[["stem", pid_col]].rename(columns={pid_col: "patient_id"}).drop_duplicates("stem")

    df = imgs.merge(meta, on="stem", how="left")
    n_missing = int(df["patient_id"].isna().sum())
    if n_missing:
        print(f"WARNING: {n_missing}/{len(df)} images have no patient_id in the CSV; "
              "they are treated as separate patients (check that file names match).")
        df.loc[df["patient_id"].isna(), "patient_id"] = [f"unknown_{i}" for i in range(n_missing)]
    df["patient_id"] = df["patient_id"].astype(str)

    df = split_by_patient(df, seed)

    # Leakage check: no patient may appear in more than one split
    n_splits_per_patient = df.groupby("patient_id")["split"].nunique()
    assert (n_splits_per_patient == 1).all(), "Some patients appear in more than one split!"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Folder with one subfolder per disease")
    ap.add_argument("--csv", required=True, help="Metadata CSV file")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = build_manifest(args.data_dir, args.csv, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["path", "label", "patient_id", "split"]].to_csv(out, index=False)

    print("\nImages per class and split:")
    print(pd.crosstab(df["label"], df["split"]))
    print("\nPatients per split:")
    print(df.groupby("split")["patient_id"].nunique())
    print(f"\nManifest saved to {out}")


if __name__ == "__main__":
    main()
