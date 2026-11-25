"""Bước 6: 5-fold CV theo site (site không trùng giữa train/val), cân bằng tỉ lệ biến cố.

Ví dụ:
    python AttentionDeepMIL/make_splits.py --labels data/TCGA-LUAD/labels/os.csv \
        --out_dir data/TCGA-LUAD/splits
Output: fold_{k}.csv với cột case_id, split (train/val)
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    n_sites = df["site"].nunique()
    if n_sites < args.k:
        raise SystemExit(f"Chỉ có {n_sites} site, không đủ cho {args.k} fold theo site")

    os.makedirs(args.out_dir, exist_ok=True)
    skf = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    for k, (tr, va) in enumerate(skf.split(df, y=df["event"], groups=df["site"])):
        tr_df, va_df = df.iloc[tr], df.iloc[va]
        assert not set(tr_df["site"]) & set(va_df["site"]), "site bị trùng giữa train/val"
        out = pd.concat([tr_df[["case_id"]].assign(split="train"),
                         va_df[["case_id"]].assign(split="val")])
        out.to_csv(os.path.join(args.out_dir, f"fold_{k}.csv"), index=False)
        print(f"fold {k}: train {len(tr_df)} ({tr_df['event'].sum()} biến cố, "
              f"{tr_df['site'].nunique()} site) | val {len(va_df)} "
              f"({va_df['event'].sum()} biến cố, {va_df['site'].nunique()} site)")


if __name__ == "__main__":
    main()
