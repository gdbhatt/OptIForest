#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature pruner for OptIForest CSVs.
- Selects a subset of statistics per sensor/channel by suffix (e.g., mean, std, ...)
- Drops highly-correlated duplicate columns (abs Pearson >= threshold)
- (Optional) PCA to N components
Keeps `label` as the last column if present.

Usage:
  python reduce_dims.py\
    --in data/uci_hydraulic_optiforest_smallest.csv \
    --out data/uci_hydraulic_optiforest_smallest.pruned.csv \
    --keep mean std min max skew kurt \
    --corr 0.995

Optional PCA (after pruning):
    --pca 64   # produces pc1..pc64 instead of raw features
"""
import argparse
import numpy as np
import pandas as pd

def pretty_mem(df):
    return f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB"

def select_by_suffix(df, suffixes, label_col):
    if not suffixes:
        return df, df.columns.tolist()
    keep = []
    for c in df.columns:
        if c == label_col:
            continue
        # suffix = part after last underscore
        parts = c.split('_')
        suf = parts[-1] if len(parts) > 1 else ''
        if suf in suffixes:
            keep.append(c)
    cols = keep + ([label_col] if label_col in df.columns else [])
    return df[cols], keep

def drop_correlated(df, label_col, thresh=0.995, sample_rows=1200, seed=0):
    # Only on feature columns
    feats = [c for c in df.columns if c != label_col]
    if len(feats) == 0:
        return df, []

    n = len(df)
    if n > sample_rows:
        work = df[feats].sample(n=sample_rows, random_state=seed)
    else:
        work = df[feats]

    # Corr matrix
    corr = work.corr(method='pearson').abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] >= thresh)]

    pruned = [c for c in feats if c not in to_drop]
    cols_final = pruned + ([label_col] if label_col in df.columns else [])
    return df[cols_final], to_drop

def do_pca(df, label_col, n_components):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    feats = [c for c in df.columns if c != label_col]
    X = df[feats].values
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    n_comp = min(n_components, Xs.shape[1])
    pca = PCA(n_components=n_comp, svd_solver='auto', random_state=0)
    Z = pca.fit_transform(Xs)

    cols = [f"pc{i+1}" for i in range(n_comp)]
    out = pd.DataFrame(Z, columns=cols, index=df.index)
    if label_col in df.columns:
        out[label_col] = df[label_col].values
    return out, pca.explained_variance_ratio_.sum()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_csv", required=True)
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--keep", nargs="*", default=[],
                    help="Suffixes to keep per channel (e.g., mean std min max skew kurt). If empty, keep all.")
    ap.add_argument("--corr", type=float, default=0.995,
                    help="Correlation threshold to drop columns (abs Pearson).")
    ap.add_argument("--pca", type=int, default=0, help="If >0, replace features with N PCA components.")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    print(f"[load] shape={df.shape} mem={pretty_mem(df)}")

    # Step 1: suffix selection
    if args.keep:
        df, kept_cols = select_by_suffix(df, set(args.keep), args.label_col)
        print(f"[keep] suffixes={args.keep} -> kept {len(kept_cols)} feature cols")

    # Step 2: correlated drop
    df, dropped = drop_correlated(df, args.label_col, thresh=args.corr)
    print(f"[corr] dropped {len(dropped)} highly-correlated cols at |r|≥{args.corr}")

    # Step 3: optional PCA
    if args.pca > 0:
        df, varexp = do_pca(df, args.label_col, args.pca)
        print(f"[pca] components={args.pca}, variance explained ≈ {varexp*100:.1f}%")

    # Save
    # Ensure label is last (if present)
    if args.label_col in df.columns:
        cols = [c for c in df.columns if c != args.label_col] + [args.label_col]
        df = df[cols]

    df.to_csv(args.out_csv, index=False)
    print(f"[save] {args.out_csv} shape={df.shape} mem={pretty_mem(df)}")

if __name__ == "__main__":
    main()
