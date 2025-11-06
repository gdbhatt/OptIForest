#!/usr/bin/env python3
"""
prep_uci_for_optiforest.py

Create weak labels for UCI Hydraulic features and emit an OptIForest-ready CSV.

Steps
- Load features (expects columns like ce_mean, cp_mean, eps1_mean, se_mean + sensor stats)
- Cluster the 4 settings features (KMeans)
- Label rule:
    * 'minority_all'  -> largest cluster = 0 (normal), all others = 1 (fault)
    * 'smallest_only' -> smallest cluster = 1 (fault), all others = 0 (normal)
- Save:
    1) labeled CSV      -> {--out_labeled}
    2) optiforest CSV   -> {--out_optiforest} (label column named 'label' and as the LAST column)

Usage
python prep_uci_for_optiforest.py \
  --in data/processed/uci_hydraulic_features.csv \
  --out_labeled data/processed/uci_hydraulic_features_labeled.csv \
  --out_optiforest data/uci_hydraulic_optiforest.csv \
  --k 3 --rule minority_all --seed 42
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SETTINGS_COLS_CANDIDATES = ["ce_mean", "cp_mean", "eps1_mean", "se_mean"]
DROP_META_LIKE = {"row_id", "id", "ts", "time", "timestamp"}

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # fill per-column median for any NaNs
    return out.fillna(out.median(numeric_only=True))

def make_labels_from_settings(df: pd.DataFrame, k: int, rule: str, seed: int) -> pd.Series:
    missing = [c for c in SETTINGS_COLS_CANDIDATES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing settings columns {missing}. "
                         f"Columns present: {sorted(df.columns.tolist())[:20]} ...")

    Z = df[SETTINGS_COLS_CANDIDATES].copy()
    Z = _coerce_numeric(Z)
    Zs = StandardScaler().fit_transform(Z.values)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    clusters = km.fit_predict(Zs)

    counts = pd.Series(clusters).value_counts().sort_values(ascending=False)
    largest = counts.index[0]
    smallest = counts.index[-1]

    if rule == "minority_all":
        y = (clusters != largest).astype(np.int8)  # largest => 0, others => 1
    elif rule == "smallest_only":
        y = (clusters == smallest).astype(np.int8) # only smallest => 1
    else:
        raise ValueError("rule must be one of: 'minority_all', 'smallest_only'")

    print(f"[cluster] counts={counts.to_dict()} rule={rule} "
          f"fault_rate={float(y.mean()):.4f} (k={k}, seed={seed})")
    return pd.Series(y, name="is_fault")

def write_optiforest_csv(df_labeled: pd.DataFrame, out_path: Path, label_col: str = "is_fault"):
    # Drop obvious non-feature meta
    drop_cols = [c for c in df_labeled.columns if c.lower() in DROP_META_LIKE]
    feats = df_labeled.drop(columns=drop_cols, errors="ignore").copy()

    # Ensure numeric & no NaNs
    feats = _coerce_numeric(feats)

    if label_col not in feats.columns:
        raise ValueError(f"'{label_col}' not found in columns.")

    # Move label to last and rename to 'label'
    y = (feats.pop(label_col) > 0).astype(int).rename("label")
    out_df = pd.concat([feats, y], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[save] OptIForest CSV: {out_path} shape={out_df.shape} (label last)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True,
                   help="Input features CSV (e.g., data/processed/uci_hydraulic_features.csv)")
    p.add_argument("--out_labeled", default="data/processed/uci_hydraulic_features_labeled.csv")
    p.add_argument("--out_optiforest", default="data/uci_hydraulic_optiforest.csv")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--rule", choices=["minority_all", "smallest_only"], default="minority_all")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    inp = Path(args.inp)
    out_labeled = Path(args.out_labeled)
    out_optiforest = Path(args.out_optiforest)

    df = pd.read_csv(inp)
    y = make_labels_from_settings(df, k=args.k, rule=args.rule, seed=args.seed)

    df_lab = df.copy()
    df_lab["is_fault"] = y.values
    out_labeled.parent.mkdir(parents=True, exist_ok=True)
    df_lab.to_csv(out_labeled, index=False)
    print(f"[save] Labeled CSV: {out_labeled} shape={df_lab.shape}")

    write_optiforest_csv(df_lab, out_path=out_optiforest, label_col="is_fault")

if __name__ == "__main__":
    main()
