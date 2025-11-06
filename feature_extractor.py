#!/usr/bin/env python3
# (see docstring at top in previous message)

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PREFIXES = [
    "PS1","PS2","PS3","PS4","PS5","PS6",
    "TS1","TS2","TS3","TS4",
    "FS1","FS2",
    "VS1",
    "CE","CP","EPS1","SE",
]

def list_columns(csv_path: Path) -> list:
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip()
    return [c.strip() for c in header.split(',')]

def find_group_columns(all_cols, prefix: str):
    pfx = prefix.lower()
    out = []
    for c in all_cols:
        cl = c.lower()
        if cl == pfx or cl.startswith(pfx + "_"):
            out.append(c)
    return out

def stats_for_block(df: pd.DataFrame, base: str) -> pd.DataFrame:
    X = df.apply(pd.to_numeric, errors='coerce')
    eps = 1e-12
    feats = {}
    feats[f"{base}_mean"]   = X.mean(axis=1).astype('float32')
    feats[f"{base}_std"]    = X.std(axis=1).astype('float32')
    feats[f"{base}_min"]    = X.min(axis=1).astype('float32')
    feats[f"{base}_max"]    = X.max(axis=1).astype('float32')
    feats[f"{base}_median"] = X.median(axis=1).astype('float32')
    feats[f"{base}_q25"]    = X.quantile(0.25, axis=1).astype('float32')
    feats[f"{base}_q75"]    = X.quantile(0.75, axis=1).astype('float32')
    feats[f"{base}_rms"]    = np.sqrt((X.pow(2)).mean(axis=1) + eps).astype('float32')
    feats[f"{base}_energy"] = X.pow(2).sum(axis=1).astype('float32')
    feats[f"{base}_skew"]   = X.skew(axis=1).astype('float32')
    feats[f"{base}_kurt"]   = X.kurt(axis=1).astype('float32')
    return pd.DataFrame(feats)

def build_label_from_profile(csv_path: Path, all_cols):
    prof_cols = [c for c in all_cols if c.lower().startswith("profile")]
    if not prof_cols:
        raise RuntimeError("No 'profile_*' columns found; cannot derive weak labels.")
    prof = pd.read_csv(csv_path, usecols=prof_cols, dtype='float32')
    prof_rounded = prof.round(6).astype(str)
    keys = prof_rounded.apply(lambda r: '|'.join(r.values.tolist()), axis=1)
    mode_key = keys.value_counts().idxmax()
    is_fault = (keys != mode_key).astype('int8')
    return is_fault

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/processed/uci_hydraulic_features.csv")
    args = ap.parse_args()

    src = Path(args.csv)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)

    all_cols = list_columns(src)
    print(f"[load] header OK: {len(all_cols)} columns")

    # Label
    try:
        is_fault = build_label_from_profile(src, all_cols)
        print(f"[label] derived (normal={int((is_fault==0).sum())}, fault={int((is_fault==1).sum())})")
    except Exception as e:
        print(f"[warn] label derivation failed: {e}")
        is_fault = None

    out_blocks = []
    n_rows = None
    for p in PREFIXES:
        cols = find_group_columns(all_cols, p)
        if not cols:
            continue
        df = pd.read_csv(src, usecols=cols, dtype='float32', low_memory=False)
        if n_rows is None:
            n_rows = len(df)
        base = p.lower()
        if df.shape[1] == 1:
            tmp = pd.DataFrame({f"{base}_mean": pd.to_numeric(df.iloc[:,0], errors='coerce').astype('float32')})
            out_blocks.append(tmp)
        else:
            out_blocks.append(stats_for_block(df, base))
        print(f"[feat] {p}: in_cols={df.shape[1]} -> out_cols={out_blocks[-1].shape[1]}")

    if not out_blocks:
        raise RuntimeError("No features generated; check prefixes and input header.")

    feats = pd.concat(out_blocks, axis=1)
    feats.insert(0, "row_id", range(len(feats)))
    if is_fault is not None:
        feats["is_fault"] = is_fault.values

    feats.to_csv(dst, index=False)
    print(f"[save] {dst} rows={feats.shape[0]} cols={feats.shape[1]}")

if __name__ == "__main__":
    main()
