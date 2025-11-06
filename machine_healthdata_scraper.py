#!/usr/bin/env python3
"""
scrape_machine_health.py (v1.1)
--------------------------------
Updates:
- Uses sep=r"\\s+" instead of deprecated delim_whitespace
- Skips textual docs when parsing (.txt with 'description'/'documentation' in name)
- Horizontally assembles multiple numeric .txt signals (e.g., UCI Hydraulic PS1.txt, TS1.txt, ...)
- More robust parsing (fallback encodings), optional dtype downcast for memory
"""

import argparse
import io
import json
import re
import sys
import zipfile
import hashlib
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
import pandas as pd
import yaml
from tqdm import tqdm
import urllib.robotparser as robotparser

# -----------------------
# Utility helpers
# -----------------------
def mkparent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", (s or "").strip()).lower()

def session_with_retries():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    s.headers.update({"User-Agent": "MachineHealthScraper/1.1 (+research use)"})
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def allowed_by_robots(url: str, user_agent="*") -> bool:
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def normalize_df(df: pd.DataFrame, rename_map=None, label_map=None) -> pd.DataFrame:
    if rename_map:
        df = df.rename(columns=rename_map)
    if label_map:
        for col, mapping in label_map.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x).strip() if pd.notnull(x) else x)
                df[col] = df[col].map({str(k): v for k, v in mapping.items()}).fillna(df[col])
                try:
                    df[col] = df[col].astype(float)
                    if set(pd.unique(df[col].dropna())) <= {0.0, 1.0}:
                        df[col] = df[col].astype("int64")
                except Exception:
                    pass
    df.columns = [str(c).strip() for c in df.columns]
    return df

def save_df(df: pd.DataFrame, path: Path):
    mkparent(path)
    df.to_csv(path, index=False)
    print(f"[csv] {path} rows={len(df)} cols={len(df.columns)}")

def fetch_binary(sess: requests.Session, url: str) -> bytes:
    with sess.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        buf = io.BytesIO()
        if total > 0:
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc="download")
        else:
            pbar = None
        for chunk in r.iter_content(chunk_size=1024 * 64):
            if chunk:
                buf.write(chunk)
                if pbar:
                    pbar.update(len(chunk))
        if pbar:
            pbar.close()
        return buf.getvalue()

# -----------------------
# Handlers
# -----------------------
def handle_csv(cfg, sess, roots):
    url = cfg["url"]
    if not allowed_by_robots(url):
        print(f"[robots] disallowed: {url}")
        return None
    raw = fetch_binary(sess, url)
    raw_path = Path(roots["data_dir"]) / cfg.get("save_as", f"{slug(cfg['name'])}.csv")
    mkparent(raw_path)
    raw_path.write_bytes(raw)
    print(f"[saved] {raw_path} ({len(raw)} bytes, sha256={sha256_bytes(raw)[:12]})")

    try:
        df = pd.read_csv(io.BytesIO(raw))
        df = normalize_df(df, cfg.get("rename_map"), cfg.get("label_map"))
        save_df(df, Path(roots["out_dir"]) / raw_path.with_suffix(".csv").name)
        return df
    except Exception as e:
        print(f"[warn] CSV parse failed for {cfg['name']}: {e}")
        return None

def _is_textual_doc(name: str) -> bool:
    n = name.lower()
    return ("description" in n) or ("documentation" in n) or ("readme" in n) or ("profile" in n)

def _try_read_txt_numeric(path: Path, downcast=True) -> pd.DataFrame | None:
    """Try to parse whitespace-delimited numeric .txt into a DataFrame.
       Returns DataFrame or None if not numeric.
    """
    try:
        # First quick sniff: read a small chunk and see if it's mostly digits/whitespace/.-+eE
        with open(path, "rb") as f:
            chunk = f.read(2048)
        if chunk:
            sample = chunk.decode("utf-8", errors="ignore")
            score = sum(ch.isdigit() or ch.isspace() or ch in ".-,+eE" for ch in sample) / max(1, len(sample))
            if score < 0.6:  # likely not purely numeric table
                return None
        # Parse as whitespace-delimited; headerless numeric
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            engine="python",
            dtype="float32" if downcast else None,
            na_values=["NaN", "nan", "", "?"],
        )
        return df
    except UnicodeDecodeError:
        # Try latin-1 fallback
        try:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                header=None,
                engine="python",
                encoding="latin-1",
                dtype="float32" if downcast else None,
                na_values=["NaN", "nan", "", "?"],
            )
            return df
        except Exception:
            return None
    except Exception:
        return None

def handle_zip(cfg, sess, roots):
    url = cfg["url"]
    if not allowed_by_robots(url):
        print(f"[robots] disallowed: {url}")
        return None
    raw = fetch_binary(sess, url)
    raw_path = Path(roots["data_dir"]) / cfg.get("save_as", f"{slug(cfg['name'])}.zip")
    mkparent(raw_path)
    raw_path.write_bytes(raw)
    print(f"[saved] {raw_path} ({len(raw)} bytes, sha256={sha256_bytes(raw)[:12]})")

    kept_exts = set([e.lower() for e in cfg.get("keep_extensions", [".csv", ".txt"])])
    out_folder = raw_path.with_suffix("")
    frames_for_txt = {}   # stem -> DataFrame of numeric columns
    csv_frames = []       # list of parsed CSV frames

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                ext = Path(member.filename).suffix.lower()
                target = out_folder / Path(member.filename).name
                mkparent(target)
                with zf.open(member) as fsrc, open(target, "wb") as fdst:
                    data = fsrc.read()
                    fdst.write(data)
                print(f"[extract] {target} ({len(data)} bytes)")

                if ext not in kept_exts:
                    continue

                # Skip obvious docs
                if ext == ".txt" and _is_textual_doc(target.name):
                    continue

                # Attempt parsing
                try:
                    if ext == ".csv":
                        df = pd.read_csv(target)
                        df = normalize_df(df, cfg.get("rename_map"), cfg.get("label_map"))
                        csv_frames.append(df)
                    elif ext == ".txt":
                        df = _try_read_txt_numeric(target)
                        if df is not None:
                            # If it's a single column, name it by stem; otherwise prefix
                            stem = Path(target).stem
                            if df.shape[1] == 1:
                                df.columns = [stem]
                            else:
                                df.columns = [f"{stem}_{i}" for i in range(df.shape[1])]
                            frames_for_txt[stem] = df
                except Exception as e:
                    print(f"[warn] parse fail {target}: {e}")
    except zipfile.BadZipFile as e:
        print(f"[error] Bad ZIP: {e}")
        return None

    # Assemble numeric TXT frames horizontally if we have any
    if frames_for_txt:
        # Align on row index
        ordered = [frames_for_txt[k] for k in sorted(frames_for_txt.keys())]
        try:
            assembled = pd.concat(ordered, axis=1)
            assembled = normalize_df(assembled, cfg.get("rename_map"), cfg.get("label_map"))
            save_df(assembled, Path(roots["out_dir"]) / f"{slug(cfg['name'])}_txt_assembled.csv")
        except Exception as e:
            print(f"[warn] horizontal assembly failed: {e}")

    # Merge CSV frames vertically (if present)
    if csv_frames:
        try:
            merged_csv = pd.concat(csv_frames, ignore_index=True)
            save_df(merged_csv, Path(roots["out_dir"]) / f"{slug(cfg['name'])}.csv")
            return merged_csv
        except Exception as e:
            print(f"[warn] CSV merge failed: {e}")

    # If nothing to return, but we assembled TXT, optionally return assembled
    if frames_for_txt:
        try:
            return assembled
        except NameError:
            return None

    return None

def handle_html_table(cfg, sess, roots):
    url = cfg["url"]
    if not allowed_by_robots(url):
        print(f"[robots] disallowed: {url}")
        return None
    html = fetch_binary(sess, url).decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    match = cfg.get("match", {"by": "index", "index": 0})

    table = None
    by = match.get("by", "index")
    if by == "id":
        table = soup.find("table", id=match.get("value"))
    elif by == "class":
        table = soup.find("table", class_=match.get("value"))
    elif by == "caption":
        for t in soup.find_all("table"):
            cap = t.find("caption")
            if cap and match.get("value").strip().lower() in cap.get_text(strip=True).lower():
                table = t; break
    else:  # index
        tables = soup.find_all("table")
        idx = int(match.get("index", 0))
        if 0 <= idx < len(tables):
            table = tables[idx]

    if table is None:
        print(f"[warn] no matching table on {url}")
        return None

    dfs = pd.read_html(str(table))
    if not dfs:
        print(f"[warn] pandas.read_html found nothing")
        return None
    df = dfs[0]
    df = normalize_df(df, cfg.get("rename_map"), cfg.get("label_map"))
    out_path = Path(roots["out_dir"]) / Path(cfg.get("save_as", f"{slug(cfg['name'])}.csv")).name
    save_df(df, out_path)
    return df

def handle_api_json(cfg, sess, roots):
    url = cfg["url"]
    if not allowed_by_robots(url):
        print(f"[robots] disallowed: {url}")
        return None
    r = sess.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    json_path = Path(roots["data_dir"]) / cfg.get("save_as", f"{slug(cfg['name'])}.json")
    mkparent(json_path)
    json_path.write_text(json.dumps(data, indent=2))
    print(f"[saved] {json_path}")

    j2c = cfg.get("json_to_csv")
    if j2c:
        record_path = j2c.get("record_path")
        meta_fields = j2c.get("meta_fields", [])
        def extract_records(obj):
            recs = obj.get(record_path, []) if record_path else obj
            if not isinstance(recs, list):
                return []
            meta = {k: obj.get(k) for k in meta_fields if k in obj}
            for r in recs:
                if isinstance(r, dict):
                    yield {**meta, **r}

        rows = list(extract_records(data))
        if not rows:
            print("[warn] no records to tabulate from JSON")
            return None
        df = pd.DataFrame(rows)
        df = normalize_df(df, j2c.get("rename_map"), j2c.get("label_map"))
        out_csv = j2c.get("out_csv", f"{slug(cfg['name'])}.csv")
        save_df(df, Path(roots["out_dir"]) / Path(out_csv).name)
        return df
    return None

def handle_page_links(cfg, sess, roots):
    url = cfg["url"]
    allow_ext = [e.lower() for e in cfg.get("allow_ext", [".zip", ".csv"])]
    include = [s.lower() for s in cfg.get("include", [])]
    max_files = int(cfg.get("max_files", 5))
    save_dir = Path(roots["data_dir"]) / (cfg.get("save_dir") or slug(cfg["name"]))
    mkparent(save_dir / "x")

    if not allowed_by_robots(url):
        print(f"[robots] disallowed: {url}")
        return None

    html = fetch_binary(sess, url).decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(" ", strip=True) or ""
        full = urljoin(url, href)
        ext = Path(full.split("?")[0]).suffix.lower()
        if ext not in allow_ext:
            continue
        hay = (href + " " + text).lower()
        if include and not any(s in hay for s in include):
            continue
        links.append(full)

    seen = set()
    picked = []
    for L in links:
        if L in seen:
            continue
        seen.add(L)
        picked.append(L)
        if len(picked) >= max_files:
            break

    print(f"[page] matched {len(picked)} file link(s)")
    for L in picked:
        try:
            print(f"[link] {L}")
            blob = fetch_binary(sess, L)
            fname = Path(L.split('?')[0]).name or f"file_{len(picked)}.bin"
            dst = save_dir / fname
            dst.write_bytes(blob)
            print(f"[saved] {dst} ({len(blob)} bytes)")
            if cfg.get("extract", True) and dst.suffix.lower() == ".zip":
                out_folder = save_dir / dst.stem
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    for member in zf.infolist():
                        if member.is_dir():
                            continue
                        target = out_folder / Path(member.filename).name
                        mkparent(target)
                        with zf.open(member) as fsrc, open(target, "wb") as fdst:
                            fdst.write(fsrc.read())
                        print(f"[extract] {target}")
        except Exception as e:
            print(f"[warn] failed {L}: {e}")
    return None

HANDLERS = {
    "csv": handle_csv,
    "zip": handle_zip,
    "html_table": handle_html_table,
    "api_json": handle_api_json,
    "page_links": handle_page_links,
}

def consolidate(dfs, out_dir: Path):
    wanted = ["temperature", "vibration", "pressure", "runtime_hours", "humidity", "is_fault"]
    parts = []
    for name, df in dfs.items():
        try:
            cols = [c for c in wanted if c in df.columns]
            if cols:
                tmp = df[cols].copy()
                tmp["__source__"] = name
                parts.append(tmp)
        except Exception as e:
            print(f"[warn] consolidate skipped {name}: {e}")
    if parts:
        merged = pd.concat(parts, ignore_index=True)
        save_df(merged, out_dir / "machine_health_all.csv")
    else:
        print("[info] no overlapping columns to consolidate (skipping)")

def main():
    ap = argparse.ArgumentParser(description="Scrape machine-health data from configured websites.")
    ap.add_argument("--config", "-c", required=True, help="Path to sources.yml")
    args = ap.parse_args()

    try:
        cfg = yaml.safe_load(open(args.config, "r"))
    except Exception as e:
        print(f"[fatal] cannot read config: {e}")
        sys.exit(2)

    data_dir = Path(cfg.get("data_dir", "data/raw"))
    out_dir  = Path(cfg.get("out_dir",  "data/processed"))
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = session_with_retries()
    collected = {}

    for src in cfg.get("sources", []):
        name = src.get("name") or src.get("url")
        t = src.get("type")
        print(f"\n=== [{name}] type={t} ===")
        handler = HANDLERS.get(t)
        if not handler:
            print(f"[skip] unknown type: {t}")
            continue
        try:
            df = handler(src, sess, {"data_dir": data_dir, "out_dir": out_dir})
            if isinstance(df, pd.DataFrame):
                collected[name] = df
        except requests.HTTPError as e:
            print(f"[HTTP error] {e}")
        except KeyboardInterrupt:
            print("\n[abort] interrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}")

    consolidate(collected, out_dir)
    print("\nDone.")

if __name__ == "__main__":
    main()
