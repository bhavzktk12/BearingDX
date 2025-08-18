# health/td_scan.py
"""
Scan the four CWRU 'Normal' files and compute Time-Domain features
using features/td.compute_time_domain_features. Saves per-segment CSV
and prints compact summaries.

Usage (from repo root):
  py -m health.td_normal_scan --normal-dir "<PATH>/CWRU_train/Normal" --sensor DE
"""

from __future__ import annotations
import os, sys, glob
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# --- Ensure project root on sys.path ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Parser + TD features
from bdxio.vp1 import parse_vibration_file, DEFAULT_SAMPLING_RATE  # preferred package path
from features.td import compute_time_domain_features  # borrow TD feature logic here

NORMAL_NAME_PATTERNS = {
    0: "normal_0_*.mat",
    1: "normal_1_*.mat",
    2: "normal_2_*.mat",
    3: "normal_3_*.mat",
}

TD_FEATURE_KEYS = [
    "rms", "peak", "peak_to_peak", "crest_factor", "kurtosis", "skewness",
    "form_factor", "impulse_factor", "margin_factor", "variance", "mean", "std",
]

def find_normal_files(normal_dir: str) -> List[str]:
    """Return up to one file for each load (0..3) in the Normal folder."""
    files: List[str] = []
    for _, pat in NORMAL_NAME_PATTERNS.items():
        matches = sorted(glob.glob(os.path.join(normal_dir, pat)))
        if matches:
            files.append(matches[0])
    if not files:
        files = sorted(glob.glob(os.path.join(normal_dir, "*.mat")))[:4]
    return files

def compute_td_for_segments(parsed: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    """Recompute TD features from raw/normalized signal for every segment."""
    rows: List[Dict[str, Any]] = []
    segs = parsed.get("segments") or []
    for seg in segs:
        sig = np.asarray(seg.get("signal") if "signal" in seg else seg["normalized_signal"], dtype=float)
        feats = compute_time_domain_features(sig)  # ← borrow from features/td
        rows.append({
            "file": file_name,
            "segment_id": int(seg.get("segment_id", len(rows))),
            "sampling_rate": float(seg.get("sampling_rate", DEFAULT_SAMPLING_RATE)),
            "quality_score": float(seg.get("quality_score", 1.0)),
            "is_valid": bool(seg.get("is_valid", True)),
            **{k: float(feats.get(k, 0.0)) for k in TD_FEATURE_KEYS},
        })
    return rows

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    def bands(x: pd.Series) -> pd.Series:
        return pd.Series({
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p5": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x)),
        })
    parts = []
    for col in TD_FEATURE_KEYS:
        if col in df.columns:
            s = bands(df[col]); s.name = col
            parts.append(s)
    return pd.DataFrame(parts)

def per_file_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pylance-safe named aggregation (no MultiIndex, no overload warnings)."""
    if df.empty:
        return pd.DataFrame()
    aggs = {}
    for col in TD_FEATURE_KEYS:
        if col in df.columns:
            aggs[f"{col}_mean"] = (col, "mean")
            aggs[f"{col}_std"]  = (col, "std")
            aggs[f"{col}_max"]  = (col, "max")
    out = df.groupby("file").agg(**aggs).reset_index()
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compute TD features for Normal data (4 loads) using features/td.")
    ap.add_argument("--normal-dir", required=True, help="Path to CWRU Normal folder (contains normal_*.mat)")
    ap.add_argument("--sensor", default="DE", help="Sensor key: DE/FE/BA (default DE)")
    ap.add_argument("--fs", type=int, default=DEFAULT_SAMPLING_RATE, help="Sampling rate (default from parser)")
    ap.add_argument("--save-prefix", default="normal_td", help="Output filename prefix (CSV in reports/)")
    args = ap.parse_args()

    files = find_normal_files(args.normal_dir)
    if not files:
        print(f"No Normal files found in: {args.normal_dir}")
        sys.exit(1)

    print("Normal files to scan:")
    for f in files:
        print(" -", os.path.basename(f))

    all_rows: List[Dict[str, Any]] = []
    for fpath in files:
        parsed = parse_vibration_file(fpath, sensor_key=args.sensor, sampling_rate=args.fs)
        if not parsed.get("success") or not parsed.get("segments"):
            print(f"[skip] Parse failed or no segments: {os.path.basename(fpath)}  →  {parsed.get('errors')}")
            continue
        rows = compute_td_for_segments(parsed, os.path.basename(fpath))
        all_rows.extend(rows)

    if not all_rows:
        print("No segment features computed.")
        sys.exit(2)

    df = pd.DataFrame(all_rows)
    overall = summarize(df)
    perfile = per_file_summary(df)

    # Console preview
    print("\n=== OVERALL TD SUMMARY (Normal) ===")
    print(overall.round(4).to_string() if not overall.empty else "<empty>")

    print("\n=== PER-FILE TD SUMMARY (mean/std/max) ===")
    print(perfile.round(4).to_string(index=False) if not perfile.empty else "<empty>")

    # Save
    os.makedirs("reports", exist_ok=True)
    seg_csv = os.path.join("reports", f"{args.save_prefix}_segments.csv")
    overall_csv = os.path.join("reports", f"{args.save_prefix}_overall.csv")
    perfile_csv = os.path.join("reports", f"{args.save_prefix}_perfile.csv")
    df.to_csv(seg_csv, index=False)
    overall.to_csv(overall_csv)
    perfile.to_csv(perfile_csv, index=False)

    print("\nSaved CSVs:")
    print(" ", seg_csv)
    print(" ", overall_csv)
    print(" ", perfile_csv)

if __name__ == "__main__":
    main()
