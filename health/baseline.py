# baseline.py
"""
Per-sensor Normal baselines for BearingDX (time-domain).
Stores percentile bands per feature under by_sensor: {"DE": {...}, "FE": {...}}.
Fall back rules (used in bk1.validate_signal_quality):
- If sensor baseline exists: use its percentile bands
- Else if BA (or missing): use amplitude-invariant features only with broad sanity bands
- Else: fallback to fixed DEFAULT_SIGNAL_THRESHOLDS
"""

import os, json
import numpy as np
from typing import List, Dict, Any

FEATURE_KEYS = [
    "rms","peak","crest_factor","kurtosis","skewness",
    "form_factor","impulse_factor","margin_factor"
]
INVARIANT_KEYS = ["crest_factor","kurtosis","skewness","form_factor","impulse_factor","margin_factor"]

def _collect_feature_matrix(segments: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    X = {k: [] for k in FEATURE_KEYS}
    for s in segments:
        for k in FEATURE_KEYS:
            X[k].append(float(s.get(k, 0.0)))
    return {k: np.array(v, dtype=float) for k, v in X.items()}

def _stats_block(arr: np.ndarray, p_low=5, p_high=95):
    if arr.size == 0 or not np.isfinite(arr).any():
        return None
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p_low": float(np.percentile(arr, p_low)),
        "p_high": float(np.percentile(arr, p_high)),
    }

def build_baseline_from_segments(segments: List[Dict[str, Any]], p_low=5, p_high=95) -> Dict[str, Any]:
    X = _collect_feature_matrix(segments)
    per_feature = {}
    for k, arr in X.items():
        sb = _stats_block(arr, p_low, p_high)
        if sb: per_feature[k] = sb
    return {
        "per_feature": per_feature,
        "info": {
            "segments_used": int(len(segments)),
            "percentile_band": [p_low, p_high],
        }
    }

def build_sensor_baseline_from_files(sensor_key: str, file_paths: List[str],
                                     sampling_rate: int = 12000,
                                     p_low=5, p_high=95) -> Dict[str, Any]:
    from bdxio.vp1 import parse_vibration_file  # lazy import to avoid circular
    all_segments: List[Dict[str, Any]] = []
    for fp in file_paths:
        res = parse_vibration_file(fp, sensor_key=sensor_key, sampling_rate=sampling_rate)
        if res.get("success") and res.get("segments"):
            all_segments.extend(res["segments"])
    b = build_baseline_from_segments(all_segments, p_low=p_low, p_high=p_high)
    b["info"]["sensor_key"] = sensor_key
    return b

def merge_into_by_sensor(existing: Dict[str, Any], sensor_key: str, baseline: Dict[str, Any]) -> Dict[str, Any]:
    existing = existing or {}
    existing.setdefault("by_sensor", {})
    existing["by_sensor"][sensor_key] = baseline
    return existing

def save_baseline(baseline_root: Dict[str, Any], path: str = "models/baseline.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(baseline_root, f, indent=2)

def load_baseline(path: str = "models/baseline.json") -> Dict[str, Any]:
    if not os.path.exists(path): return {}
    with open(path, "r") as f: return json.load(f)

def get_sensor_baseline(root: Dict[str, Any], sensor_key: str) -> Dict[str, Any]:
    return (root or {}).get("by_sensor", {}).get(sensor_key, {})

def score_td_health(parsed_result, sensor_baseline, min_features=3):
    """
    Compute a simple TD health score based on Normal baseline bands.
    Returns:
      {"health_score": float[0..1], "status": "Healthy|Degraded", "feature_outliers": {feature: frac_outside}}
    """
    import numpy as np

    # no segments => degraded
    if not parsed_result or not parsed_result.get("segments"):
        return {"health_score": 0.0, "status": "Degraded", "feature_outliers": {}}

    segs = parsed_result["segments"]

    # per-feature p5/p95 bands
    bands = (sensor_baseline or {}).get("per_feature", {})

    # use only bands that exist in the baseline
    candidates = ["rms","peak","crest_factor","kurtosis","skewness","impulse_factor","margin_factor","form_factor"]
    features = [f for f in candidates if f in bands]

    # if baseline is thin, return neutral-ish health
    if len(features) < min_features:
        return {"health_score": 0.5, "status": "Healthy", "feature_outliers": {}}

    inside_ratios = []
    outliers = {}
    for f in features:
        p5  = float(bands[f].get("p5",  -1e9))
        p95 = float(bands[f].get("p95",  1e9))
        vals = np.array([float(s.get(f, 0.0)) for s in segs], dtype=float)
        inside = np.mean((vals >= p5) & (vals <= p95))
        inside_ratios.append(float(inside))
        outliers[f] = float(1.0 - inside)

    score = float(np.mean(inside_ratios)) if inside_ratios else 0.5
    status = "Healthy" if score >= 0.70 else "Degraded"
    return {"health_score": score, "status": status, "feature_outliers": outliers}


