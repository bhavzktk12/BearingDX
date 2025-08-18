# dx1.py
"""
BearingDX - Phase-1 Diagnostic Module
- Time-domain features & file-level indicators
- Rule-based fault location inference (Normal/IR/OR/Ball)
- Intelligent confidence score (rule strength × segment quality × sensor weight × top-2 separation)
- Phase 1.2: Lightweight spectral indicators (auto envelope band, sidebands)
"""
import os, json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import joblib
from scipy.signal import hilbert, butter, filtfilt
from numpy.fft import rfft, irfft



from bk1 import get_fault_frequencies

# ======== Paths / constants ========
TD_CAL_MODEL_PATH = "models/td_clf_calibrated.joblib"
TD_LABELS_IDX = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}

TD_THRESHOLDS_PATH = "models/td_thresholds.json"


# Candidate envelope bands to try (auto-select the best per file)

# Type alias for envelope-band parameter


# ========= File-level indicators (TD) =========
def compute_diagnostic_indicators(parsed_result: Dict) -> Dict[str, Any]:
    if not parsed_result.get("success") or not parsed_result.get("segments"):
        return {"error": "Invalid input or processing failure"}

    segments = parsed_result["segments"]
    n = len(segments)
    rms = np.array([s["rms"] for s in segments])
    peaks = np.array([s["peak"] for s in segments])
    kurt = np.array([s["kurtosis"] for s in segments])
    cf = np.array([s["crest_factor"] for s in segments])
    skew = np.array([s["skewness"] for s in segments])

    diagnostics = {"tier1": {}, "tier2": {}, "tier3": {}, "tier4": {}, "tier5": {}, "warnings": []}

    t = np.arange(n)
    slope = np.polyfit(t, rms, 1)[0]
    diagnostics["tier1"]["rms_trend_slope"] = float(slope)

    kb = 100 * np.sum(kurt > 5) / n
    diagnostics["tier1"]["kurtosis_burst_index"] = float(kb)

    p2r = peaks / np.clip(rms, 1e-12, None)
    diagnostics["tier1"]["peak_to_rms_ratio_sd"] = float(np.std(p2r))

    sev = peaks * kurt
    diagnostics["tier1"]["top_3_severity"] = float(np.mean(np.sort(sev)[-3:]) if n >= 3 else np.max(sev))

    med_peak = np.median(peaks)
    diagnostics["tier2"]["transient_event_density"] = float(100 * np.sum(peaks > 5 * med_peak) / n)
    diagnostics["tier2"]["crest_factor_stability"] = float(1 / (np.std(cf) + 1e-3))
    kflag = kurt > 4.5
    diagnostics["tier2"]["impulse_persistence"] = float(max(_streaks(kflag), default=0) / n)

    quad = np.polyfit(t, rms, 2)[0] if n > 2 else 0.0
    diagnostics["tier3"]["degradation_acceleration"] = float(quad)
    diagnostics["tier3"]["dynamic_range_collapse"] = float(np.ptp(peaks) / np.maximum(np.mean(rms), 1e-12))
    pos = 100 * np.sum(skew > 0.5) / n
    neg = 100 * np.sum(skew < -0.5) / n
    diagnostics["tier3"]["skewness_polarity_shift"] = float(pos - neg)

    covar = np.cov(kurt, rms)[0, 1]
    diagnostics["tier4"]["kurtosis_rms_covariance"] = float(covar)
    low_rms = rms < 0.1 * np.median(rms)
    high_peak = peaks > 3 * rms
    noise_idx = 100 * np.sum(low_rms & high_peak) / n
    diagnostics["tier4"]["noise_contamination_index"] = float(noise_idx)
    res = rms - (slope * t + np.mean(rms))
    r2 = 1 - np.var(res) / np.maximum(np.var(rms), 1e-12)
    diagnostics["tier4"]["trend_inconsistency_flag"] = int(r2 < 0.7)

    prog = (np.max(peaks) * kb) / (abs(slope) + 0.01)
    diagnostics["tier5"]["fault_progression_score"] = float(prog)
    rms_sk = np.mean((rms - np.mean(rms)) ** 3) / (np.std(rms) + 1e-12) ** 3
    peak_ku = np.mean((peaks - np.mean(peaks)) ** 4) / (np.std(peaks) + 1e-12) ** 4 - 3
    diagnostics["tier5"]["harmonic_distortion_indicator"] = float(rms_sk * peak_ku)

    if kb > 20:
        diagnostics["warnings"].append(f"High kurtosis bursts ({kb:.1f}% segments > 5)")
    if noise_idx > 15:
        diagnostics["warnings"].append(f"Potential noise contamination ({noise_idx:.1f}%)")
    if prog > 5000:
        diagnostics["warnings"].append(f"Severe fault progression (score={prog:.0f})")

    return diagnostics


def _streaks(arr: np.ndarray) -> List[int]:
    streaks = []
    cur = 0
    for v in arr:
        if v:
            cur += 1
        elif cur > 0:
            streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    return streaks

# ========= Segment feature utils =========

def print_segment_features(segment: Dict[str, Any], segment_id: int = 0):
    print(f"\n🕒 Time-Domain Features (Segment {segment_id}):")
    print("=" * 60)
    for k in ("rms", "peak", "crest_factor", "kurtosis", "skewness", "form_factor", "impulse_factor", "margin_factor"):
        print(f"{k:<16} {segment[k]:>10.4f}")
    print(f"\n🧪 Quality Score: {segment['quality_score']:.2f} — {'✅ Valid' if segment['is_valid'] else '⚠️ Low Quality'}")


def summarize_segment_features(segments: List[Dict[str, Any]], top_n: int = 3):
    df = pd.DataFrame(segments)
    if df.empty:
        print("No segments available for summary.")
        return
    feature_cols = ["rms", "peak", "crest_factor", "kurtosis", "skewness", "form_factor", "impulse_factor", "margin_factor"]
    print("\n📊 Segment Feature Summary (All Segments):")
    print("=" * 65)
    print(f"{'Feature':<16} {'Mean':>10} {'Std':>10} {'Max':>10}")
    print("-" * 65)
    for f in feature_cols:
        vals = df[f]
        print(f"{f:<16} {vals.mean():>10.4f} {vals.std():>10.4f} {vals.max():>10.4f}")
    df["severity"] = df["peak"] * df["kurtosis"]
    tops = df.sort_values(by="severity", ascending=False).head(top_n)
    for _, row in tops.iterrows():
        print_segment_features(row.to_dict(), segment_id=int(row["segment_id"]))


# ====== Phase 1.2: Spectral utilities (lightweight, safe defaults) ======

# ========= TD ML (calibrated) =========
def _file_level_vector_for_ml(parsed_result: Dict[str, Any]) -> np.ndarray:
    """Must mirror td_train’s vectorization logic as closely as possible."""

    segs = parsed_result.get("segments", [])
    if not segs:
        return np.zeros((1, 1), dtype=float)
    df = pd.DataFrame(segs)
    vec: List[float] = []
    keys = ["rms", "peak", "crest_factor", "kurtosis", "skewness", "form_factor", "impulse_factor", "margin_factor"]
    for k in keys:
        if k in df:
            v = df[k].to_numpy(dtype=float)
            vec += [float(np.mean(v)), float(np.std(v)), float(np.percentile(v, 90))]
        else:
            vec += [0.0, 0.0, 0.0]

    # threshold proportions
    if {"kurtosis", "crest_factor", "impulse_factor"}.issubset(df.columns):
        v_k = (df["kurtosis"].to_numpy() > 5.0).mean()
        v_c = (df["crest_factor"].to_numpy() > 7.0).mean()
        v_i = (df["impulse_factor"].to_numpy() > 12.0).mean()
    else:
        v_k = v_c = v_i = 0.0
    vec += [float(v_k), float(v_c), float(v_i)]

    # file-level indicators
    di = compute_diagnostic_indicators(parsed_result)
    take = [
        ("tier1", "rms_trend_slope"),
        ("tier1", "kurtosis_burst_index"),
        ("tier1", "peak_to_rms_ratio_sd"),
        ("tier1", "top_3_severity"),
        ("tier2", "transient_event_density"),
        ("tier2", "impulse_persistence"),
        ("tier3", "degradation_acceleration"),
        ("tier3", "dynamic_range_collapse"),
        ("tier3", "skewness_polarity_shift"),
        ("tier4", "kurtosis_rms_covariance"),
        ("tier4", "noise_contamination_index"),
        ("tier4", "trend_inconsistency_flag"),
        ("tier5", "fault_progression_score"),
        ("tier5", "harmonic_distortion_indicator"),
    ]
    for tier, key in take:
        vec.append(float(di[tier].get(key, 0.0)))

    return np.asarray(vec, dtype=float).reshape(1, -1)


def predict_fault_location_ml(parsed_result: Dict[str, Any]) -> Dict[str, Any]:
    """Data-driven prediction using calibrated model (time-domain features only)."""
    if not parsed_result.get("success") or not parsed_result.get("segments"):
        return {"available": False, "reason": "no segments"}
    if not os.path.exists(TD_CAL_MODEL_PATH):
        return {"available": False, "reason": "model not found"}
    bundle = joblib.load(TD_CAL_MODEL_PATH)
    model = bundle["model"]
    x = _file_level_vector_for_ml(parsed_result)
    proba = model.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    return {
        "available": True,
        "label": TD_LABELS_IDX.get(idx, str(idx)),
        "confidence": float(proba[idx]),  # calibrated probability
        "probs": {TD_LABELS_IDX[i]: float(p) for i, p in enumerate(proba)},
    }


# ========= Optional: combined TD+FD vector (for future Phase 1.2 training) =========