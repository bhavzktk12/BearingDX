#td.py : time domain feaures
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import json, os

# ----- TD config & types -----
TD_THRESHOLDS_PATH = "models/td_thresholds.json"
TIME_DOMAIN_RELIABILITY = {"IR": 0.70, "OR": 0.60, "Ball": 0.50}
SENSOR_WEIGHT = {"DE": 1.00, "FE": 0.95, "BA": 0.85}
EnvBand = Optional[Union[Tuple[float, float], List[Tuple[float, float]], str]]

def _load_td_thresholds():
    try:
        if os.path.exists(TD_THRESHOLDS_PATH):
            with open(TD_THRESHOLDS_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"probs_min": {"IR": 0.55, "OR": 0.60, "Ball": 0.65}, "confidence_min": 0.50}

# ----- per-segment TD features -----
def compute_time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    mean = float(np.mean(signal)); std = float(np.std(signal))
    rms = float(np.sqrt(np.mean(np.square(signal)))); peak = float(np.max(np.abs(signal)))
    mean_abs = float(np.mean(np.abs(signal))); mean_root = float(np.mean(np.sqrt(np.abs(signal))))
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.ptp(signal)),
        "variance": float(np.var(signal)),
        "skewness": float(np.mean(((signal - mean) / std) ** 3)) if std else 0.0,
        "kurtosis": float(np.mean(((signal - mean) / std) ** 4) - 3) if std else 0.0,
        "crest_factor": (peak / rms) if rms else 0.0,
        "form_factor": (rms / mean_abs) if mean_abs > 0 else 0.0,
        "impulse_factor": (peak / mean_abs) if mean_abs > 0 else 0.0,
        "margin_factor": (peak / (mean_root**2)) if mean_root > 0 else 0.0,
    }

def _streaks(arr: np.ndarray) -> List[int]:
    streaks, cur = [], 0
    for v in arr:
        if v: cur += 1
        elif cur > 0: streaks.append(cur); cur = 0
    if cur > 0: streaks.append(cur)
    return streaks

# ----- file-level TD indicators -----
def compute_diagnostic_indicators(parsed_result: Dict) -> Dict[str, Any]:
    if not parsed_result.get("success") or not parsed_result.get("segments"):
        return {"error": "Invalid input or processing failure"}
    segments = parsed_result["segments"]; n = len(segments); t = np.arange(n)
    df = pd.DataFrame(segments)
    rms = df["rms"].to_numpy(); peaks = df["peak"].to_numpy()
    kurt = df["kurtosis"].to_numpy(); cf = df["crest_factor"].to_numpy()
    skew = df["skewness"].to_numpy()

    diagnostics = {"tier1": {}, "tier2": {}, "tier3": {}, "tier4": {}, "tier5": {}, "warnings": []}
    slope = np.polyfit(t, rms, 1)[0]; diagnostics["tier1"]["rms_trend_slope"] = float(slope)
    kb = 100 * np.sum(kurt > 5) / n; diagnostics["tier1"]["kurtosis_burst_index"] = float(kb)
    p2r = peaks / np.clip(rms, 1e-12, None); diagnostics["tier1"]["peak_to_rms_ratio_sd"] = float(np.std(p2r))
    sev = peaks * kurt; diagnostics["tier1"]["top_3_severity"] = float(np.mean(np.sort(sev)[-3:]) if n >= 3 else np.max(sev))
    med_peak = np.median(peaks)
    diagnostics["tier2"]["transient_event_density"] = float(100 * np.sum(peaks > 5 * med_peak) / n)
    diagnostics["tier2"]["crest_factor_stability"] = float(1 / (np.std(cf) + 1e-3))
    kflag = kurt > 4.5; diagnostics["tier2"]["impulse_persistence"] = float(max(_streaks(kflag), default=0) / n)
    quad = np.polyfit(t, rms, 2)[0] if n > 2 else 0.0; diagnostics["tier3"]["degradation_acceleration"] = float(quad)
    diagnostics["tier3"]["dynamic_range_collapse"] = float(np.ptp(peaks) / np.maximum(np.mean(rms), 1e-12))
    pos = 100 * np.sum(skew > 0.5) / n; neg = 100 * np.sum(skew < -0.5) / n
    diagnostics["tier3"]["skewness_polarity_shift"] = float(pos - neg)
    covar = np.cov(kurt, rms)[0, 1]; diagnostics["tier4"]["kurtosis_rms_covariance"] = float(covar)
    low_rms = rms < 0.1 * np.median(rms); high_peak = peaks > 3 * rms
    noise_idx = 100 * np.sum(low_rms & high_peak) / n; diagnostics["tier4"]["noise_contamination_index"] = float(noise_idx)
    res = rms - (slope * t + np.mean(rms)); r2 = 1 - np.var(res) / np.maximum(np.var(rms), 1e-12)
    diagnostics["tier4"]["trend_inconsistency_flag"] = int(r2 < 0.7)
    prog = (np.max(peaks) * kb) / (abs(slope) + 0.01); diagnostics["tier5"]["fault_progression_score"] = float(prog)
    rms_sk = np.mean((rms - np.mean(rms)) ** 3) / (np.std(rms) + 1e-12) ** 3
    peak_ku = np.mean((peaks - np.mean(peaks)) ** 4) / (np.std(peaks) + 1e-12) ** 4 - 3
    diagnostics["tier5"]["harmonic_distortion_indicator"] = float(rms_sk * peak_ku)
    if kb > 20: diagnostics["warnings"].append(f"High kurtosis bursts ({kb:.1f}% segments > 5)")
    if noise_idx > 15: diagnostics["warnings"].append(f"Potential noise contamination ({noise_idx:.1f}%)")
    if prog > 5000: diagnostics["warnings"].append(f"Severe fault progression (score={prog:.0f})")
    return diagnostics

# ----- TD rules screen -----
def infer_fault_location_time_domain(parsed_result: Dict, load_hp: int = 1) -> Dict[str, Any]:
    if not parsed_result.get("success"): return {"available": False, "reason": "parse failed"}
    segments = parsed_result["segments"]; n = len(segments)
    if n == 0: return {"available": False, "reason": "no segments"}
    df = pd.DataFrame(segments)
    kval = df["kurtosis"].to_numpy(); cf = df["crest_factor"].to_numpy()
    peak = df["peak"].to_numpy(); rms = df["rms"].to_numpy(); skew = df["skewness"].to_numpy()
    q = df["quality_score"].to_numpy() if "quality_score" in df else np.ones(len(df))
    avg_quality = float(np.mean(q)) if len(q) else 0.0

    di = compute_diagnostic_indicators(parsed_result)
    kb = di["tier1"]["kurtosis_burst_index"]; sev = di["tier1"]["top_3_severity"]
    evd = di["tier2"]["transient_event_density"]; imp = di["tier2"]["impulse_persistence"]
    p2rsd = di["tier1"]["peak_to_rms_ratio_sd"]
    mean_cf = float(np.mean(cf)); mean_k = float(np.mean(kval))
    mean_abs_skew = float(np.mean(np.abs(skew)))
    mean_impulse = float(np.mean(df["impulse_factor"].to_numpy())) if "impulse_factor" in df else 0.0

    score_IR = 0.45*(kb/30.0) + 0.25*imp + 0.20*(mean_cf/10.0) + 0.10*(sev/(np.max(peak)+1e-6))
    score_OR = 0.50*(evd/60.0) + 0.20*p2rsd + 0.20*(mean_cf/10.0) + 0.10*(mean_k/8.0)
    score_Ball = 0.40*(mean_abs_skew/2.0) + 0.30*(mean_cf/10.0) + 0.20*(mean_impulse/10.0) + 0.10*(p2rsd/5.0)
    score_Normal = 1.0 - min(1.0, 0.4*(kb/30.0) + 0.3*(evd/60.0) + 0.3*(float(np.mean(peak))/max(float(np.mean(rms)),1e-6)))

    raw = {"Normal": score_Normal, "IR": score_IR, "OR": score_OR, "Ball": score_Ball}
    vals = np.array(list(raw.values()), dtype=float)
    ex = np.exp(vals - np.max(vals)); probs = ex / np.sum(ex)
    classes = list(raw.keys()); top_idx = int(np.argmax(probs))
    top = classes[top_idx]; top_p = float(probs[top_idx]); second_p = float(np.sort(probs)[-2])

    sensor = str(parsed_result.get("metadata", {}).get("sensor_key", "DE")).upper()
    sensor_w = SENSOR_WEIGHT.get(sensor, 0.9)
    reliability = TIME_DOMAIN_RELIABILITY.get(top, 0.6 if top != "Normal" else 0.9)
    sep = max(0.0, top_p - second_p)

    confidence = top_p
    confidence *= (0.5 + 0.5 * avg_quality)
    confidence *= (0.8 + 0.2 * sensor_w)
    confidence *= (0.6 + 0.4 * min(1.0, sep * 3))
    confidence *= reliability

    th = _load_td_thresholds()
    probs_min = th.get("probs_min", {}); conf_min = float(th.get("confidence_min", 0.50))
    abstain = False; abstain_reason = ""
    if top != "Normal":
        need_prob = float(probs_min.get(top, 0.55))
        if top_p < need_prob or confidence < conf_min:
            abstain = True
            abstain_reason = f"{top} prob {top_p:.2f} (need ≥ {need_prob:.2f}) or confidence {confidence:.2f} (need ≥ {conf_min:.2f})"

    return {
        "available": True,
        "label": top if not abstain else "Uncertain",
        "confidence": float(max(0.0, min(1.0, confidence))),
        "probs": {c: float(p) for c, p in zip(classes, probs)},
        "abstain": abstain,
        "reason": abstain_reason,
        "details": {"avg_quality": avg_quality, "sensor": sensor}
    }
