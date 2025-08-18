# td_train.py
"""
Train a data-driven time-domain classifier (DE only for Phase 1.1a).
- Builds file-level feature vectors from your existing parser outputs
- Trains Logistic Regression with probability calibration (Platt/sigmoid)
- Prints leave-one-load-out metrics (0/1/2/3 HP) + overall report
- Saves calibrated model to models/td_clf_calibrated.joblib
"""

import os, re, json, joblib
import numpy as np
from typing import Dict, Any, List, Tuple

from bdxio.vp1 import parse_vibration_file, DEFAULT_SAMPLING_RATE
from dx1 import compute_diagnostic_indicators

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ---- Settings ----
DATASET_ROOT = r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train"   # <-- put your train root here
SENSOR_KEY    = "DE"
SAMPLING_RATE = 12000

LABELS = {"Normal": 0, "IR": 1, "OR": 2, "Ball": 3}
INV_LABELS = {v:k for k,v in LABELS.items()}
RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

FEATURE_KEYS = [
    "rms","peak","crest_factor","kurtosis","skewness",
    "form_factor","impulse_factor","margin_factor"
]

def _rpm_to_load(rpm: int) -> int:
    best, err = None, 10**9
    for l, r in RPM_BY_LOAD.items():
        e = abs(int(rpm) - int(r))
        if e < err:
            best, err = l, e
    return best if best is not None else 1

def _infer_load_from_name(path: str) -> int:
    # e.g. ..._1730.mat  or normal_2_1750.mat
    name = os.path.basename(path)
    m = re.search(r"(\d{3,5})(?=\.mat$)", name)
    if m: return _rpm_to_load(int(m.group(1)))
    m2 = re.search(r"normal_(\d)_\d+", name, re.I)
    if m2: return int(m2.group(1))
    return 1

def _file_level_vector(parsed: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate segments → file-level vector (pure time-domain).
    Uses basic feature stats + your diagnostic indicators.
    """
    segs = parsed.get("segments", [])
    if not segs: 
        return np.zeros(1), ["_empty"]

    import pandas as pd
    df = pd.DataFrame(segs)

    cols = []
    vec  = []

    # aggregate primary TD features
    for k in FEATURE_KEYS:
        if k in df:
            v = df[k].to_numpy(dtype=float)
            for stat_name, stat_val in (
                (f"{k}_mean", float(np.mean(v))),
                (f"{k}_std",  float(np.std(v))),
                (f"{k}_p90",  float(np.percentile(v, 90))),
            ):
                cols.append(stat_name); vec.append(stat_val)

    # burst/threshold proportions
    def frac(cond): 
        arr = cond.astype(float)
        return float(arr.mean()) if arr.size else 0.0
    if {"kurtosis","crest_factor","impulse_factor"}.issubset(df.columns):
        cols += ["frac_kurt_gt5","frac_cf_gt7","frac_imp_gt12"]
        vec  += [frac(df["kurtosis"].to_numpy()>5.0),
                 frac(df["crest_factor"].to_numpy()>7.0),
                 frac(df["impulse_factor"].to_numpy()>12.0)]

    # file-level indicators from dx1
    di = compute_diagnostic_indicators(parsed)
    take_t1 = ["rms_trend_slope","kurtosis_burst_index","peak_to_rms_ratio_sd","top_3_severity"]
    take_t2 = ["transient_event_density","impulse_persistence"]
    take_t3 = ["degradation_acceleration","dynamic_range_collapse","skewness_polarity_shift"]
    take_t4 = ["kurtosis_rms_covariance","noise_contamination_index","trend_inconsistency_flag"]
    take_t5 = ["fault_progression_score","harmonic_distortion_indicator"]
    for k in take_t1:
        cols.append(f"t1_{k}"); vec.append(float(di["tier1"].get(k, 0.0)))
    for k in take_t2:
        cols.append(f"t2_{k}"); vec.append(float(di["tier2"].get(k, 0.0)))
    for k in take_t3:
        cols.append(f"t3_{k}"); vec.append(float(di["tier3"].get(k, 0.0)))
    for k in take_t4:
        cols.append(f"t4_{k}"); vec.append(float(di["tier4"].get(k, 0.0)))
    for k in take_t5:
        cols.append(f"t5_{k}"); vec.append(float(di["tier5"].get(k, 0.0)))

    return np.asarray(vec, dtype=float), cols

def _iter_labeled_files(dataset_root: str):
    for cls_name, label in LABELS.items():
        cls_dir = os.path.join(dataset_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith(".mat"):
                yield os.path.join(cls_dir, fn), label

def _build_dataset(dataset_root: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    X, y, loads = [], [], []
    for fp, label in _iter_labeled_files(dataset_root):
        parsed = parse_vibration_file(fp, sensor_key=SENSOR_KEY, sampling_rate=SAMPLING_RATE)
        if not parsed.get("success") or not parsed.get("segments"):
            print(f"⚠️ Skipping (parse fail): {fp}")
            continue
        vec, _ = _file_level_vector(parsed)
        if vec.size <= 1:
            print(f"⚠️ Skipping (empty vector): {fp}")
            continue
        X.append(vec); y.append(label); loads.append(_infer_load_from_name(fp))
    return np.vstack(X), np.array(y), loads

def _lolo_eval(X: np.ndarray, y: np.ndarray, loads: List[int], model) -> Dict[str, Any]:
    """Leave-one-load-out evaluation."""
    seen_loads = sorted(set(loads))
    report = {}
    f1s = []
    for L in seen_loads:
        idx_te = [i for i,l in enumerate(loads) if l==L]
        idx_tr = [i for i,l in enumerate(loads) if l!=L]
        if not idx_te or not idx_tr:
            continue
        model.fit(X[idx_tr], y[idx_tr])
        yhat = model.predict(X[idx_te])
        f1  = f1_score(y[idx_te], yhat, average="macro", zero_division=0)
        f1s.append(f1)
        report[f"load{L}"] = {
            "macro_f1": float(f1),
            "support": int(len(idx_te)),
            "confusion": confusion_matrix(y[idx_te], yhat, labels=[0,1,2,3]).tolist(),
            "cls_report": classification_report(y[idx_te], yhat, target_names=[INV_LABELS[i] for i in [0,1,2,3]], zero_division=0, output_dict=True)
        }
    report["macro_f1_mean"] = float(np.mean(f1s)) if f1s else None
    return report

def main():
    print("Building dataset from:", DATASET_ROOT)
    X, y, loads = _build_dataset(DATASET_ROOT)
    # Choose calibration folds based on the smallest class (and at least 2)
    unique, counts = np.unique(y, return_counts=True)
    min_per_class = int(counts.min())
    cv_folds = max(2, min(5, min_per_class))
    print(f"\n[calibration] using {cv_folds}-fold CV (min per class = {min_per_class})")

    print("Samples:", X.shape[0], "Dims:", X.shape[1])

    # Base model
    # Base model
    base = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),  # no multi_class arg
    ])


    # LOLO metrics (data-driven, no peeking)
    lolo_report = _lolo_eval(X, y, loads, base)
    print("\n===== Leave-One-Load-Out (0/1/2/3 HP) =====")
    print(json.dumps({k: (v if k=="macro_f1_mean" else {"macro_f1": v["macro_f1"], "support": v["support"]})
                      for k,v in lolo_report.items()}, indent=2))

   
# Calibrated model (Platt/sigmoid) using chosen cv_folds
    cal = CalibratedClassifierCV(base, cv=cv_folds, method="sigmoid")
    cal.fit(X, y)                  # <— FIT FIRST
    yhat = cal.predict(X)          # then predict

    # Train-set report (for a feel; calibration uses CV internally)
    yhat = cal.predict(X)
    print("\n===== Train-set report (post-calibration, indicative) =====")
    print(classification_report(y, yhat, target_names=[INV_LABELS[i] for i in [0,1,2,3]], zero_division=0))

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": cal, "feature_names": None, "labels": INV_LABELS}, "models/td_clf_calibrated.joblib")
    print("✅ Saved models/td_clf_calibrated.joblib")

if __name__ == "__main__":
    main()
