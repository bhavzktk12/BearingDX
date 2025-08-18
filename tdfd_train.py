# tdfd_train.py
"""
Train TD+FD (FFT + Envelope) classifier on DE 12k data.
Saves calibrated model to models/tdfd_clf_calibrated.joblib
"""

import os, re, json, joblib
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from engine.vectorize import file_level_vector_td_fd as _file_level_vector_td_fd

from bdxio.vp1 import parse_vibration_file
from bk1 import RPM_BY_LOAD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

DATASET_ROOT = r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train"
SENSOR_KEY    = "DE"
SAMPLING_RATE = 12000
LABELS = {"Normal": 0, "IR": 1, "OR": 2, "Ball": 3}
INV_LABELS = {v:k for k,v in LABELS.items()}

def _rpm_to_load(rpm: int) -> int:
    best = min(RPM_BY_LOAD.items(), key=lambda kv: abs(kv[1]-rpm))[0]
    return best

def _infer_load_from_name(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"(\d{3,5})(?=\.mat$)", name)
    if m: return _rpm_to_load(int(m.group(1)))
    m2 = re.search(r"normal_(\d)_\d+", name, re.I)
    if m2: return int(m2.group(1))
    return 1

def _iter_labeled_files(dataset_root: str):
    for cls_name, label in LABELS.items():
        cls_dir = os.path.join(dataset_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith(".mat"):
                yield os.path.join(cls_dir, fn), label

def _build_dataset(dataset_root: str) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    X, y, loads, paths = [], [], [], []
    for fp, label in _iter_labeled_files(dataset_root):
        parsed = parse_vibration_file(fp, sensor_key=SENSOR_KEY, sampling_rate=SAMPLING_RATE)
        if not parsed.get("success") or not parsed.get("segments"):
            print(f"⚠️ Skipping (parse fail): {fp}")
            continue
        load = _infer_load_from_name(fp)
        rpm = RPM_BY_LOAD.get(load, 1772)
        vec = _file_level_vector_td_fd(parsed, rpm)
        if vec.size <= 1:
            print(f"⚠️ Skipping (empty vector): {fp}")
            continue
        X.append(vec); y.append(label); loads.append(load); paths.append(fp)
    return np.vstack(X), np.array(y), loads, paths

def _lolo_eval(X: np.ndarray, y: np.ndarray, loads: List[int], model) -> Dict[str, Any]:
    seen_loads = sorted(set(loads))
    report = {}; f1s = []
    for L in seen_loads:
        idx_te = [i for i,l in enumerate(loads) if l==L]
        idx_tr = [i for i,l in enumerate(loads) if l!=L]
        if not idx_te or not idx_tr: continue
        model.fit(X[idx_tr], y[idx_tr])
        yhat = model.predict(X[idx_te])
        f1  = f1_score(y[idx_te], yhat, average="macro", zero_division=0)
        f1s.append(f1)
        report[f"load{L}"] = {
            "macro_f1": float(f1),
            "support": int(len(idx_te)),
            "confusion": confusion_matrix(y[idx_te], yhat, labels=[0,1,2,3]).tolist(),
        }
    report["macro_f1_mean"] = float(np.mean(f1s)) if f1s else None
    return report

def main():
    print("Building TD+FD dataset from:", DATASET_ROOT)
    X, y, loads, _ = _build_dataset(DATASET_ROOT)
    print("Samples:", X.shape[0], "Dims:", X.shape[1])

    # Choose CV folds (like td_train)
    unique, counts = np.unique(y, return_counts=True)
    min_per_class = int(counts.min())
    cv_folds = max(2, min(5, min_per_class))
    print(f"[calibration] using {cv_folds}-fold CV (min per class = {min_per_class})")

    base = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=600, class_weight="balanced")),
    ])

    lolo = _lolo_eval(X, y, loads, base)
    print("\n===== LOLO (0/1/2/3 HP) TD+FD =====")
    print(json.dumps(lolo, indent=2))

    cal = CalibratedClassifierCV(base, cv=cv_folds, method="sigmoid")
    cal.fit(X, y)
    yhat = cal.predict(X)
    print("\n===== Train-set report (post-calibration, indicative) =====")
    print(classification_report(y, yhat, target_names=[INV_LABELS[i] for i in [0,1,2,3]], zero_division=0))

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": cal, "feature_names": None, "labels": INV_LABELS}, "models/tdfd_clf_calibrated.joblib")
    print("✅ Saved models/tdfd_clf_calibrated.joblib")

if __name__ == "__main__":
    main()
