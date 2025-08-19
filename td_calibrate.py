# td_calibrate.py
import os, json, re
from typing import Dict, List
from bdxio.vp1 import parse_vibration_file
from features.td import infer_fault_location_time_domain
from bk1 import RPM_BY_LOAD

NORMAL_FILES = [
    r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train\Normal\normal_0_1797.mat",
    r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train\Normal\normal_1_1772.mat",
    r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train\Normal\normal_2_1750.mat",
    r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train\Normal\normal_3_1730.mat",
]


OUT_PATH = "models/td_thresholds.json"

def rpm_to_load(rpm: int) -> int:
    best, best_err = None, 10**9
    for load, r in RPM_BY_LOAD.items():
        err = abs(int(rpm) - int(r))
        if err < best_err:
            best, best_err = load, err
    return best if best is not None else 1

def infer_load_from_name(path: str) -> int:
    m = re.search(r"(\d{3,5})(?=\.mat$)", os.path.basename(path))
    if m:
        return rpm_to_load(int(m.group(1)))
    # fallback by filename convention normal_X_YYYY
    m2 = re.search(r"normal_(\d)_\d+", os.path.basename(path), re.I)
    if m2:
        return int(m2.group(1))
    return 1

def main():
    max_fault_prob = {"IR": 0.0, "OR": 0.0, "Ball": 0.0}
    print("Calibrating thresholds from Normal files...")
    for fp in NORMAL_FILES:
        if not os.path.exists(fp):
            print(f"  ⚠️ Missing: {fp}")
            continue
        load_hp = infer_load_from_name(fp)
        res = parse_vibration_file(fp, sensor_key="DE", sampling_rate=12000)
        pred = infer_fault_location_time_domain(res, load_hp)
        if not pred.get("available", False):
            print(f"  ⚠️ Skipped (parse/pred fail): {fp}")
            continue
        # track how 'fault-like' Normal ever looks
        for c in ("IR", "OR", "Ball"):
            p = float(pred["probs"].get(c, 0.0))
            if p > max_fault_prob[c]:
                max_fault_prob[c] = p
        print(f"  {os.path.basename(fp)} → non-Normal max = "
              f"IR {max_fault_prob['IR']:.2f}, OR {max_fault_prob['OR']:.2f}, Ball {max_fault_prob['Ball']:.2f}")

    # Floors = max seen on Normal + margin, with sensible minimums per class
    floors = {
        "IR":  max(0.55, max_fault_prob["IR"]  + 0.05),
        "OR":  max(0.60, max_fault_prob["OR"]  + 0.05),
        "Ball":max(0.65, max_fault_prob["Ball"]+ 0.05),
    }
    bundle = {
        "probs_min": floors,
        "confidence_min": 0.50,  # also require final confidence ≥ 0.50
        "source": "calibrated_from_normals",
        "normals_seen": [os.path.basename(f) for f in NORMAL_FILES if os.path.exists(f)]
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(bundle, f, indent=2)
    print("✅ Wrote", OUT_PATH)
    print(" Floors:", floors)

if __name__ == "__main__":
    main()
