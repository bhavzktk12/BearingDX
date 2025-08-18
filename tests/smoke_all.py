# tests/smoke_all.py
# Readable smoke report for TD+FD fusion
# Usage:
#   py tests\smoke_all.py
#   py tests\smoke_all.py --root "<path>" --save reports\smoke_report.json --perload

from __future__ import annotations
import os, sys, json, argparse
from collections import defaultdict, Counter

# --- path shim so project modules import cleanly ---
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bdxio.vp1 import parse_vibration_file
from utils.naming import infer_load_from_name
from engine.fusion import infer_fault_location_fused

CLASSES = ["Normal", "IR", "OR", "Ball"]
COLUMNS = ["Normal", "IR", "OR", "Ball", "Uncertain"]

def _pretty_table(headers, rows, colw=10):
    line = " ".join(h.ljust(colw) for h in headers)
    bar  = "-" * len(line)
    print(line); print(bar)
    for r in rows:
        print(" ".join(str(x).ljust(colw) for x in r))

def _confusion_to_rows(conf: dict[str, dict[str, int]]):
    rows: list[list[str]] = []
    for true_lbl in CLASSES:
        row: list[str] = [true_lbl]
        row_counts = conf.get(true_lbl, {})
        for pred_lbl in COLUMNS:
            row.append(str(row_counts.get(pred_lbl, 0)))  # cast to str for pretty print
        rows.append(row)
    return rows

def _metrics(conf: dict[str, dict[str,int]]):
    total = 0
    correct = 0
    uncertain = 0
    false_ball_ir = conf.get("IR", {}).get("Ball", 0)
    false_ball_or = conf.get("OR", {}).get("Ball", 0)
    ball_total = sum(conf.get("Ball", {}).values())
    ball_correct = conf.get("Ball", {}).get("Ball", 0)

    for t in CLASSES:
        row = conf.get(t, {})
        total += sum(row.values())
        correct += row.get(t, 0)
        uncertain += row.get("Uncertain", 0)

    acc = (correct / total) if total else 0.0
    ball_recall = (ball_correct / ball_total) if ball_total else 0.0
    uncertain_rate = (uncertain / total) if total else 0.0

    return {
        "total": total,
        "accuracy": acc,
        "ball_recall": ball_recall,
        "false_ball_ir": false_ball_ir,
        "false_ball_or": false_ball_or,
        "uncertain_rate": uncertain_rate,
    }

def _merge_pred(conf, true_lbl, pred_lbl):
    conf[true_lbl][pred_lbl] += 1

def run(root: str, save: str|None, perload: bool):
    conf = defaultdict(Counter)
    per_load = defaultdict(lambda: defaultdict(Counter))
    file_count = 0
    loads_seen = set()

    for cls in CLASSES:
        d = os.path.join(root, cls)
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            if not fn.lower().endswith(".mat"): continue
            fp = os.path.join(d, fn)
            parsed = parse_vibration_file(fp, sensor_key="DE", sampling_rate=12000)
            if not parsed.get("success") or not parsed.get("segments"): 
                continue
            L = infer_load_from_name(fp)
            loads_seen.add(L)
            fused = infer_fault_location_fused(parsed, load_hp=L)
            pred = fused.get("label", "Uncertain")
            _merge_pred(conf, cls, pred)
            _merge_pred(per_load[L], cls, pred)
            file_count += 1

    # --- Topline summary ---
    m = _metrics({k: dict(v) for k, v in conf.items()})
    print("\n=== BearingDX Fused Smoke Report ===")
    print(f"Dataset root : {root}")
    print(f"Files parsed : {file_count}")
    print(f"Loads seen   : {sorted(loads_seen)}")
    print("\n--- Topline ---")
    print(f"Accuracy         : {m['accuracy']*100:5.1f}%  "
          f"(correct/total from 4-class + Uncertain)")
    print(f"Ball recall      : {m['ball_recall']*100:5.1f}%  "
          f"(true Ball predicted as Ball)")
    print(f"False Ball (IR)  : {m['false_ball_ir']}")
    print(f"False Ball (OR)  : {m['false_ball_or']}")
    print(f"Uncertain rate   : {m['uncertain_rate']*100:5.1f}%")

    # --- Confusion (all loads) ---
    print("\n--- Confusion (all loads) ---")
    headers = ["true\\pred"] + COLUMNS
    rows = _confusion_to_rows({k: dict(v) for k,v in conf.items()})
    _pretty_table(headers, rows, colw=12)

    # --- Per-load breakdown (optional) ---
    if perload:
        print("\n--- Per-load breakdown ---")
        for L in sorted(per_load.keys()):
            print(f"\nLoad {L}")
            rowsL = _confusion_to_rows({k: dict(v) for k,v in per_load[L].items()})
            _pretty_table(headers, rowsL, colw=12)
            mL = _metrics({k: dict(v) for k,v in per_load[L].items()})
            print(f"  accuracy={mL['accuracy']*100:4.1f}%  "
                  f"ball_recall={mL['ball_recall']*100:4.1f}%  "
                  f"falseBall(IR)={mL['false_ball_ir']}  falseBall(OR)={mL['false_ball_or']}  "
                  f"uncertain={mL['uncertain_rate']*100:4.1f}%")

    # --- Optional save ---
    report = {
        "confusion": {k: dict(v) for k,v in conf.items()},
        "per_load": {int(L): {k: dict(v) for k,v in d.items()} for L,d in per_load.items()},
        "metrics": m,
    }
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        with open(save, "w", encoding="utf-8") as f: json.dump(report, f, indent=2)
        print(f"\n✅ saved: {save}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train")
    ap.add_argument("--save", default=r"reports\smoke_report.json")
    ap.add_argument("--perload", action="store_true", help="Show per-load confusion + metrics")
    args = ap.parse_args()
    run(args.root, args.save, args.perload)
