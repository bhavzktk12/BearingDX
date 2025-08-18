"""
BearingDX - Phase-1 Bearing Knowledge Module
+ Baseline-aware quality validation (per-sensor DE/FE; BA fallback)
+ Load→RPM mapping (used now for metadata, later for spectral)
"""

import os, json, math
from datetime import datetime
from typing import Dict, Any, List

# ===== BEARING SPEC =====
BEARING_SPECS = {
    'default_bearing': {
        'manufacturer': 'SKF',
        'model': '6205-2RS_JEM',
        'type': 'deep_groove_ball',
        'num_balls': 9,
        'ball_diameter_mm': 7.94,
        'pitch_diameter_mm': 39.05,
        'contact_angle_deg': 0
    }
}

# ===== CONSTANTS =====
RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}  # CWRU nominal
DEFAULT_RPM = RPM_BY_LOAD[1]
ENABLE_LEARNING = False

DEFAULT_SIGNAL_THRESHOLDS = {   # final fallback
    'rms': (0.05, 0.8),
    'peak': (0.3, 4.0),
    'crest_factor': (2.0, 15.0),
    'kurtosis': (1.0, 20.0)
}

# === Baseline access ===
try:
    from health.baseline import load_baseline, get_sensor_baseline, INVARIANT_KEYS
except Exception:
    load_baseline = None
    get_sensor_baseline = None
    INVARIANT_KEYS = ["crest_factor","kurtosis","skewness","form_factor","impulse_factor","margin_factor"]

_BASELINE_CACHE: Dict[str, Any] = {}
def _baseline_root() -> Dict[str, Any]:
    global _BASELINE_CACHE
    if _BASELINE_CACHE: return _BASELINE_CACHE
    if load_baseline is None: return {}
    try:
        _BASELINE_CACHE = load_baseline() or {}
    except Exception:
        _BASELINE_CACHE = {}
    return _BASELINE_CACHE

def get_fault_frequencies(rpm: float, bearing_id: str = 'default_bearing') -> Dict[str, float]:
    spec = BEARING_SPECS[bearing_id]
    n = spec['num_balls']; Bd = spec['ball_diameter_mm']; Pd = spec['pitch_diameter_mm']
    beta = math.radians(spec['contact_angle_deg'])
    fr = rpm / 60.0; dr = Bd / Pd; cosb = math.cos(beta)
    return {
        'shaft_freq': round(fr, 2),
        'BPFI': round(0.5 * n * fr * (1 + dr * cosb), 2),
        'BPFO': round(0.5 * n * fr * (1 - dr * cosb), 2),
        'BSF':  round((Pd / (2 * Bd)) * fr * (1 - (dr * cosb)**2), 2),
        'FTF':  round(0.5 * fr * (1 - dr * cosb), 2)
    }

def validate_signal_quality(stats: Dict[str, float], sensor_key: str = "DE") -> Dict[str, Any]:
    """
    Priority:
      (A) If baseline exists for this sensor: use per-feature percentile bands
      (B) Else if BA or unknown sensor: check only amplitude-invariant features with broad sanity
      (C) Else: fallback to default coarse thresholds
    """
    warnings: List[str] = []
    score, total = 0, 0

    root = _baseline_root()
    sensor_bl = get_sensor_baseline(root, sensor_key) if get_sensor_baseline else {}

    # (A) sensor baseline exists
    if sensor_bl and "per_feature" in sensor_bl:
        pf = sensor_bl["per_feature"]
        for k, v in stats.items():
            if k in pf:
                total += 1
                low, high = float(pf[k]["p_low"]), float(pf[k]["p_high"])
                if low <= v <= high:
                    score += 1
                else:
                    warnings.append(f"{sensor_key}:{k}={v:.3f} outside baseline [{low:.3f},{high:.3f}]")

    # (B) BA / missing baseline → invariant subset
    elif sensor_key.upper() == "BA" or not sensor_bl:
        for k in INVARIANT_KEYS:
            if k in stats:
                total += 1
                # very broad sanity bands for invariants
                bands = {
                    "kurtosis": (0.0, 25.0),
                    "crest_factor": (1.5, 20.0),
                    "skewness": (-3.0, 3.0),
                    "form_factor": (0.5, 3.0),
                    "impulse_factor": (1.0, 30.0),
                    "margin_factor": (1.0, 60.0),
                }
                low, high = bands[k]
                val = float(stats[k])
                if low <= val <= high:
                    score += 1
                else:
                    warnings.append(f"{sensor_key}:{k}={val:.3f} sanity out [{low},{high}]")

    # (C) last resort default numeric thresholds
    else:
        for key, (low, high) in DEFAULT_SIGNAL_THRESHOLDS.items():
            if key in stats:
                total += 1
                val = stats[key]
                if low <= val <= high:
                    score += 1
                else:
                    warnings.append(f"{sensor_key}:{key}={val:.3f} out [{low},{high}]")

    quality_score = round(score / total, 2) if total else 0.0
    return {"is_valid": quality_score >= 0.7, "quality_score": quality_score, "warnings": warnings}

# ===== Passive log =====
class BearingKnowledge:
    def __init__(self):
        self.log = {'success': [], 'failure': []}
    def learn_from_success(self, filename: str, stats: Dict[str, Any]):
        if not ENABLE_LEARNING: return
        entry = {"timestamp": datetime.now().isoformat(), "filename": filename, **{k:stats.get(k) for k in ("rms","crest_factor","kurtosis","quality_score")}}
        self.log['success'].append(entry); self._truncate('success')
    def learn_from_failure(self, filename: str, error: str):
        if not ENABLE_LEARNING: return
        entry = {"timestamp": datetime.now().isoformat(), "filename": filename, "error": error}
        self.log['failure'].append(entry); self._truncate('failure')
    def _truncate(self, key: str, limit: int = 1000):
        if len(self.log[key]) > limit: self.log[key] = self.log[key][-limit:]
    def export_log(self, path: str = "logs/knowledge_log.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f: json.dump(self.log, f, indent=2)

bearing_knowledge = BearingKnowledge()
