# engine/fusion.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import json

# Reuse helpers implemented in dx1/bk1
from features.td import infer_fault_location_time_domain
from features.fd import compute_fd_indicators, decide_ball_fault_fd

from bk1 import RPM_BY_LOAD

# ============================
# Thresholds (defaults + JSON)
# ============================
# Defaults used if JSON not found; JSON keys override values here
DEFAULT_THRESHOLDS: Dict[str, float] = {
    # Ball gate / promotion
    "min_score": 0.72,
    "min_bsf_snr": 1.90,
    "min_ftf_snr": 1.12,
    "min_sb_ratio": 1.02,
    "promote_score": 0.72,
    "bpfi_block_ratio": 0.85,
    "bpfo_block_ratio": 0.85,
    "no_fallback_ball_score": 0.70,
    "no_fallback_ball_rel": 0.55,

    # IR/OR dominance (env-only for Normal gate + fallback)
    "iror_snr_min": 1.70,
    "iror_ratio_min": 1.10,

    # Normal fast-path (env-only so FFT energy can't fake faults)
    "normal_td_conf_min": 0.60,
    "normal_ball_score_max": 0.65,
    "normal_iror_dom_max": 1.60,
    "normal_iror_ratio_max": 1.15,
    "normal_fault_energy_max": 2.00,
    "min_health_for_normal": 0.70,

}

_THRESHOLDS: Optional[Dict[str, float]] = None
_THRESHOLDS_SRC: Optional[str] = None

def _load_thresholds_once() -> None:
    """Load thresholds from JSON (if found), falling back to DEFAULT_THRESHOLDS."""
    global _THRESHOLDS, _THRESHOLDS_SRC
    if _THRESHOLDS is not None:
        return

    # repo root is parent of engine/
    base = Path(__file__).resolve().parents[1]
    candidates = [
        base / "config" / "fusion_thresholds.json",         # preferred
        base / "fusion_thresholds.json",                    # legacy root fallback
        Path.cwd() / "config" / "fusion_thresholds.json",   # last resort: CWD
    ]

    for p in candidates:
        try:
            if p.exists():
                data = json.loads(p.read_text())
                if isinstance(data, dict):
                    t = dict(DEFAULT_THRESHOLDS)
                    # Only accept numeric values
                    for k, v in data.items():
                        try:
                            t[k] = float(v)
                        except Exception:
                            pass
                    _THRESHOLDS = t
                    _THRESHOLDS_SRC = str(p)
                    break
        except Exception:
            pass

    if _THRESHOLDS is None:
        _THRESHOLDS = dict(DEFAULT_THRESHOLDS)
        _THRESHOLDS_SRC = None

def get_active_thresholds() -> Tuple[Dict[str, float], Optional[str]]:
    """Returns a copy of the active thresholds and the JSON source path (or None)."""
    _load_thresholds_once()
    return dict(_THRESHOLDS or DEFAULT_THRESHOLDS), _THRESHOLDS_SRC

# Ensure thresholds are loaded at import time
_load_thresholds_once()


def infer_fault_location_fused(
    parsed_result: Dict[str, Any],
    load_hp: int = 1,
    thresholds: Optional[Dict[str, float]] = None,
    health: Optional[Dict[str, Any]] = None,   # ← new
) -> Dict[str, Any]:

    """
    TD+FD fusion for fault location:
      - TD screen first
      - Ball decision gated by FD (BSF/FTF + sidebands), 2xBSF-aware
      - Envelope-only dominance for Normal gate + IR/OR fallback (prevents FFT energy from faking faults)
      - Blended dominance (env + FFT) is used for anti-IR/OR guards
      - Per-load easing at 2–3 HP
      - Soft Ball confirm/promotion paths to recover recall safely
      - Normal fast-path: TD Normal/Uncertain/weak IR-OR + quiet FD => Normal
    """
    if not parsed_result.get("success"):
        return {"available": False, "label": "Uncertain", "reason": "parse failed"}

    # thresholds (config-driven)
    thr, _src = get_active_thresholds()
    if thresholds:
        thr.update(thresholds)
    hs = float((health or {}).get("health_score") or 0.0)
    min_h = float(thr.get("min_health_for_normal", 0.70))

    # Per-load easing (Ball often faint at higher loads)
    def _per_load_tweak(L: int, t: Dict[str, float]) -> Dict[str, float]:
        if int(L) in (2, 3):
            t = dict(t)
            t["promote_score"] = max(0.0, t["promote_score"] - 0.02)
            t["min_score"] = max(0.0, t["min_score"] - 0.02)
            t["no_fallback_ball_rel"] = 0.52
        return t

    thr = _per_load_tweak(load_hp, thr)

    # TD pass
    td = infer_fault_location_time_domain(parsed_result, load_hp=load_hp)
    fused: Dict[str, Any] = dict(td) if isinstance(td, dict) else {"label": "Uncertain", "confidence": 0.0}
    if not fused.get("label"):
        fused["label"] = "Uncertain"

    # FD evidence
    rpm = RPM_BY_LOAD.get(int(load_hp), 1772)
    fd = compute_fd_indicators(parsed_result, rpm, env_band="auto")

    # Envelope SNRs
    ir_s = float(fd.get("env_bpfi_snr", 0.0))
    or_s = float(fd.get("env_bpfo_snr", 0.0))
    bsf  = float(fd.get("env_bsf_snr", 0.0))
    bsf2 = float(fd.get("env_bsf2_snr", 0.0))
    ftf  = float(fd.get("env_ftf_snr", 0.0))
    bsf_eff = max(bsf, bsf2)  # 2xBSF often stronger

    # FFT norms
    fft_ir = float(fd.get("fft_bpfi_norm", 0.0))
    fft_or = float(fd.get("fft_bpfo_norm", 0.0))

    # Dominance metrics
    # ENV-ONLY (use for Normal gate + IR/OR fallback)
    ir_dom_env, or_dom_env = ir_s, or_s
    # BLENDED (env + FFT) for anti-IR/OR guards
    ir_dom_mix = ir_s + 0.7 * fft_ir
    or_dom_mix = or_s + 0.7 * fft_or

    def _dom_ratio(a: float, b: float) -> Tuple[float, float, str]:
        dom = max(a, b)
        sub = min(a, b)
        ratio = dom / (sub + 1e-6)
        lab = "IR" if a >= b else "OR"
        return dom, ratio, lab

    # Ball FD oracle (your gate)
    g = decide_ball_fault_fd(fd, rpm, {
        "min_bsf_snr": thr["min_bsf_snr"],
        "min_ftf_snr": thr["min_ftf_snr"],
        "min_sb_ratio": thr["min_sb_ratio"],
        "min_score": thr["min_score"],
    })
    fused["fd_ball_gate"] = g

    # Normal fast-path (ENV-ONLY checks so FFT energy can't fake faults)
    td_label = fused.get("label") or "Uncertain"
    td_conf  = float(fused.get("confidence", 0.0))

    env_dom, env_ratio, env_lab = _dom_ratio(ir_dom_env, or_dom_env)
    fault_energy_env = max(bsf_eff, ftf, ir_dom_env, or_dom_env)

    quiet_fd = (
        g.get("score", 0.0) <= float(thr.get("normal_ball_score_max", 0.65)) and
        env_dom              <= float(thr.get("normal_iror_dom_max",   1.60)) and
        env_ratio            <= float(thr.get("normal_iror_ratio_max", 1.15)) and
        fault_energy_env     <= float(thr.get("normal_fault_energy_max", 2.00))
    )

    # A) TD Normal + quiet FD + healthy -> Normal
    if td_label == "Normal" and quiet_fd and hs >= min_h:
        fused["reason"] = (
            f"TD Normal + quiet FD + healthy "
            f"(td={td_conf:.2f}, ball={g.get('score',0.0):.2f}, "
            f"env_dom={env_dom:.2f}, env_ratio={env_ratio:.2f}, fe={fault_energy_env:.2f}, "
            f"health={hs:.2f}>={min_h:.2f})"
        )
        fused["label"] = "Normal"
        fused["health_score"] = hs
        fused["fd"] = {
            "env_band_lo": float(fd.get("env_band_lo", 0.0)),
            "env_band_hi": float(fd.get("env_band_hi", 0.0)),
            "env_bpfi_snr": ir_s, "env_bpfo_snr": or_s,
            "env_bsf_snr": bsf, "env_bsf2_snr": bsf2, "env_ftf_snr": ftf,
            "fft_bpfi_norm": fft_ir, "fft_bpfo_norm": fft_or,
            "score_ball": float(g.get("score", 0.0)),
        }
        return fused


      # B) TD Uncertain + quiet FD + healthy -> Normal
    if td_label == "Uncertain" and quiet_fd and hs >= min_h:
        fused["reason"] = (
            f"TD Uncertain + quiet FD + healthy -> Normal "
            f"(td={td_conf:.2f}, ball={g.get('score',0.0):.2f}, "
            f"env_dom={env_dom:.2f}, env_ratio={env_ratio:.2f}, fe={fault_energy_env:.2f}, "
            f"health={hs:.2f}>={min_h:.2f})"
        )
        fused["label"] = "Normal"
        fused["health_score"] = hs
        fused["fd"] = {
            "env_band_lo": float(fd.get("env_band_lo", 0.0)),
            "env_band_hi": float(fd.get("env_band_hi", 0.0)),
            "env_bpfi_snr": ir_s, "env_bpfo_snr": or_s,
            "env_bsf_snr": bsf, "env_bsf2_snr": bsf2, "env_ftf_snr": ftf,
            "fft_bpfi_norm": fft_ir, "fft_bpfo_norm": fft_or,
            "score_ball": float(g.get("score", 0.0)),
        }
        return fused


   # C) TD weak IR/OR (low conf) + quiet FD + healthy -> Normal
    if (td_label in ("IR", "OR")) and (td_conf < float(thr.get("normal_td_conf_min", 0.60))) and quiet_fd and hs >= min_h:
        fused["reason"] = (
            f"TD {td_label} low conf + quiet FD + healthy -> Normal "
            f"(td={td_conf:.2f}, ball={g.get('score',0.0):.2f}, "
            f"env_dom={env_dom:.2f}, env_ratio={env_ratio:.2f}, fe={fault_energy_env:.2f}, "
            f"health={hs:.2f}>={min_h:.2f})"
        )
        fused["label"] = "Normal"
        fused["health_score"] = hs
        fused["fd"] = {
            "env_band_lo": float(fd.get("env_band_lo", 0.0)),
            "env_band_hi": float(fd.get("env_band_hi", 0.0)),
            "env_bpfi_snr": ir_s, "env_bpfo_snr": or_s,
            "env_bsf_snr": bsf, "env_bsf2_snr": bsf2, "env_ftf_snr": ftf,
            "fft_bpfi_norm": fft_ir, "fft_bpfo_norm": fft_or,
            "score_ball": float(g.get("score", 0.0)),
        }
        return fused


    # Fault decision logic
    if td_label == "Ball":
        # Confirm or soft-confirm Ball
        if g.get("is_ball") and g.get("score", 0.0) >= thr["min_score"]:
            fused["reason"] = f"Ball confirmed by FD (score={g.get('score',0):.2f})"
        elif g.get("is_ball") and (g.get("score", 0.0) >= 0.70) and (bsf_eff >= 1.80 or ftf >= 1.08):
            fused["reason"] = f"Ball soft-confirmed (score={g.get('score',0):.2f})"
        else:
            # Use ENV-ONLY for IR/OR dominance so FFT energy alone can't veto Ball into IR/OR
            env_dom2, env_ratio2, env_lab2 = _dom_ratio(ir_dom_env, or_dom_env)
            if (
                env_dom2 >= max(thr["iror_snr_min"], 2.0)
                and env_ratio2 >= max(thr["iror_ratio_min"], 1.25)
                and env_dom2 > 1.12 * max(bsf_eff, ftf)
            ):
                fused["label"] = env_lab2
                fused["reason"] = f"Ball vetoed -> {env_lab2} dominance (env_ratio={env_ratio2:.2f})"
            else:
                fused["label"] = "Uncertain"
                fused["reason"] = "Ball inconclusive; IR/OR not clearly dominant"
    else:
        # Promotion (strict then soft) with anti-IR/OR guards (mix ok here)
        anti_ir = (ir_dom_mix < thr["bpfi_block_ratio"] * max(bsf_eff, 1e-6))
        anti_or = (or_dom_mix < thr["bpfo_block_ratio"] * max(bsf_eff, 1e-6))
        if g.get("is_ball") and g.get("score", 0.0) >= thr["promote_score"] and anti_ir and anti_or:
            fused["label"] = "Ball"
            fused["reason"] = f"Promoted to Ball (score={g.get('score',0):.2f}, BSF={bsf_eff:.2f}, FTF={ftf:.2f})"
        elif g.get("is_ball") and (g.get("score", 0.0) >= 0.70) and (
            (max(bsf_eff, ftf) / (max(ir_dom_env, or_dom_env, 1e-6))) >= thr["no_fallback_ball_rel"]
        ) and anti_ir and anti_or:
            fused["label"] = "Ball"
            fused["reason"] = f"Promoted (soft) to Ball (score={g.get('score',0):.2f}, rel={(max(bsf_eff, ftf) / (max(ir_dom_env, or_dom_env, 1e-6))):.2f})"
        else:
            # TD Uncertain -> IR/OR fallback only when ENV-ONLY dominance is strong
            if td_label in ("Uncertain", "Unknown", None):
                env_dom3, env_ratio3, env_lab3 = _dom_ratio(ir_dom_env, or_dom_env)
                if g.get("score", 0.0) >= thr["no_fallback_ball_score"] or (
                    (max(bsf_eff, ftf) / (max(ir_dom_env, or_dom_env, 1e-6))) >= thr["no_fallback_ball_rel"]
                ):
                    fused["reason"] = "Held Uncertain (Ball plausible; avoid IR/OR fallback)"
                elif env_dom3 >= thr["iror_snr_min"] and env_ratio3 >= thr["iror_ratio_min"]:
                    fused["label"] = env_lab3
                    fused["reason"] = f"FD fallback (ENV {env_lab3} dominance: ratio={env_ratio3:.2f})"
                else:
                    fused["reason"] = "Uncertain: no clear TD or FD dominance"

    # Compact FD dump
    fused["fd"] = {
        "env_band_lo": float(fd.get("env_band_lo", 0.0)),
        "env_band_hi": float(fd.get("env_band_hi", 0.0)),
        "env_bpfi_snr": ir_s, "env_bpfo_snr": or_s,
        "env_bsf_snr": bsf, "env_bsf2_snr": bsf2, "env_ftf_snr": ftf,
        "fft_bpfi_norm": fft_ir, "fft_bpfo_norm": fft_or,
        "score_ball": float(g.get("score", 0.0)),
    }
    return fused
