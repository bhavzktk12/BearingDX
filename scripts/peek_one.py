# scripts/peek_one.py
import os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# parser (still at repo root for now)
from bdxio.vp1 import parse_vibration_file, DEFAULT_SAMPLING_RATE

# fusion + thresholds
from engine.fusion import infer_fault_location_fused, get_active_thresholds

# health baseline
from health.baseline import load_baseline, get_sensor_baseline, score_td_health

# ---------- CONFIG (set these BEFORE parsing) ----------
FP = r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train\Normal\normal_0_1797.mat"
LOAD = 0
BASELINE_PATH = os.path.join(ROOT, "models", "baseline.json")
# -------------------------------------------------------

# parse
parsed = parse_vibration_file(FP, sensor_key="DE", sampling_rate=DEFAULT_SAMPLING_RATE)

# health
sensor = (parsed.get("metadata") or {}).get("sensor_key", "DE")
root = load_baseline(BASELINE_PATH) if os.path.exists(BASELINE_PATH) else None
sb = get_sensor_baseline(root, sensor) if root else None
health = score_td_health(parsed, sb) if sb else {"health_score": 0.5, "status": None}

# fuse (pass health)
fused = infer_fault_location_fused(parsed, load_hp=LOAD, health=health)

# thresholds + source
thr, src = get_active_thresholds()
print("loaded fusion thresholds from:", src or "<built-in defaults>")

# ----- FD/quiet debug that MATCHES fusion’s quiet gate (ENV-ONLY) -----
fd = fused.get("fd", {})
td_label = fused.get("label")
td_conf = float(fused.get("confidence", 0.0))
bsf  = float(fd.get("env_bsf_snr", 0.0))
bsf2 = float(fd.get("env_bsf2_snr", 0.0))
ftf  = float(fd.get("env_ftf_snr", 0.0))
ir_s = float(fd.get("env_bpfi_snr", 0.0))
or_s = float(fd.get("env_bpfo_snr", 0.0))
fft_ir = float(fd.get("fft_bpfi_norm", 0.0))
fft_or = float(fd.get("fft_bpfo_norm", 0.0))
ball_score = float(fd.get("score_ball", 0.0))
bsf_eff = max(bsf, bsf2)

# ENV-ONLY dominance (this matches fusion’s quiet gate)
env_dom = max(ir_s, or_s)
env_ratio = env_dom / (min(ir_s, or_s) + 1e-6)
fault_energy_env = max(bsf_eff, ftf, ir_s, or_s)

quiet_fd = (
    ball_score <= float(thr.get("normal_ball_score_max", 0.65)) and
    env_dom    <= float(thr.get("normal_iror_dom_max",   1.60)) and
    env_ratio  <= float(thr.get("normal_iror_ratio_max", 1.15)) and
    fault_energy_env <= float(thr.get("normal_fault_energy_max", 2.00))
)

# Blended dominance (for display only)
ir_dom_mix = ir_s + 0.7 * fft_ir
or_dom_mix = or_s + 0.7 * fft_or
dom_mix = max(ir_dom_mix, or_dom_mix)
sub_mix = min(ir_dom_mix, or_dom_mix)
ratio_mix = dom_mix / (sub_mix + 1e-6)
fault_energy_mix = max(bsf_eff, ftf, ir_dom_mix, or_dom_mix)

print("\n=== PEAK NORMAL ===")
print("TD:", td_label, f"(conf={td_conf:.2f})  reason:", fused.get("reason",""))
print("Health:", json.dumps(health))
print("FD ball_score:", ball_score)
print(f"env: BPFI={ir_s:.2f}  BPFO={or_s:.2f}  BSF={bsf:.2f}  2xBSF={bsf2:.2f}  FTF={ftf:.2f}")
print(f"fft: BPFI_norm={fft_ir:.2f}  BPFO_norm={fft_or:.2f}")
print(f"[quiet gate] env_dom={env_dom:.2f}  env_ratio={env_ratio:.2f}  fault_energy_env={fault_energy_env:.2f}")
print(f"[blended dbg] dom_mix={dom_mix:.2f}  ratio_mix={ratio_mix:.2f}  fault_energy_mix={fault_energy_mix:.2f}")
print("quiet_fd? ->", quiet_fd)

print("\nthresholds:", json.dumps(thr, indent=2))
