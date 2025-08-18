import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from bdxio.vp1 import parse_vibration_file, DEFAULT_SAMPLING_RATE
from bk1 import RPM_BY_LOAD
from features.fd import compute_fd_indicators
from features.td import infer_fault_location_time_domain
from engine.fusion import infer_fault_location_fused

# 👉 change these to real files you have
tests = [
    (r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train\Normal\normal_1_1772.mat", "Normal", 1),
    (r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train\IR\IR014_1_1772.mat", "IR", 1),
    (r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train\OR\OR007@6_1_1772.mat", "OR", 2),
    (r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train\Ball\B021_1_1772.mat", "Ball", 2),
]

for fp, label, load_hp in tests:
    print("\n=== ", label, " ===")
    res = parse_vibration_file(fp, sensor_key="DE", sampling_rate=DEFAULT_SAMPLING_RATE)
    if not res["success"]:
        print("Parse failed:", res["errors"]); continue

    rpm = RPM_BY_LOAD[load_hp]
    fd = compute_fd_indicators(res, rpm, env_band="auto")
    print("Chosen env band:", round(fd["env_band_lo"],1), "→", round(fd["env_band_hi"],1))
    print("FFT norms:  bpfi", fd["fft_bpfi_norm"], "bpfo", fd["fft_bpfo_norm"], "bsf", fd["fft_bsf_norm"], "ftf", fd["fft_ftf_norm"])
    print("Env SNRs:   bpfi", fd["env_bpfi_snr"], "bpfo", fd["env_bpfo_snr"], "bsf", fd["env_bsf_snr"], "ftf", fd["env_ftf_snr"])
    print("Sidebands:   bpfi", fd["env_bpfi_sb"],  "bpfo", fd["env_bpfo_sb"],  "bsf", fd["env_bsf_sb"])

    td = infer_fault_location_time_domain(res)
    print("TD-only:", td["label"], f"(conf={td['confidence']:.2f})")

    # NEW: fused decision with Ball FD gate
    fused = infer_fault_location_fused(res, load_hp=load_hp)
    print("FUSED:", fused["label"], f"(TD conf={td['confidence']:.2f})", "-", fused.get("reason", ""))
