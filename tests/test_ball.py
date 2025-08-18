
# test_ball.py
"""
Quick probe for ball-fault-focused FD metrics and decision.
Usage:
  py test_ball.py <file.mat> [load_hp] [fs]
Prints: chosen env band, SNRs, sidebands, composite score, and final decision.
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(__file__)); sys.path.insert(0, ROOT)
import os, sys, re, numpy as np
from bdxio.vp1 import parse_vibration_file, DEFAULT_SAMPLING_RATE
from bk1 import RPM_BY_LOAD, get_fault_frequencies
from features.fd import compute_fd_indicators, decide_ball_fault_fd, _ball_fd_metrics

def infer_load_from_name(path: str) -> int:
    m = re.search(r"(\d{3,5})(?=\.mat$)", os.path.basename(path))
    if m:
        rpm = int(m.group(1))
        return min(RPM_BY_LOAD.items(), key=lambda kv: abs(kv[1]-rpm))[0]
    m2 = re.search(r"normal_(\d)_\d+", os.path.basename(path), re.I)
    return int(m2.group(1)) if m2 else 1

def main():
    if len(sys.argv) < 2:
        print("Usage: py test_ball.py <file.mat> [load_hp] [fs]")
        sys.exit(1)

    fpath = sys.argv[1]
    load_hp = int(sys.argv[2]) if len(sys.argv) >= 3 else infer_load_from_name(fpath)
    fs = int(sys.argv[3]) if len(sys.argv) >= 4 else DEFAULT_SAMPLING_RATE

    rpm = RPM_BY_LOAD.get(load_hp, 1772)
    ff = get_fault_frequencies(rpm)
    print(f"\nFile: {os.path.basename(fpath)} | Load={load_hp} → RPM={rpm} | fs={fs} Hz")
    print(f"Fault freqs (Hz): BPFI={ff['BPFI']:.2f}, BPFO={ff['BPFO']:.2f}, BSF={ff['BSF']:.2f}, FTF={ff['FTF']:.2f}")

    parsed = parse_vibration_file(fpath, sensor_key="DE", sampling_rate=fs)
    if not parsed.get("success"):
        print("Parse failed:", parsed.get("errors")); sys.exit(2)

    fd = compute_fd_indicators(parsed, rpm, env_band='auto')

    print("\nChosen env band:", round(fd.get('env_band_lo',0.0),1), "→", round(fd.get('env_band_hi',0.0),1), "Hz")
    print("FFT norms:    bpfi={:.3f}  bpfo={:.3f}  bsf={:.3f}  ftf={:.3f}".format(
        fd.get("fft_bpfi_norm",0.0), fd.get("fft_bpfo_norm",0.0), fd.get("fft_bsf_norm",0.0), fd.get("fft_ftf_norm",0.0)))
    print("Envelope SNR: bpfi={:.2f}  bpfo={:.2f}  bsf={:.2f}  ftf={:.2f}".format(
        fd.get("env_bpfi_snr",0.0), fd.get("env_bpfo_snr",0.0), fd.get("env_bsf_snr",0.0), fd.get("env_ftf_snr",0.0)))
    print("Sidebands:    bpfi={:.3f}  bpfo={:.3f}  bsf={:.3f}".format(
        fd.get("env_bpfi_sb",0.0), fd.get("env_bpfo_sb",0.0), fd.get("env_bsf_sb",0.0)))

    # Ball-focused metrics & decision
    metrics = _ball_fd_metrics(fd)
    decision = decide_ball_fault_fd(fd, rpm, thresholds=None)

    print("\nBall metrics:  BSF_SNR={:.2f}  FTF_SNR={:.2f}  SB_ratio={:.2f}  Harm_ratio={:.2f}  (variant={})".format(
        metrics["ball_bsf_snr"], metrics["ball_ftf_snr"], metrics["ball_sideband_ratio"], metrics["ball_harmonic_ratio"], "2x" if metrics.get("ball_used_2x") else "1x"))
    print("Composite ball_fd_score = {:.2f}".format(metrics["ball_fd_score"]))

    verdict = "BALL" if decision["is_ball"] else "NOT-BALL"
    print("\nFinal decision:", verdict, "| score={:.2f} | gates:".format(decision["score"]), decision["gates"], "| reason:", decision["reason"])

if __name__ == "__main__":
    main()
