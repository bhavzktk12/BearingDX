# fd_freq_probe.py
import os, sys, re, numpy as np
from typing import Tuple, Dict
from bdxio.vp1 import parse_vibration_file, DEFAULT_SAMPLING_RATE
from bk1 import RPM_BY_LOAD, get_fault_frequencies
# we import internal helpers — fine for a probe
from features.fd import _rfft_mag, _bandpass, _envelope

def infer_load_from_name(path: str) -> int:
    m = re.search(r"(\d{3,5})(?=\.mat$)", os.path.basename(path))
    if m:
        rpm = int(m.group(1))
        return min(RPM_BY_LOAD.items(), key=lambda kv: abs(kv[1]-rpm))[0]
    m2 = re.search(r"normal_(\d)_\d+", os.path.basename(path), re.I)
    return int(m2.group(1)) if m2 else 1

def nearest_peak(freqs: np.ndarray, mag: np.ndarray, f0: float, tol_frac: float = 0.035) -> Tuple[float, float, float]:
    """Return (f_peak, delta_hz, peak_mag) within ±tol around f0 (0 if none)."""
    if f0 <= 0: return 0.0, 0.0, 0.0
    bw = f0 * tol_frac
    m = (freqs >= f0 - bw) & (freqs <= f0 + bw)
    if not np.any(m): return 0.0, 0.0, 0.0
    idx = np.argmax(mag[m])
    fwin = freqs[m]
    pwin = mag[m]
    f_peak = float(fwin[idx])
    return f_peak, abs(f_peak - f0), float(pwin[idx])

def main():
    if len(sys.argv) < 2:
        print("Usage: py fd_freq_probe.py <file.mat> [load_hp] [env_lo] [env_hi] [fs]")
        sys.exit(1)

    fpath    = sys.argv[1]
    load_hp  = int(sys.argv[2]) if len(sys.argv) >= 3 else infer_load_from_name(fpath)
    env_lo   = float(sys.argv[3]) if len(sys.argv) >= 4 else 2000.0
    env_hi   = float(sys.argv[4]) if len(sys.argv) >= 5 else 5000.0
    fs       = int(sys.argv[5]) if len(sys.argv) >= 6 else DEFAULT_SAMPLING_RATE

    rpm = RPM_BY_LOAD.get(load_hp, 1772)
    ff  = get_fault_frequencies(rpm)
    print(f"\nFile: {os.path.basename(fpath)} | Load={load_hp} → RPM={rpm} | fs={fs} Hz")
    print(f"Fault freqs (Hz): BPFI={ff['BPFI']:.2f}, BPFO={ff['BPFO']:.2f}, BSF={ff['BSF']:.2f}, FTF={ff['FTF']:.2f}")
    print(f"Envelope band: {env_lo:.0f}-{env_hi:.0f} Hz")

    parsed = parse_vibration_file(fpath, sensor_key="DE", sampling_rate=fs)
    if not parsed.get("success"):
        print("Parse failed:", parsed.get("errors")); sys.exit(2)

    # Take a few windows to probe
    segs = parsed["segments"]
    step = max(1, int(len(segs)/30))
    # Use first sampled window
    sig = np.asarray(segs[0]["normalized_signal"], float)

    # Base FFT
    f_fft, p_fft = _rfft_mag(sig, fs)

    # Envelope FFT
    try:
        banded = _bandpass(sig, fs, env_lo, env_hi, order=4)
    except Exception:
        banded = sig
    env = _envelope(banded)
    f_env, p_env = _rfft_mag(env, fs)

    def report(tag: str, freqs, mags, f0):
        fpk, dhz, pk = nearest_peak(freqs, mags, f0, tol_frac=0.035)
        # simple SNR: peak / median outside a small notch
        side = (freqs > max(0.5, f0*0.5)) & (freqs < min(fs*0.45, f0*3))
        noise = float(np.median(mags[side])) + 1e-12 if np.any(side) else float(np.median(mags) + 1e-12)
        snr = pk / noise
        print(f"{tag:<6} → peak {fpk:8.2f} Hz | Δ {dhz:6.2f} Hz | SNR {snr:6.2f}")
        return snr, dhz

    print("\nEnvelope peaks near targets:")
    snr_bpfi, d_bpfi = report("BPFI", f_env, p_env, ff["BPFI"])
    snr_bpfo, d_bpfo = report("BPFO", f_env, p_env, ff["BPFO"])
    snr_bsf,  d_bsf  = report("BSF",  f_env, p_env, ff["BSF"])
    snr_ftf,  d_ftf  = report("FTF",  f_env, p_env, ff["FTF"])

    print("\nRaw FFT peaks near targets (sanity):")
    report("BPFI", f_fft, p_fft, ff["BPFI"])
    report("BPFO", f_fft, p_fft, ff["BPFO"])
    report("BSF",  f_fft, p_fft, ff["BSF"])
    report("FTF",  f_fft, p_fft, ff["FTF"])

    # Quick verdicts
    winners: Dict[str, float] = {"IR": snr_bpfi, "OR": snr_bpfo, "Ball": snr_bsf}
    likely_key, likely_val = max(winners.items(), key=lambda kv: kv[1])

    print(f"\nLikely strongest (envelope): {likely_key}  | "
      f"SNRs: BPFI={snr_bpfi:.2f}, BPFO={snr_bpfo:.2f}, BSF={snr_bsf:.2f}")

    print("\nRule-of-thumb pass criteria:")
    print("  IR file → BPFI SNR should be the largest and Δ< ~5 Hz")
    print("  OR file → BPFO SNR largest and Δ< ~5 Hz")
    print("  Ball    → BSF competitive; may need sidebands/auto-band to win")
    print()

if __name__ == "__main__":
    main()
