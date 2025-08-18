from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Union, cast
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from numpy.fft import rfft, irfft
from bk1 import get_fault_frequencies

# Candidate envelope bands to try (auto-select)
CAND_ENV_BANDS: List[Tuple[float, float]] = [
    (1500.0, 4000.0), (1800.0, 4200.0), (2000.0, 5000.0), (2300.0, 5500.0)
]

# None / "auto" / list[(lo,hi)] / (lo,hi)
EnvBand = Optional[Union[Tuple[float, float], List[Tuple[float, float]], str]]

# ----------------- low-level helpers -----------------

def _prewhiten_signal(x: np.ndarray, fs: float, smooth_bins: int = 41) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = int(len(x))
    if n < 32 or fs <= 0:
        return x
    try:
        w = np.hanning(n)
        X = rfft(x * w)
        mag = np.abs(X) + 1e-12
        log_mag = np.log(mag)
        k = max(5, int(smooth_bins) | 1)
        kernel = np.ones(k, dtype=float) / k
        pad = np.pad(log_mag, (k // 2, k // 2), mode="edge")
        smooth = np.convolve(pad, kernel, mode="same")[k // 2 : -k // 2]
        W = np.exp(smooth)
        Xw = X / (W + 1e-12)
        return irfft(Xw, n=n).astype(float)
    except Exception:
        return x

def _rfft_mag(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    n = int(len(x))
    if n <= 0 or fs <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    w = np.hanning(n).astype(float)
    xw = x * w
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    mag = np.abs(X) / (np.sum(w) / 2.0 + 1e-12)
    return freqs, mag

def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    ny = 0.5 * float(fs)
    lo = max(5.0, float(lo))
    hi = min(ny - 1.0, float(hi))
    if hi <= lo:
        # widen a touch to keep butter happy
        mid = 0.5 * (lo + hi)
        lo = max(5.0, mid * 0.9)
        hi = min(ny - 1.0, mid * 1.1)
    # be explicit about output='ba' so butter returns (b, a), not (z, p, k)
    b, a = cast(Tuple[np.ndarray, np.ndarray],
                butter(int(order), [lo / ny, hi / ny], btype="band", output="ba"))
    return filtfilt(b, a, x)


def _envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(hilbert(np.asarray(x, dtype=float))))

def _sum_harmonic_energy(
    freqs: np.ndarray, mag: np.ndarray, f0: float, n_harm: int = 5, tol_frac: float = 0.02
) -> float:
    if f0 <= 0:
        return 0.0
    total = 0.0
    for k in range(1, n_harm + 1):
        fk = k * f0
        bw = fk * tol_frac
        mask = (freqs >= fk - bw) & (freqs <= fk + bw)
        if np.any(mask):
            total += float(np.mean(mag[mask]))
    return total

def _sum_sideband_energy(
    freqs: np.ndarray,
    mag: np.ndarray,
    carrier: float,
    mod: float,
    n_harm: int = 3,
    n_sb: int = 2,
    tol_frac: float = 0.02,
) -> float:
    if carrier <= 0 or mod <= 0:
        return 0.0
    total = 0.0
    for k in range(1, n_harm + 1):
        f0 = k * carrier
        for m in range(1, n_sb + 1):
            for f in (f0 - m * mod, f0 + m * mod):
                if f <= 0:
                    continue
                bw = max(1.0, f * tol_frac)
                mask = (freqs >= f - bw) & (freqs <= f + bw)
                if np.any(mask):
                    total += float(np.mean(mag[mask]))
    return total

# ----------------- envelope band chooser -----------------

def _choose_env_band(
    signal: np.ndarray,
    fs: float,
    rpm: float,
    bands: Optional[List[Tuple[float, float]]] = None,
    prewhiten: bool = True,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    if not bands:
        bands = CAND_ENV_BANDS

    lo_fb, hi_fb = bands[0]
    try:
        banded_fb = _bandpass(signal, fs, lo_fb, hi_fb, order=4)
    except Exception:
        banded_fb = signal
    env_fb = _envelope(banded_fb)
    ef_fb, em_fb = _rfft_mag(env_fb, fs)

    best_lo, best_hi = float(lo_fb), float(hi_fb)
    best_ef, best_em = ef_fb, em_fb
    best_score = -1.0

    ff = get_fault_frequencies(rpm)
    shaft, bpfi, bpfo, bsf, ftf = (
        ff["shaft_freq"],
        ff["BPFI"],
        ff["BPFO"],
        ff["BSF"],
        ff["FTF"],
    )

    def snr_at(ef: np.ndarray, em: np.ndarray, f0: float) -> float:
        if f0 <= 0:
            return 0.0
        bw = max(1.0, f0 * 0.02)
        mask = (ef >= f0 - bw) & (ef <= f0 + bw)
        peak = float(np.max(em[mask])) if np.any(mask) else 0.0
        side = (ef > max(0.5, f0 * 0.5)) & (ef < min(fs * 0.45, f0 * 3))
        noise = float(np.median(em[side])) + 1e-12
        return peak / noise

    for (lo, hi) in bands:
        base = _prewhiten_signal(signal, fs) if prewhiten else signal
        try:
            banded = _bandpass(base, fs, lo, hi, order=4)
        except Exception:
            banded = base
        env = _envelope(banded)
        ef, em = _rfft_mag(env, fs)
        score = (
            1.0 * snr_at(ef, em, bsf)
            + 0.6 * snr_at(ef, em, ftf)
            + 0.3 * snr_at(ef, em, 2 * bsf)
            - 0.2 * snr_at(ef, em, bpfi)
            - 0.2 * snr_at(ef, em, bpfo)
        )
        if score > best_score:
            best_score = score
            best_lo, best_hi = float(lo), float(hi)
            best_ef, best_em = ef, em

    return float(best_lo), float(best_hi), best_ef, best_em

# ----------------- mid-level ratios -----------------

def _spectral_ratios(
    signal: np.ndarray,
    fs: float,
    rpm: float,
    env_band: EnvBand = (2000.0, 5000.0),
    prewhiten: bool = True,
) -> Dict[str, float]:
    ff = get_fault_frequencies(rpm)
    shaft, bpfi, bpfo, bsf, ftf = (
        ff["shaft_freq"],
        ff["BPFI"],
        ff["BPFO"],
        ff["BSF"],
        ff["FTF"],
    )

    # FFT (broadband normalization)
    freqs, mag = _rfft_mag(signal, fs)
    broadband = float(np.median(mag[(freqs > 50) & (freqs < 0.45 * fs)])) + 1e-12

    def norm_sum(f0: float) -> float:
        return _sum_harmonic_energy(freqs, mag, f0) / broadband

    fft_bpfi = norm_sum(bpfi)
    fft_bpfo = norm_sum(bpfo)
    fft_bsf = norm_sum(bsf)
    fft_ftf = norm_sum(ftf)

    # Envelope spectrum selection
    is_auto = (
        (env_band is None)
        or (isinstance(env_band, str) and env_band.lower() == "auto")
        or (
            isinstance(env_band, list)
            and len(env_band) > 0
            and isinstance(env_band[0], (tuple, list))
        )
    )

    if is_auto:
        lo_f, hi_f, ef, em = _choose_env_band(
            signal,
            fs,
            rpm,
            bands=(env_band if isinstance(env_band, list) else CAND_ENV_BANDS),
            prewhiten=prewhiten,
        )
    elif isinstance(env_band, tuple) and len(env_band) == 2:
        lo_f, hi_f = float(env_band[0]), float(env_band[1])
        base = _prewhiten_signal(signal, fs) if prewhiten else signal
        try:
            banded = _bandpass(base, fs, lo_f, hi_f, order=4)
        except Exception:
            banded = base
        env = _envelope(banded)
        ef, em = _rfft_mag(env, fs)
    else:
        raise ValueError("env_band must be None/'auto'/list[(lo,hi)] or a (lo,hi) tuple")

    def env_peak_and_snr(
        f0: float, tol_frac: float = 0.02, local_bw: float = 0.25, min_bins: int = 8
    ) -> Tuple[float, float]:
        if f0 <= 0:
            return 0.0, 0.0
        # frequency resolution
        df = (ef[1] - ef[0]) if len(ef) > 1 else 1.0
        # ensure >= 1 bin in the peak window
        desired = max(1.0, f0 * tol_frac)
        half_bins = max(1, int(np.ceil(desired / df)))
        bw = half_bins * df
        peak_mask = (ef >= f0 - bw) & (ef <= f0 + bw)
        peak = float(np.max(em[peak_mask])) if np.any(peak_mask) else 0.0
        # local noise window
        half_span = max(f0 * local_bw, (min_bins // 2 + 1) * df)
        lo = max(0.5, f0 - half_span)
        hi = min(ef[-1] * 0.95, f0 + half_span)
        local_mask = (ef >= lo) & (ef <= hi) & (~peak_mask)
        if np.count_nonzero(local_mask) < min_bins:
            lo = max(0.5, f0 - 3 * half_span)
            hi = min(ef[-1] * 0.95, f0 + 3 * half_span)
            local_mask = (ef >= lo) & (ef <= hi) & (~peak_mask)
        band_vals = em[local_mask] if np.any(local_mask) else em
        n = band_vals.size
        if n <= 2:
            noise = float(np.median(band_vals)) + 1e-12
        else:
            k = max(1, int(0.10 * n))  # drop top 10%
            trimmed = np.partition(band_vals, -k)[:-k] if k < n else band_vals
            noise = float(np.median(trimmed)) + 1e-12
        return peak, (peak / noise if noise > 0 else 0.0)

    env_bpfi_pk, env_bpfi_snr = env_peak_and_snr(bpfi)
    env_bpfo_pk, env_bpfo_snr = env_peak_and_snr(bpfo)
    env_bsf_pk, env_bsf_snr = env_peak_and_snr(bsf)
    env_bsf2_pk, env_bsf2_snr = env_peak_and_snr(2 * bsf)
    env_ftf_pk, env_ftf_snr = env_peak_and_snr(ftf)

    # Sidebands on envelope spectrum
    sb_bpfi = _sum_sideband_energy(ef, em, bpfi, shaft, n_harm=3, n_sb=2, tol_frac=0.02)
    sb_bpfo = _sum_sideband_energy(ef, em, bpfo, shaft, n_harm=3, n_sb=2, tol_frac=0.02)
    sb_bsf = _sum_sideband_energy(ef, em, bsf, ftf, n_harm=3, n_sb=2, tol_frac=0.02)
    sb_bsf2 = _sum_sideband_energy(ef, em, 2 * bsf, ftf, n_harm=2, n_sb=2, tol_frac=0.02)

    return {
        # FFT norms
        "fft_bpfi_norm": float(fft_bpfi),
        "fft_bpfo_norm": float(fft_bpfo),
        "fft_bsf_norm": float(fft_bsf),
        "fft_ftf_norm": float(fft_ftf),
        # envelope peaks / SNR
        "env_bpfi_pk": float(env_bpfi_pk),
        "env_bpfi_snr": float(env_bpfi_snr),
        "env_bpfo_pk": float(env_bpfo_pk),
        "env_bpfo_snr": float(env_bpfo_snr),
        "env_bsf_pk": float(env_bsf_pk),
        "env_bsf2_pk": float(env_bsf2_pk),
        "env_bsf_snr": float(env_bsf_snr),
        "env_bsf2_snr": float(env_bsf2_snr),
        "env_ftf_pk": float(env_ftf_pk),
        "env_ftf_snr": float(env_ftf_snr),
        # sidebands
        "env_bpfi_sb": float(sb_bpfi),
        "env_bpfo_sb": float(sb_bpfo),
        "env_bsf_sb": float(sb_bsf),
        "env_bsf2_sb": float(sb_bsf2),
        # chosen band
        "env_band_lo": float(lo_f),
        "env_band_hi": float(hi_f),
    }

# ----------------- ball metrics & gate -----------------

def _normalize01(v: float, scale: float) -> float:
    return float(max(0.0, min(1.0, v / max(1e-9, scale))))

def _ball_fd_metrics(fd: Dict[str, float]) -> Dict[str, Any]:
    bsf1_snr = float(fd.get("env_bsf_snr", 0.0))
    bsf2_snr = float(fd.get("env_bsf2_snr", 0.0))
    use_2x = bsf2_snr > bsf1_snr * 1.05
    bsf_snr = bsf2_snr if use_2x else bsf1_snr
    ftf_snr = float(fd.get("env_ftf_snr", 0.0))
    sb_bsf = float(fd.get("env_bsf2_sb" if use_2x else "env_bsf_sb", 0.0))
    sb_bpfi = float(fd.get("env_bpfi_sb", 0.0))
    sb_bpfo = float(fd.get("env_bpfo_sb", 0.0))
    fft_bsf = float(fd.get("fft_bsf_norm", 0.0))
    fft_bpfi = float(fd.get("fft_bpfi_norm", 0.0))
    fft_bpfo = float(fd.get("fft_bpfo_norm", 0.0))
    denom_sb = max(sb_bpfi, sb_bpfo, 1e-12)
    sideband_ratio = sb_bsf / denom_sb
    harmonic_ratio = (fft_bsf) / (fft_bpfi + fft_bpfo + 1e-12)
    score = (
        0.50 * _normalize01(bsf_snr, 4.0)
        + 0.25 * _normalize01(ftf_snr, 3.0)
        + 0.15 * _normalize01(sideband_ratio, 1.3)
        + 0.10 * _normalize01(harmonic_ratio, 1.2)
    )
    return {
        "ball_bsf_snr": bsf_snr,
        "ball_ftf_snr": ftf_snr,
        "ball_sideband_ratio": sideband_ratio,
        "ball_harmonic_ratio": harmonic_ratio,
        "ball_fd_score": score,
        "ball_used_2x": bool(use_2x),
        "ball_bsf_variant": ("2x" if use_2x else "1x"),
    }

DEFAULT_BALL_THRESHOLDS = {
    "min_bsf_snr": 2.5,
    "min_ftf_snr": 1.3,
    "min_sb_ratio": 1.05,
    "min_score": 0.60,
}

def decide_ball_fault_fd(
    fd: Dict[str, float], rpm: float, thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    th = dict(DEFAULT_BALL_THRESHOLDS)
    th.update(thresholds or {})
    m = _ball_fd_metrics(fd)
    gates = {
        "bsf_snr_ok": bool(m["ball_bsf_snr"] >= th["min_bsf_snr"]),
        "ftf_snr_ok": bool(m["ball_ftf_snr"] >= th["min_ftf_snr"]),
        "sb_ratio_ok": bool(m["ball_sideband_ratio"] >= th["min_sb_ratio"]),
    }
    score = float(m["ball_fd_score"])
    is_ball = all(gates.values()) and (score >= th["min_score"])
    fails = [k for k, v in gates.items() if not v]
    reason = "pass" if is_ball else ("gates_failed:" + ",".join(fails) if fails else f"score<{th['min_score']}")
    # NOTE: if you want the softer paths you had in dx1.py, we can port them here verbatim.
    return {"is_ball": is_ball, "score": score, "gates": gates, "metrics": m, "reason": reason}

# ----------------- file-level FD aggregation -----------------

def compute_fd_indicators(
    parsed_result: Dict[str, Any], rpm: float, env_band: EnvBand = None
) -> Dict[str, Any]:
    segs = parsed_result.get("segments", [])
    if not segs:
        zeros = {
            k: 0.0
            for k in (
                "fft_bpfi_norm",
                "fft_bpfo_norm",
                "fft_bsf_norm",
                "fft_ftf_norm",
                "env_bpfi_pk",
                "env_bpfi_snr",
                "env_bpfo_pk",
                "env_bpfo_snr",
                "env_bsf_pk",
                "env_bsf_snr",
                "env_ftf_pk",
                "env_ftf_snr",
                "env_bpfi_sb",
                "env_bpfo_sb",
                "env_bsf_sb",
                "env_band_lo",
                "env_band_hi",
                "ball_bsf_snr",
                "ball_ftf_snr",
                "ball_sideband_ratio",
                "ball_harmonic_ratio",
                "ball_fd_score",
            )
        }
        return zeros

    fs = segs[0].get("sampling_rate", 12000)
    segs_sorted = sorted(segs, key=lambda s: s.get("quality_score", 0.0), reverse=True)
    segs_use = segs_sorted[: max(8, len(segs) // 4)]

    vals: List[Dict[str, float]] = []
    for seg in segs_use:
        sig = np.asarray(seg.get("signal") if "signal" in seg else seg["normalized_signal"], dtype=float)
        vals.append(_spectral_ratios(sig, fs, rpm, env_band=env_band, prewhiten=True))

    # presence/proportion helpers for small-ball soft logic if you want it later
    bsf2_snrs = [v.get("env_bsf2_snr", 0.0) for v in vals]
    presence_bsf2 = float(np.mean(np.array(bsf2_snrs) >= 1.4)) if vals else 0.0
    bsf1_snrs = [v.get("env_bsf_snr", 0.0) for v in vals]
    presence_bsf1 = float(np.mean(np.array(bsf1_snrs) >= 1.6)) if vals else 0.0

    out: Dict[str, Any] = {}
    for k in vals[0].keys():
        arr = [v[k] for v in vals]
        out[k] = float(np.percentile(arr, 75))

    out.update(_ball_fd_metrics(out))
    out["ball_bsf2_presence"] = presence_bsf2
    out["ball_bsf1_presence"] = presence_bsf1
    return out
