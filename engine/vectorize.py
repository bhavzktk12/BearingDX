# engine/vectorize.py
from typing import Any, Dict
import numpy as np
# engine/vectorize.py
from features.fd import compute_fd_indicators, _ball_fd_metrics
from features.td import td_file_level_vector


def file_level_vector_td_fd(parsed_result: Dict[str, Any], rpm: float) -> np.ndarray:
    """
    File-level TD+FD vector for training.
    TD: stats + indicators (dx1._file_level_vector_for_ml)
    FD: FFT & envelope peaks/SNRs/sidebands (+2xBSF) + Ball-focused metrics
    """
    td = td_file_level_vector(parsed_result)  # shape (1, N_td)
    fd = compute_fd_indicators(parsed_result, rpm, env_band=None)

    fd_keys = [
        "fft_bpfi_norm","fft_bpfo_norm","fft_bsf_norm","fft_ftf_norm",
        "env_bpfi_pk","env_bpfi_snr","env_bpfo_pk","env_bpfo_snr",
        "env_bsf_pk","env_bsf_snr","env_bsf2_pk","env_bsf2_snr",
        "env_ftf_pk","env_ftf_snr","env_bpfi_sb","env_bpfo_sb","env_bsf_sb","env_bsf2_sb"
    ]
    fd_vec = np.array([float(fd.get(k, 0.0)) for k in fd_keys], dtype=float).reshape(1, -1)

    # Ball-focused metrics (must exist inside dx1.compute_fd_indicators result or helpers)
    # If you already had _ball_fd_metrics in dx1, import and use it here.
    # Compute consistent ball metrics from FD indicators
    m = _ball_fd_metrics(fd)

    ball_vec = np.array([
        float(m.get("ball_bsf_snr", 0.0)),
        float(m.get("ball_ftf_snr", 0.0)),
        float(m.get("ball_sideband_ratio", 0.0)),
        float(m.get("ball_harmonic_ratio", 0.0)),
        float(m.get("ball_fd_score", 0.0)),
        float(1.0 if m.get("ball_used_2x", False) else 0.0),
    ], dtype=float).reshape(1, -1)

    return np.concatenate([td, fd_vec, ball_vec], axis=1)
