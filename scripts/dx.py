import argparse, json
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from bdxio.vp1 import parse_vibration_file, DEFAULT_SAMPLING_RATE
from utils.naming import infer_load_from_name
from bk1 import RPM_BY_LOAD
from features.fd import compute_fd_indicators
from engine.fusion import infer_fault_location_fused

def cmd_fd_check(args):
    rpm = RPM_BY_LOAD.get(args.load, RPM_BY_LOAD[1]) if args.load is not None else RPM_BY_LOAD.get(infer_load_from_name(args.file), RPM_BY_LOAD[1])
    parsed = parse_vibration_file(args.file, sensor_key=args.sensor, sampling_rate=args.fs)
    fd = compute_fd_indicators(parsed, rpm, env_band=(args.env_lo, args.env_hi) if args.env_lo and args.env_hi else "auto")
    keys = ("fft_bpfi_norm","fft_bpfo_norm","fft_bsf_norm","fft_ftf_norm",
            "env_bpfi_snr","env_bpfo_snr","env_bsf_snr","env_ftf_snr",
            "env_bpfi_sb","env_bpfo_sb","env_bsf_sb")
    slim = {k: round(float(fd.get(k, 0.0)), 3) for k in keys}
    print(json.dumps(slim, indent=2))

def cmd_smoke(args):
    from tests.smoke_all import run
    run(args.root, args.save, args.perload)

def main():
    ap = argparse.ArgumentParser(prog="dx")
    sub = ap.add_subparsers(dest="cmd", required=True)

    fd = sub.add_parser("fd-check", help="Inspect one file's FD indicators")
    fd.add_argument("file"); fd.add_argument("--load", type=int); fd.add_argument("--sensor", default="DE")
    fd.add_argument("--fs", type=int, default=DEFAULT_SAMPLING_RATE)
    fd.add_argument("--env-lo", type=float); fd.add_argument("--env-hi", type=float)
    fd.set_defaults(func=cmd_fd_check)

    sm = sub.add_parser("smoke", help="Dataset-wide fused smoke")
    sm.add_argument("--root", required=True); sm.add_argument("--save"); sm.add_argument("--perload", action="store_true")
    sm.set_defaults(func=cmd_smoke)

    args = ap.parse_args(); args.func(args)

if __name__ == "__main__":
    main()
