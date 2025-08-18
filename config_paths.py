# config_paths.py  (place in repo root)
import os

# Resolve project root (folder that contains this file)
PROJECT_ROOT = os.path.dirname(__file__)

# ---- Data roots (override with env vars if you like) ----
CWRU_TRAIN_ROOT = os.environ.get(
    "CWRU_TRAIN_ROOT",
    r"C:\AI STUFF\BearingDX\data\training_data\CWRU_train"
)

# Sampling rate default
DEFAULT_FS = int(os.environ.get("BEARING_FS", "12000"))

# Convenience joiner so you can do: p("Ball","B007_2_1750.mat")
def p(*parts):  # path helper
    return os.path.join(CWRU_TRAIN_ROOT, *parts)

# Handy lists/globs if you need them
NORMAL_DIR = os.path.join(CWRU_TRAIN_ROOT, "Normal")
IR_DIR     = os.path.join(CWRU_TRAIN_ROOT, "IR")
OR_DIR     = os.path.join(CWRU_TRAIN_ROOT, "OR")
BALL_DIR   = os.path.join(CWRU_TRAIN_ROOT, "Ball")
