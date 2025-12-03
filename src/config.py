# src/config.py
import os

# ---- Root Paths ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "Gravity Falls")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# Input folders
CORRECT_DIR = os.path.join(DATA_DIR, "correct")
PUZZLES_2_DIR = os.path.join(DATA_DIR, "puzzle_2x2")
PUZZLES_4_DIR = os.path.join(DATA_DIR, "puzzle_4x4")
PUZZLES_8_DIR = os.path.join(DATA_DIR, "puzzle_8x8")

# Preprocessed outputs
PREPROC_DIR = os.path.join(ARTIFACTS_DIR, "preprocessed")
PREPROC_CORRECT_DIR = os.path.join(PREPROC_DIR, "correct")
PREPROC_P2_DIR = os.path.join(PREPROC_DIR, "puzzle_2x2")
PREPROC_P4_DIR = os.path.join(PREPROC_DIR, "puzzle_4x4")
PREPROC_P8_DIR = os.path.join(PREPROC_DIR, "puzzle_8x8")

# Tile outputs
TILES_DIR = os.path.join(ARTIFACTS_DIR, "tiles")
TILES_2_DIR = os.path.join(TILES_DIR, "2x2")
TILES_4_DIR = os.path.join(TILES_DIR, "4x4")
TILES_8_DIR = os.path.join(TILES_DIR, "8x8")

# Visuals (before/after comparisons, etc.)
VISUALS_DIR = os.path.join(ARTIFACTS_DIR, "visuals")


def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [
        DATA_DIR,
        ARTIFACTS_DIR,
        PREPROC_DIR,
        PREPROC_CORRECT_DIR,
        PREPROC_P2_DIR,
        PREPROC_P4_DIR,
        PREPROC_P8_DIR,
        TILES_DIR,
        TILES_2_DIR,
        TILES_4_DIR,
        TILES_8_DIR,
        VISUALS_DIR,
    ]:
        os.makedirs(d, exist_ok=True)
