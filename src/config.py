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

# ======================
# Enhanced Tiles (Phase-1 report use only)
# ======================
TILES_ENH_DIR = os.path.join(ARTIFACTS_DIR, "tiles_enhanced")
TILES_ENH_2_DIR = os.path.join(TILES_ENH_DIR, "2x2")
TILES_ENH_4_DIR = os.path.join(TILES_ENH_DIR, "4x4")
TILES_ENH_8_DIR = os.path.join(TILES_ENH_DIR, "8x8")

# ======================
# Report Visuals subfolders
# ======================
VISUALS_TILES_BA_DIR = os.path.join(VISUALS_DIR, "tiles_before_after")
VISUALS_TILES_EDGES_DIR = os.path.join(VISUALS_DIR, "tiles_edges")

# 📌 NEW: Contour Tile Output Folders
TILES_CNT_DIR = os.path.join(ARTIFACTS_DIR, "tiles_contours")
TILES_CNT_2_DIR = os.path.join(TILES_CNT_DIR, "2x2")
TILES_CNT_4_DIR = os.path.join(TILES_CNT_DIR, "4x4")
TILES_CNT_8_DIR = os.path.join(TILES_CNT_DIR, "8x8")

# 📌 NEW: Visual Reports for Contour Tiles
VISUALS_TILES_CNT_DIR = os.path.join(VISUALS_DIR, "tiles_contours")


def ensure_dirs():
    dirs = [
        ARTIFACTS_DIR, VISUALS_DIR,

        # original tiles
        TILES_2_DIR, TILES_4_DIR, TILES_8_DIR,

        # enhanced tiles
        TILES_ENH_DIR, TILES_ENH_2_DIR, TILES_ENH_4_DIR, TILES_ENH_8_DIR,

        # 📌 NEW contour folders
        TILES_CNT_DIR, TILES_CNT_2_DIR, TILES_CNT_4_DIR, TILES_CNT_8_DIR,

        # 📌 visual folders
        VISUALS_TILES_BA_DIR, VISUALS_TILES_EDGES_DIR, VISUALS_TILES_CNT_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

