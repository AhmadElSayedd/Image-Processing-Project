import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from config import (
    CORRECT_DIR, PUZZLES_2_DIR, PUZZLES_4_DIR, PUZZLES_8_DIR,
    TILES_2_DIR, TILES_4_DIR, TILES_8_DIR,
    PREPROC_P2_DIR, PREPROC_P4_DIR, PREPROC_P8_DIR,
    TILES_ENH_2_DIR, TILES_ENH_4_DIR, TILES_ENH_8_DIR,
    TILES_CNT_2_DIR, TILES_CNT_4_DIR, TILES_CNT_8_DIR,
    VISUALS_TILES_BA_DIR, VISUALS_TILES_CNT_DIR,
    ensure_dirs
)

# ===================================
# Enhancement steps
# ===================================

def equalize_luminance(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_c = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_c,a,b]), cv2.COLOR_LAB2BGR)

def sharpen(img):
    blur = cv2.GaussianBlur(img, (0,0), 3)
    return cv2.addWeighted(img, 1.3, blur, -0.3, 0)

def preprocess(img):
    img = equalize_luminance(img)
    img = apply_clahe(img)
    img = sharpen(img)
    return img

# ===================================
# Contour Extraction
# ===================================

def extract_contours(tile):
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 120, 240)
    return edges

# ===================================
# Segmentation
# ===================================

def segment_and_save(puzzle_path, tiles_dir, N):
    img = cv2.imread(puzzle_path)
    if img is None: return

    h,w = img.shape[:2]
    th, tw = h//N, w//N
    base = os.path.splitext(os.path.basename(puzzle_path))[0]

    for r in range(N):
        for c in range(N):
            tile = img[r*th:(r+1)*th, c*tw:(c+1)*tw]
            out = os.path.join(tiles_dir, f"{base}_r{r}_c{c}.png")
            cv2.imwrite(out, tile)
            yield out, tile

def save_side_by_side(a, b, out_path):
    both = np.hstack((a, b))
    cv2.imwrite(out_path, both)

# ===================================
# Full Pipeline
# ===================================

def run_preprocessing():
    ensure_dirs()
    folders = [
        (PUZZLES_2_DIR, PREPROC_P2_DIR, TILES_2_DIR, TILES_ENH_2_DIR, TILES_CNT_2_DIR, 2),
        (PUZZLES_4_DIR, PREPROC_P4_DIR, TILES_4_DIR, TILES_ENH_4_DIR, TILES_CNT_4_DIR, 4),
        (PUZZLES_8_DIR, PREPROC_P8_DIR, TILES_8_DIR, TILES_ENH_8_DIR, TILES_CNT_8_DIR, 8)
    ]

    for input_dir, preproc_dir, tiles_orig, tiles_enh, tiles_cnt, N in folders:
        imgs = sorted(glob(os.path.join(input_dir, "*.jpg")) +
                      glob(os.path.join(input_dir, "*.png")))
        if not imgs: continue

        print(f"\n[PHASE1] Processing {N}x{N} puzzles ({len(imgs)} images)")
        os.makedirs(preproc_dir, exist_ok=True)
        os.makedirs(tiles_orig, exist_ok=True)
        os.makedirs(tiles_enh, exist_ok=True)
        os.makedirs(tiles_cnt, exist_ok=True)

        for img_path in tqdm(imgs):
            img = cv2.imread(img_path)
            if img is None: continue

            # preprocess full puzzle
            img_enh = preprocess(img)
            cv2.imwrite(os.path.join(preproc_dir, os.path.basename(img_path)), img_enh)

            # segment + save all versions
            for (orig_path, orig_tile) in segment_and_save(img_path, tiles_orig, N):
                fname = os.path.basename(orig_path)
                r = int(fname.split("_r")[1].split("_")[0])
                c = int(fname.split("_c")[1].split(".")[0])
                h,w = orig_tile.shape[:2]

                enh_tile = img_enh[r*h:(r+1)*h, c*w:(c+1)*w]
                cv2.imwrite(os.path.join(tiles_enh, fname), enh_tile)

                cnt_tile = extract_contours(enh_tile)
                cv2.imwrite(os.path.join(tiles_cnt, fname), cnt_tile)

                # save visuals for report
                save_side_by_side(orig_tile, enh_tile,
                                  os.path.join(VISUALS_TILES_BA_DIR, f"{N}x{N}_{fname}_BA.png"))
                save_side_by_side(orig_tile, cv2.cvtColor(cnt_tile, cv2.COLOR_GRAY2BGR),
                                  os.path.join(VISUALS_TILES_CNT_DIR, f"{N}x{N}_{fname}_CNT.png"))

if __name__ == "__main__":
    run_preprocessing()
