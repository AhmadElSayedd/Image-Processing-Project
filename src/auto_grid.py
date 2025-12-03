# src/auto_grid.py
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

from config import (
    PREPROC_P2_DIR, PREPROC_P4_DIR, PREPROC_P8_DIR,
    ARTIFACTS_DIR, ensure_dirs
)

# Where we save overlays + accuracy logs
TILES_AUTO_DIR = os.path.join(ARTIFACTS_DIR, "tiles_auto")
os.makedirs(TILES_AUTO_DIR, exist_ok=True)


# ---------------------------------------------------------
# AUTO GRID DETECTION (Gradient Projection)
# ---------------------------------------------------------

def gradient_projection_scores(img_bgr, n_cuts):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype("float32")
    grad_x = np.sum(np.abs(np.diff(lab, axis=1)), axis=2)
    grad_y = np.sum(np.abs(np.diff(lab, axis=0)), axis=2)

    col_profile = np.sum(grad_x, axis=0)
    row_profile = np.sum(grad_y, axis=1)
    col_profile /= np.median(col_profile) + 1e-5
    row_profile /= np.median(row_profile) + 1e-5

    h, w = img_bgr.shape[:2]
    step_w = w / n_cuts
    step_h = h / n_cuts

    score = 0
    count = 0

    for i in range(1, n_cuts):
        if i % 2 == 0:
            continue

        idx_x = int(i * step_w)
        score += col_profile[max(idx_x-2, 0):idx_x+3].max()
        count += 1

        idx_y = int(i * step_h)
        score += row_profile[max(idx_y-2, 0):idx_y+3].max()
        count += 1

    return score / max(count, 1)


def detect_grid_size(img_bgr):
    score_8 = gradient_projection_scores(img_bgr, 8)
    score_4 = gradient_projection_scores(img_bgr, 4)

    if score_8 > 2.0:
        return 8
    elif score_4 > 2.5:
        return 4
    else:
        return 2


# ---------------------------------------------------------
# DRAW GRID + SAVE
# ---------------------------------------------------------

def save_grid_overlay(img_bgr, img_path, n_detected, out_subdir):
    os.makedirs(out_subdir, exist_ok=True)

    h, w = img_bgr.shape[:2]
    tile_h = h // n_detected
    tile_w = w // n_detected

    overlay = img_bgr.copy()
    for r in range(1, n_detected):
        cv2.line(overlay, (0, r*tile_h), (w, r*tile_h), (0,255,0), 2)
    for c in range(1, n_detected):
        cv2.line(overlay, (c*tile_w,0), (c*tile_w,h), (0,255,0), 2)

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_subdir, f"{base}_grid.png")
    cv2.imwrite(out_path, overlay)
    return out_path


# ---------------------------------------------------------
# MAIN — Accuracy + Saving correct & incorrect
# ---------------------------------------------------------

def run_auto_grid_eval_and_save():
    ensure_dirs()

    print("\n================================================")
    print(" AUTO GRID DETECTION → ACCURACY + OVERLAY SAVES ")
    print("================================================\n")

    tests = [
        (PREPROC_P2_DIR, 2),
        (PREPROC_P4_DIR, 4),
        (PREPROC_P8_DIR, 8),
    ]

    for preproc_dir, expected_n in tests:

        files = sorted(glob(os.path.join(preproc_dir, "*.jpg")) +
                       glob(os.path.join(preproc_dir, "*.png")))
        if not files:
            continue

        folder_name = os.path.basename(preproc_dir)
        print(f"\nChecking folder: {folder_name} (expected: {expected_n}×{expected_n})")

        correct_list = []
        wrong_list = []

        out_sub_correct = os.path.join(TILES_AUTO_DIR, f"{expected_n}x{expected_n}", "correct")
        out_sub_wrong   = os.path.join(TILES_AUTO_DIR, f"{expected_n}x{expected_n}", "wrong")

        for img_path in tqdm(files):
            img = cv2.imread(img_path)
            if img is None:
                continue

            detected_n = detect_grid_size(img)

            if detected_n == expected_n:
                correct_list.append(os.path.basename(img_path))
                save_grid_overlay(img, img_path, detected_n, out_sub_correct)
            else:
                wrong_list.append(f"{os.path.basename(img_path)} → {detected_n}x{detected_n}")
                save_grid_overlay(img, img_path, detected_n, out_sub_wrong)

        total = len(correct_list) + len(wrong_list)
        acc = 100.0 * len(correct_list) / max(total, 1)

        # Save text logs
        # Replace Unicode arrow to avoid encoding issues
        safe_wrong_list = [w.replace("→", "->") for w in wrong_list]

        with open(os.path.join(TILES_AUTO_DIR, f"{expected_n}x{expected_n}_correct.txt"),
                "w", encoding="utf-8") as f:
            f.write("\n".join(correct_list))

        with open(os.path.join(TILES_AUTO_DIR, f"{expected_n}x{expected_n}_wrong.txt"),
                "w", encoding="utf-8") as f:
            f.write("\n".join(safe_wrong_list))


        print(f"✔ Correct: {len(correct_list)}/{total} ({acc:.2f}%)")
        print(f"→ Logs saved in: {os.path.relpath(TILES_AUTO_DIR)}")

    print("\n[FINISHED] Accuracy evaluation + overlays saved successfully.\n")


if __name__ == "__main__":
    run_auto_grid_eval_and_save()
