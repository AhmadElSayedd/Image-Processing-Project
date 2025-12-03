# src/preprocess.py
import os
import cv2
import numpy as np
from typing import Tuple, Optional
from glob import glob
from tqdm import tqdm

from config import (
    CORRECT_DIR, PUZZLES_2_DIR, PUZZLES_4_DIR, PUZZLES_8_DIR,
    PREPROC_CORRECT_DIR, PREPROC_P2_DIR, PREPROC_P4_DIR, PREPROC_P8_DIR,
    VISUALS_DIR, ensure_dirs
)

# -----------------------------
# Core preprocessing functions
# -----------------------------

def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_if_needed(img: np.ndarray,
                     max_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Optionally resize image so that max(H, W) <= max_size.
    If max_size is None, return as is.
    """
    if max_size is None:
        return img

    max_h, max_w = max_size
    h, w = img.shape[:2]

    scale = min(max_h / h, max_w / w, 1.0)  # don't upscale
    if scale == 1.0:
        return img

    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def equalize_luminance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR -> YCrCb, equalize Y channel, convert back.
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


def denoise(img_bgr: np.ndarray, method: str = "gaussian") -> np.ndarray:
    if method == "gaussian":
        return cv2.GaussianBlur(img_bgr, (5, 5), 0)
    elif method == "bilateral":
        # Bilateral filter preserves edges better
        return cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        return img_bgr


def sharpen(img_bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Simple unsharp masking.
    """
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 3)
    sharp = cv2.addWeighted(img_bgr, 1 + strength, blur, -strength, 0)
    return sharp


def preprocess_image(
    img_bgr: np.ndarray,
    resize_to: Optional[Tuple[int, int]] = None,
    denoise_method: str = "gaussian",
    sharpen_strength: float = 0.3
) -> np.ndarray:
    """
    Full preprocessing pipeline for one image.
    """
    img = img_bgr.copy()

    # 1) Optional resize
    if resize_to is not None:
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)

    # 2) Equalize luminance
    img = equalize_luminance(img)

    # 3) Denoise
    img = denoise(img, method=denoise_method)

    # 4) Optional mild sharpening
    if sharpen_strength > 0:
        img = sharpen(img, strength=sharpen_strength)

    return img


# ---------------------------------
# Batch processing helper function
# ---------------------------------

def preprocess_folder(input_dir: str,
                      output_dir: str,
                      resize_to: Optional[Tuple[int, int]] = None,
                      denoise_method: str = "gaussian"):
    """
    Preprocess all images in input_dir and write to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(
        glob(os.path.join(input_dir, "*.jpg")) +
        glob(os.path.join(input_dir, "*.png"))
    )

    for path in tqdm(image_paths, desc=f"Preprocessing {os.path.basename(input_dir)}"):
        img = load_image(path)
        processed = preprocess_image(
            img,
            resize_to=resize_to,
            denoise_method=denoise_method,
            sharpen_strength=0.3
        )

        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, processed)


# ---------------
# Visualization
# ---------------

def save_before_after(original_path: str,
                      processed_img: np.ndarray,
                      out_path: str):
    """
    Save a side-by-side comparison (for report).
    """
    import matplotlib.pyplot as plt

    orig = load_image(original_path)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    proc_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(orig_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Preprocessed")
    plt.imshow(proc_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def run_preprocessing():
    # Example global choice: keep native resolution (no resize)
    resize_to = None  # or (1024, 1024)

    preprocess_folder(CORRECT_DIR, PREPROC_CORRECT_DIR, resize_to=resize_to)
    preprocess_folder(PUZZLES_2_DIR, PREPROC_P2_DIR, resize_to=resize_to)
    preprocess_folder(PUZZLES_4_DIR, PREPROC_P4_DIR, resize_to=resize_to)
    preprocess_folder(PUZZLES_8_DIR, PREPROC_P8_DIR, resize_to=resize_to)

if __name__ == "__main__":
    ensure_dirs()

    # Example global choice: keep native resolution (no resize)
    resize_to = None  # or something like (1024, 1024)

    # Run for all folders
    preprocess_folder(CORRECT_DIR, PREPROC_CORRECT_DIR, resize_to=resize_to)
    preprocess_folder(PUZZLES_2_DIR, PREPROC_P2_DIR, resize_to=resize_to)
    preprocess_folder(PUZZLES_4_DIR, PREPROC_P4_DIR, resize_to=resize_to)
    preprocess_folder(PUZZLES_8_DIR, PREPROC_P8_DIR, resize_to=resize_to)

    # Example: create one before/after visualization for the report
    sample_input = os.path.join(PUZZLES_4_DIR, "0.jpg")
    sample_output = os.path.join(PREPROC_P4_DIR, "0.jpg")
    if os.path.exists(sample_input) and os.path.exists(sample_output):
        proc = load_image(sample_output)
        comparison_path = os.path.join(VISUALS_DIR, "preprocess_before_after_0.png")
        save_before_after(sample_input, proc, comparison_path)
