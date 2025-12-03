# src/segment_tiles.py
import os
import cv2
from glob import glob
from tqdm import tqdm

from config import (
    PREPROC_P2_DIR, PREPROC_P4_DIR, PREPROC_P8_DIR,
    TILES_2_DIR, TILES_4_DIR, TILES_8_DIR,
    VISUALS_DIR, ensure_dirs
)

# Segment a single image into N x N tiles and save them
def segment_image_into_tiles(img_path: str,
                             output_dir: str,
                             N: int):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    tile_h = h // N
    tile_w = w // N

    basename = os.path.splitext(os.path.basename(img_path))[0]

    for r in range(N):
        for c in range(N):
            y0 = r * tile_h
            y1 = (r + 1) * tile_h
            x0 = c * tile_w
            x1 = (c + 1) * tile_w

            tile = img[y0:y1, x0:x1]

            tile_name = f"{basename}_r{r}_c{c}.png"
            out_path = os.path.join(output_dir, tile_name)
            cv2.imwrite(out_path, tile)

# Segment all images in a folder
def segment_folder(preproc_dir: str,
                   tiles_dir: str,
                   N: int):
    os.makedirs(tiles_dir, exist_ok=True)
    image_paths = sorted(
        glob(os.path.join(preproc_dir, "*.jpg")) +
        glob(os.path.join(preproc_dir, "*.png"))
    )

    for path in tqdm(image_paths, desc=f"Segmenting N={N} from {os.path.basename(preproc_dir)}"):
        segment_image_into_tiles(path, tiles_dir, N=N)


def save_tile_grid_visual(preproc_img_path: str,
                          N: int,
                          out_path: str):
    """
    Save a visualization showing grid lines over a preprocessed puzzle.
    Good for the report to show 'segmentation'.
    """
    import matplotlib.pyplot as plt

    img = cv2.imread(preproc_img_path)
    if img is None:
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    tile_h = h // N
    tile_w = w // N

    plt.figure(figsize=(4, 4))
    plt.imshow(img_rgb)
    # Draw grid lines
    for r in range(1, N):
        y = r * tile_h
        plt.axhline(y, color='red', linewidth=0.8)
    for c in range(1, N):
        x = c * tile_w
        plt.axvline(x, color='red', linewidth=0.8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def run_segmentation():
    segment_folder(PREPROC_P2_DIR, TILES_2_DIR, N=2)
    segment_folder(PREPROC_P4_DIR, TILES_4_DIR, N=4)
    segment_folder(PREPROC_P8_DIR, TILES_8_DIR, N=8)

if __name__ == "__main__":
    ensure_dirs()

    # Segment tiles for each grid size
    segment_folder(PREPROC_P2_DIR, TILES_2_DIR, N=2)
    segment_folder(PREPROC_P4_DIR, TILES_4_DIR, N=4)
    segment_folder(PREPROC_P8_DIR, TILES_8_DIR, N=8)

    # Example visualization for report
    sample_preproc = os.path.join(PREPROC_P4_DIR, "0.jpg")
    if os.path.exists(sample_preproc):
        vis_path = os.path.join(VISUALS_DIR, "segmentation_grid_0.png")
        save_tile_grid_visual(sample_preproc, N=4, out_path=vis_path)
