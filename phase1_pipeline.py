import os
import cv2
import numpy as np


# ==============================
# 1. Basic helpers
# ==============================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def filename_no_ext(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


# ==============================
# 2. Pre-processing functions
# ==============================

def to_gray(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)


def denoise(gray):
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    m = cv2.medianBlur(g, 3)
    return m


def unsharp_mask(gray, amount=1.5, sigma=1.0):
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    high_freq = cv2.subtract(gray, blurred)
    sharpened = cv2.addWeighted(gray, 1.0, high_freq, amount, 0)
    return sharpened


def morph_clean(binary_edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary_edges, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def sobel_edges(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def harris_corners(gray):
    gray_f = np.float32(gray)
    dst = cv2.cornerHarris(gray_f, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    return dst


# ==============================
# 3. Tiling & Contours
# ==============================

def slice_to_tiles(rgb_img, grid_size):
    h, w, _ = rgb_img.shape
    tile_h = h // grid_size
    tile_w = w // grid_size

    tiles = []
    for r in range(grid_size):
        for c in range(grid_size):
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            tile = rgb_img[y0:y1, x0:x1].copy()
            tiles.append((tile, r, c))
    return tiles, tile_h, tile_w


# ==============================
# 4. Full pipeline for 1 image
# ==============================

def process_single_puzzle(image_path, grid_size, out_root="artifacts"):
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"[ERROR] Cannot open {image_path}")
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = to_gray(bgr)

    folder_name = os.path.basename(os.path.dirname(image_path))
    img_name = filename_no_ext(image_path)
    base_dir = os.path.join(out_root, folder_name, img_name)
    tiles_dir = os.path.join(base_dir, "tiles")
    ensure_dir(base_dir)
    ensure_dir(tiles_dir)

    # 1️⃣ Denoise
    den = denoise(gray)

    # 2️⃣ Unsharp Mask
    enhanced = unsharp_mask(den)

    # 3️⃣ Global Sobel Edges
    edges = sobel_edges(enhanced)
    edges_clean = morph_clean(edges)

    # Save global artifacts
    cv2.imwrite(os.path.join(base_dir, "0_original_rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(base_dir, "1_gray.jpg"), gray)
    cv2.imwrite(os.path.join(base_dir, "2_denoised.jpg"), den)
    cv2.imwrite(os.path.join(base_dir, "3_enhanced_unsharp.jpg"), enhanced)
    cv2.imwrite(os.path.join(base_dir, "4_edges_sobel.jpg"), edges_clean)

    # 4️⃣ Tile splitting
    tiles, _, _ = slice_to_tiles(rgb, grid_size)

    # 5️⃣ Tile edges + corners
    for tile_rgb, r, c in tiles:
        tile_gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)

        tile_edges = sobel_edges(tile_gray)
        corners = harris_corners(tile_gray)

        tile_corner_vis = tile_rgb.copy()
        tile_corner_vis[corners > 0.01 * corners.max()] = [255, 0, 0]

        # Create organized subdirectories for each tile
        tile_idx = r * grid_size + c
        tile_folder = os.path.join(tiles_dir, str(tile_idx))
        rgb_folder = os.path.join(tile_folder, "rgb")
        corners_folder = os.path.join(tile_folder, "corners")
        edges_folder = os.path.join(tile_folder, "edges_sobel")
        
        ensure_dir(rgb_folder)
        ensure_dir(corners_folder)
        ensure_dir(edges_folder)

        # Save files in organized structure
        cv2.imwrite(os.path.join(rgb_folder, f"tile_{r}_{c}_rgb.jpg"), cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(edges_folder, f"tile_{r}_{c}_edges_sobel.jpg"), tile_edges)
        cv2.imwrite(os.path.join(corners_folder, f"tile_{r}_{c}_corners.jpg"),
                    cv2.cvtColor(tile_corner_vis, cv2.COLOR_RGB2BGR))

    print(f"[OK] {image_path} → grid {grid_size}x{grid_size}, saved in '{base_dir}'")


# ==============================
# 5. Batch processor
# ==============================

def process_full_dataset(dataset_root="Gravity Falls", out_root="artifacts"):

    folder_to_grid = {
        "puzzle_2x2": 2,
        "puzzle_4x4": 4,
        "puzzle_8x8": 8
    }

    for folder, grid in folder_to_grid.items():
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.exists(folder_path):
            continue

        print(f"\n=== {folder} — {grid}x{grid} ===")
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                process_single_puzzle(os.path.join(folder_path, fname), grid, out_root)


if __name__ == "__main__":
    process_full_dataset()
