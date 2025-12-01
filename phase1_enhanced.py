import os
import cv2
import numpy as np


# ==============================
# Enhanced Phase 1 Pipeline
# ==============================
# Improvements for better Phase 2 assembly:
# 1. Better edge detection (Canny + Sobel combined)
# 2. Border enhancement for clearer edge matching
# 3. Color normalization for consistent histograms
# 4. Adaptive thresholding for varying lighting


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def filename_no_ext(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


# ==============================
# Enhanced Preprocessing
# ==============================

def adaptive_denoise(gray):
    """
    Adaptive denoising based on image noise level.
    """
    # Estimate noise level
    noise_sigma = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if noise_sigma > 500:  # High noise
        g = cv2.GaussianBlur(gray, (7, 7), 1.5)
        m = cv2.medianBlur(g, 5)
    else:  # Low noise
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        m = cv2.medianBlur(g, 3)
    
    return m


def enhance_contrast(gray):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    for better feature visibility.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def hybrid_edge_detection(gray):
    """
    Combine Canny and Sobel for robust edge detection.
    
    Canny: Good for continuous edges
    Sobel: Good for texture and gradients
    """
    # Canny edges
    canny = cv2.Canny(gray, 50, 150)
    
    # Sobel magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
    sobel_mag = sobel_mag.astype(np.uint8)
    
    # Threshold Sobel
    _, sobel_binary = cv2.threshold(sobel_mag, 30, 255, cv2.THRESH_BINARY)
    
    # Combine: union of both edge maps
    combined = cv2.bitwise_or(canny, sobel_binary)
    
    return combined, sobel_mag


def enhance_borders(edges, border_width=10):
    """
    Enhance border regions specifically for better edge matching.
    Applies morphological operations to strengthen border edges.
    """
    h, w = edges.shape
    enhanced = edges.copy()
    
    # Create border masks
    top_mask = np.zeros_like(edges)
    bottom_mask = np.zeros_like(edges)
    left_mask = np.zeros_like(edges)
    right_mask = np.zeros_like(edges)
    
    top_mask[0:border_width, :] = 255
    bottom_mask[h-border_width:h, :] = 255
    left_mask[:, 0:border_width] = 255
    right_mask[:, w-border_width:w] = 255
    
    border_mask = cv2.bitwise_or(
        cv2.bitwise_or(top_mask, bottom_mask),
        cv2.bitwise_or(left_mask, right_mask)
    )
    
    # Apply morphological closing to border regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    border_edges = cv2.bitwise_and(edges, border_mask)
    border_enhanced = cv2.morphologyEx(border_edges, cv2.MORPH_CLOSE, kernel)
    
    # Combine with original edges
    interior = cv2.bitwise_and(edges, cv2.bitwise_not(border_mask))
    enhanced = cv2.bitwise_or(interior, border_enhanced)
    
    return enhanced


def normalize_color(bgr_img):
    """
    Color normalization for consistent color histogram matching.
    Uses lab color space for perceptual uniformity.
    """
    # Convert to LAB
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return normalized


def slice_to_tiles(rgb_img, grid_size):
    """Split image into grid tiles."""
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


def compute_tile_corners(gray_tile):
    """
    Improved corner detection with non-maximum suppression.
    """
    # Harris corners
    gray_f = np.float32(gray_tile)
    dst = cv2.cornerHarris(gray_f, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    
    # Threshold
    threshold = 0.01 * dst.max()
    corner_mask = dst > threshold
    
    # Non-maximum suppression
    corner_coords = np.argwhere(corner_mask)
    if len(corner_coords) == 0:
        return corner_mask
    
    # Keep only local maxima
    nms_mask = np.zeros_like(corner_mask)
    for y, x in corner_coords:
        window_y = slice(max(0, y-2), min(gray_tile.shape[0], y+3))
        window_x = slice(max(0, x-2), min(gray_tile.shape[1], x+3))
        window = dst[window_y, window_x]
        
        if dst[y, x] == window.max():
            nms_mask[y, x] = True
    
    return nms_mask


# ==============================
# Main Processing Pipeline
# ==============================

def process_single_puzzle_enhanced(image_path, grid_size, 
                                  out_root="artifacts_enhanced"):
    """
    Enhanced Phase-1 processing with improvements for Phase-2.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"[ERROR] Cannot open {image_path}")
        return

    # Color normalization
    bgr_norm = normalize_color(bgr)
    rgb = cv2.cvtColor(bgr_norm, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr_norm, cv2.COLOR_BGR2GRAY)

    folder_name = os.path.basename(os.path.dirname(image_path))
    img_name = filename_no_ext(image_path)
    base_dir = os.path.join(out_root, folder_name, img_name)
    tiles_dir = os.path.join(base_dir, "tiles")
    ensure_dir(base_dir)
    ensure_dir(tiles_dir)

    # 1. Adaptive denoising
    denoised = adaptive_denoise(gray)
    
    # 2. Contrast enhancement
    enhanced = enhance_contrast(denoised)
    
    # 3. Hybrid edge detection
    edges_combined, edges_sobel = hybrid_edge_detection(enhanced)
    
    # 4. Border enhancement
    edges_final = enhance_borders(edges_combined)

    # Save global artifacts
    cv2.imwrite(os.path.join(base_dir, "0_original_rgb.jpg"), 
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(base_dir, "1_gray.jpg"), gray)
    cv2.imwrite(os.path.join(base_dir, "2_denoised.jpg"), denoised)
    cv2.imwrite(os.path.join(base_dir, "3_enhanced_contrast.jpg"), enhanced)
    cv2.imwrite(os.path.join(base_dir, "4_edges_hybrid.jpg"), edges_final)
    cv2.imwrite(os.path.join(base_dir, "5_edges_sobel_mag.jpg"), edges_sobel)

    # 5. Tile splitting with enhanced features
    tiles, _, _ = slice_to_tiles(rgb, grid_size)

    for tile_rgb, r, c in tiles:
        tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
        tile_gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)

        # Process tile
        tile_denoised = adaptive_denoise(tile_gray)
        tile_enhanced = enhance_contrast(tile_denoised)
        tile_edges_combined, tile_edges_sobel = hybrid_edge_detection(tile_enhanced)
        tile_edges_final = enhance_borders(tile_edges_combined)
        
        # Compute corners with NMS
        corner_mask = compute_tile_corners(tile_enhanced)
        tile_corner_vis = tile_rgb.copy()
        tile_corner_vis[corner_mask] = [255, 0, 0]

        # Organized output structure
        tile_idx = r * grid_size + c
        tile_folder = os.path.join(tiles_dir, str(tile_idx))
        rgb_folder = os.path.join(tile_folder, "rgb")
        corners_folder = os.path.join(tile_folder, "corners")
        edges_folder = os.path.join(tile_folder, "edges_sobel")
        edges_hybrid_folder = os.path.join(tile_folder, "edges_hybrid")
        
        ensure_dir(rgb_folder)
        ensure_dir(corners_folder)
        ensure_dir(edges_folder)
        ensure_dir(edges_hybrid_folder)

        # Save enhanced tile artifacts
        cv2.imwrite(
            os.path.join(rgb_folder, f"tile_{r}_{c}_rgb.jpg"),
            tile_bgr
        )
        cv2.imwrite(
            os.path.join(edges_folder, f"tile_{r}_{c}_edges_sobel.jpg"),
            tile_edges_sobel
        )
        cv2.imwrite(
            os.path.join(edges_hybrid_folder, f"tile_{r}_{c}_edges_hybrid.jpg"),
            tile_edges_final
        )
        cv2.imwrite(
            os.path.join(corners_folder, f"tile_{r}_{c}_corners.jpg"),
            cv2.cvtColor(tile_corner_vis, cv2.COLOR_RGB2BGR)
        )

    print(f"[OK] Enhanced processing: {image_path} → {base_dir}")


def process_full_dataset_enhanced(dataset_root="Gravity Falls", 
                                 out_root="artifacts_enhanced"):
    """
    Process entire dataset with enhanced Phase-1.
    """
    folder_to_grid = {
        "puzzle_2x2": 2,
        "puzzle_4x4": 4,
        "puzzle_8x8": 8
    }

    for folder, grid in folder_to_grid.items():
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.exists(folder_path):
            continue

        print(f"\n=== {folder} — {grid}x{grid} (Enhanced) ===")
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                process_single_puzzle_enhanced(
                    os.path.join(folder_path, fname), 
                    grid, 
                    out_root
                )


if __name__ == "__main__":
    process_full_dataset_enhanced()
    print("\n[COMPLETE] Enhanced Phase-1 processing finished.")
    print("Use these artifacts with the enhanced Phase-2 solver.")