import os
import cv2
import numpy as np


# ================================
# 1. Helpers
# ================================

def infer_grid_from_folder(folder_name: str) -> int:
    """
    Infer grid size N from folder name like 'puzzle_4x4'.
    """
    if "2x2" in folder_name:
        return 2
    if "4x4" in folder_name:
        return 4
    if "8x8" in folder_name:
        return 8
    raise ValueError(f"Cannot infer grid size from folder name: {folder_name}")


def filename_no_ext(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


# ================================
# 2. Load Phase-1 Tiles
# ================================

def load_phase1_tiles(image_path: str,
                      phase1_root: str = "artifacts"):
    """
    Given the original scrambled puzzle image path and the Phase-1 artifacts root,
    load the tiles produced by Phase-1 along with pre-computed sobel edges and corners.

    Returns:
      tiles: list of dicts: {
        'img': BGR tile image,
        'edges': sobel edges (grayscale),
        'corners_img': corners visualization (BGR),
        'row': r, 'col': c, 'id': r*N + c
      }
      N: grid size
      tile_h, tile_w: tile dimensions
      tiles_dir: path where tiles were loaded from
    """
    folder_name = os.path.basename(os.path.dirname(image_path))  # puzzle_4x4
    img_name = filename_no_ext(image_path)                       # e.g., "56"
    N = infer_grid_from_folder(folder_name)

    tiles_dir = os.path.join(phase1_root, folder_name, img_name, "tiles")
    if not os.path.isdir(tiles_dir):
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

    tiles = []
    # load tiles in row-major order from organized structure: tiles/0/rgb/, tiles/1/rgb/, ...
    for r in range(N):
        for c in range(N):
            tile_idx = r * N + c
            tile_folder = os.path.join(tiles_dir, str(tile_idx))
            
            # Load pre-computed RGB tile
            rgb_path = os.path.join(tile_folder, "rgb", f"tile_{r}_{c}_rgb.jpg")
            tile_bgr = cv2.imread(rgb_path)
            if tile_bgr is None:
                raise FileNotFoundError(f"Missing tile: {rgb_path}")
            
            # Load pre-computed Sobel edges
            edges_path = os.path.join(tile_folder, "edges_sobel", f"tile_{r}_{c}_edges_sobel.jpg")
            tile_edges = cv2.imread(edges_path, cv2.IMREAD_GRAYSCALE)
            if tile_edges is None:
                raise FileNotFoundError(f"Missing edges: {edges_path}")
            
            # Load pre-computed corners
            corners_path = os.path.join(tile_folder, "corners", f"tile_{r}_{c}_corners.jpg")
            tile_corners = cv2.imread(corners_path)
            if tile_corners is None:
                raise FileNotFoundError(f"Missing corners: {corners_path}")
            
            tiles.append({
                "img": tile_bgr,
                "edges": tile_edges,
                "corners_img": tile_corners,
                "row": r,
                "col": c,
                "id": tile_idx  # unique ID
            })

    # assume all tiles same size
    h, w, _ = tiles[0]["img"].shape
    return tiles, N, h, w, tiles_dir


# ================================
# 3. Feature Extraction
# ================================

def extract_harris_corners(gray_tile,
                           block_size=2,
                           ksize=3,
                           k=0.04,
                           thresh_ratio=0.01):
    """
    Harris corner detection, same as lecture.
    Returns corner positions as (y,x) ndarray.
    """
    gray_f = np.float32(gray_tile)
    dst = cv2.cornerHarris(gray_f, block_size, ksize, k)
    dst = cv2.dilate(dst, None)  # for visual robustness
    pts = np.argwhere(dst > thresh_ratio * dst.max())
    return pts  # shape (M,2): [y,x]


def side_corners(corners, side, h, w, band=5):
    """
    Filter corners near a specific border:
      T = top, B = bottom, L = left, R = right
    """
    if side == 'T':
        return corners[corners[:, 0] < band]
    if side == 'B':
        return corners[corners[:, 0] > h - band]
    if side == 'L':
        return corners[corners[:, 1] < band]
    if side == 'R':
        return corners[corners[:, 1] > w - band]
    return np.empty((0, 2), dtype=int)


def border_sobel_descriptor(gray_tile, band=3):
    """
    Simple Sobel-based border descriptor:
      - compute Sobel magnitude
      - average along a 'band'-pixel-wide strip along each border
      - returns dict with 1D vectors: 'T','B','L','R'
    """
    sobelx = cv2.Sobel(gray_tile, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_tile, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)

    h, w = mag.shape
    b = min(band, h // 2, w // 2)

    top_strip = mag[0:b, :]
    bottom_strip = mag[h - b:h, :]
    left_strip = mag[:, 0:b]
    right_strip = mag[:, w - b:w]

    desc = {
        "T": top_strip.mean(axis=0),   # vector of length w
        "B": bottom_strip.mean(axis=0),
        "L": left_strip.mean(axis=1),  # vector of length h
        "R": right_strip.mean(axis=1),
    }
    return desc


def prepare_tile_features(tiles, tile_h, tile_w):
    """
    For each tile:
      - use pre-computed Sobel edges to create border descriptors
      - extract corners from pre-computed corner images
      - keep only corners near each side
    """
    for t in tiles:
        # Use pre-computed sobel edges for border descriptor
        t["sobel"] = border_sobel_descriptor(t["edges"])
        
        # Extract corner positions from pre-computed corner visualization
        # Corners are marked in red [255, 0, 0] in the corner image
        corners_gray = cv2.cvtColor(t["corners_img"], cv2.COLOR_BGR2GRAY)
        # Find pixels that were marked as corners (non-zero in the visualization)
        corner_mask = corners_gray > 200  # High threshold for red channel
        corners = np.argwhere(corner_mask)  # Returns (y, x) positions
        
        t["corners"] = {
            s: side_corners(corners, s, tile_h, tile_w)
            for s in ['T', 'B', 'L', 'R']
        }


# ================================
# 4. Matching Costs
# ================================

def corner_cost(cA, cB):
    """
    Harris-corner-based cost from side A to side B.
    Smaller is better (more similar).
    """
    if len(cA) == 0 or len(cB) == 0:
        return 1e6  # no corners => very bad

    dists = []
    for (y, x) in cA:
        diff = cB - np.array([y, x])
        dn = np.min(np.sum(diff * diff, axis=1))
        dists.append(dn)
    return float(np.mean(dists))


def sobel_cost(descA, sideA, descB, sideB):
    """
    Border Sobel difference as cost.
    """
    v1 = descA[sideA]
    v2 = descB[sideB]
    L = min(len(v1), len(v2))
    if L == 0:
        return 1e6
    v1 = v1[:L]
    v2 = v2[:L]
    return float(np.mean(np.abs(v1 - v2)))


def combined_cost(tileA, sideA, tileB, sideB,
                  alpha=0.5, beta=0.5):
    """
    Combine Harris and Sobel costs:
      total = alpha * Sobel + beta * Harris
    """
    c_corner = corner_cost(tileA["corners"][sideA],
                           tileB["corners"][sideB])
    c_sobel = sobel_cost(tileA["sobel"], sideA,
                         tileB["sobel"], sideB)
    return alpha * c_sobel + beta * c_corner


# ================================
# 5. Greedy Reassembly on Phase-1 Tiles
# ================================

def greedy_reassemble_from_tiles(tiles, N, tile_h, tile_w):
    """
    Greedy assembly using:
      - left→right matches within a row
      - top→bottom matches across rows
    No rotation, orientation is fixed.
    Returns:
      reconstructed BGR image
      layout_ids: list of tile IDs in row-major order
      avg_neighbor_cost: mean combined cost of all matched edges
    """
    # Ensure features present
    prepare_tile_features(tiles, tile_h, tile_w)

    used = set()
    layout = [[None for _ in range(N)] for _ in range(N)]
    neighbor_costs = []

    # Start with first tile as (0,0)
    start = tiles[0]
    layout[0][0] = start
    used.add(start["id"])

    # Fill first row: left to right
    for c in range(1, N):
        left_tile = layout[0][c - 1]
        best_tile = None
        best_cost = 1e9

        for t in tiles:
            if t["id"] in used:
                continue
            cost = combined_cost(left_tile, "R", t, "L")
            if cost < best_cost:
                best_cost = cost
                best_tile = t

        layout[0][c] = best_tile
        used.add(best_tile["id"])
        neighbor_costs.append(best_cost)

    # Fill remaining rows using top neighbors
    for r in range(1, N):
        for c in range(N):
            top_tile = layout[r - 1][c]
            best_tile = None
            best_cost = 1e9

            for t in tiles:
                if t["id"] in used:
                    continue
                cost = combined_cost(top_tile, "B", t, "T")
                if cost < best_cost:
                    best_cost = cost
                    best_tile = t

            layout[r][c] = best_tile
            used.add(best_tile["id"])
            neighbor_costs.append(best_cost)

    # Build reconstructed image and ID layout
    H = N * tile_h
    W = N * tile_w
    out = np.zeros((H, W, 3), dtype=np.uint8)
    layout_ids = []

    for r in range(N):
        for c in range(N):
            t = layout[r][c]
            layout_ids.append(t["id"])
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            out[y0:y1, x0:x1] = t["img"]

    avg_neighbor_cost = float(np.mean(neighbor_costs)) if neighbor_costs else 0.0
    return out, layout_ids, avg_neighbor_cost


# ================================
# 6. High-level API
# ================================

def solve_puzzle_from_phase1(image_path: str,
                             phase1_root: str = "artifacts",
                             out_root: str = "phase2_solved"):
    """
    Main Phase-2 solver:
      - loads Phase-1 tiles for given puzzle image
      - uses Harris + Sobel to reassemble
      - saves reconstructed image
    """
    tiles, N, tile_h, tile_w, tiles_dir = load_phase1_tiles(image_path,
                                                            phase1_root)
    recon, layout_ids, avg_cost = greedy_reassemble_from_tiles(
        tiles, N, tile_h, tile_w
    )

    folder_name = os.path.basename(os.path.dirname(image_path))
    img_name = filename_no_ext(image_path)

    out_dir = os.path.join(out_root, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{img_name}_reconstructed_{N}x{N}.jpg")
    cv2.imwrite(out_path, recon)

    print(f"[INFO] {image_path} -> N={N}, avg_neighbor_cost={avg_cost:.2f}")
    print(f"[OK] Saved reconstructed image to: {out_path}")

    return {
        "image_path": image_path,
        "grid_size": N,
        "layout_ids": layout_ids,
        "avg_neighbor_cost": avg_cost,
        "output_path": out_path,
    }


def solve_full_dataset_from_phase1(
    orig_root: str = "Gravity Falls",
    phase1_root: str = "artifacts",
    out_root: str = "phase2_solved",
):
    """
    Solve all puzzles in:
      Gravity Falls/puzzle_2x2
      Gravity Falls/puzzle_4x4
      Gravity Falls/puzzle_8x8
    using Phase-1 tiles from 'artifacts'.
    """
    folders = ["puzzle_2x2", "puzzle_4x4", "puzzle_8x8"]
    results = []

    for folder in folders:
        folder_path = os.path.join(orig_root, folder)
        if not os.path.isdir(folder_path):
            print(f"[WARN] Missing folder: {folder_path}")
            continue

        print(f"\n=== Solving puzzles in {folder_path} ===")
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder_path, fname)
            info = solve_puzzle_from_phase1(
                img_path,
                phase1_root=phase1_root,
                out_root=out_root,
            )
            results.append(info)

    return results


if __name__ == "__main__":
    # Run full Phase-2 over dataset using Phase-1 tiles
    solve_full_dataset_from_phase1()
