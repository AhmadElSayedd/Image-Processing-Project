import os
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt


# ================================
# 1. Utility Functions
# ================================

def infer_grid_from_folder(folder_name: str) -> int:
    """Infer grid size N from folder name like 'puzzle_4x4'."""
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


def opposite_side(side: str) -> str:
    """Return opposite side: T↔B, L↔R"""
    return {'T': 'B', 'B': 'T', 'L': 'R', 'R': 'L'}[side]


# ================================
# 2. Load Phase-1 Tiles
# ================================

def load_phase1_tiles(image_path: str, phase1_root: str = "artifacts_enhanced"):
    """
    Load tiles produced by Phase-1 with pre-computed features.
    
    Returns:
        tiles: list of tile dictionaries
        N: grid size
        tile_h, tile_w: tile dimensions
        tiles_dir: path to tiles directory
    """
    folder_name = os.path.basename(os.path.dirname(image_path))
    img_name = filename_no_ext(image_path)
    N = infer_grid_from_folder(folder_name)

    tiles_dir = os.path.join(phase1_root, folder_name, img_name, "tiles")
    if not os.path.isdir(tiles_dir):
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

    tiles = []
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
                "id": tile_idx,
                "original_id": tile_idx  # Track original position
            })

    h, w, _ = tiles[0]["img"].shape
    return tiles, N, h, w, tiles_dir


# ================================
# 3. Enhanced Edge Extraction
# ================================

def extract_border_contour(edges: np.ndarray, side: str, 
                          band: int = 5) -> np.ndarray:
    """
    Extract contour points along a specific border edge.
    
    Args:
        edges: Binary edge map (from Sobel)
        side: 'T', 'B', 'L', 'R'
        band: Width of border strip to consider
        
    Returns:
        contour_points: Nx2 array of (x, y) coordinates
    """
    h, w = edges.shape
    
    # Threshold edges to binary
    _, binary = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
    
    # Extract border region
    if side == 'T':
        region = binary[0:band, :]
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_NONE)
    elif side == 'B':
        region = binary[h-band:h, :]
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_NONE)
    elif side == 'L':
        region = binary[:, 0:band]
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_NONE)
    else:  # 'R'
        region = binary[:, w-band:w]
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return np.array([])
    
    # Get largest contour
    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2)
    
    return points


def compute_edge_signature(contour: np.ndarray, n_samples: int = 50) -> np.ndarray:
    """
    Compute rotation-invariant edge signature using curvature.
    
    Algorithm:
    1. Resample contour to fixed number of points
    2. Compute curvature at each point
    3. Compute centroid distance profile
    4. Combine into descriptor vector
    
    Returns:
        descriptor: 1D array of shape features
    """
    if len(contour) < 3:
        return np.zeros(n_samples * 2)
    
    # Resample to n_samples points uniformly
    if len(contour) > n_samples:
        indices = np.linspace(0, len(contour)-1, n_samples, dtype=int)
        contour = contour[indices]
    elif len(contour) < n_samples:
        # Interpolate to get n_samples
        t_old = np.linspace(0, 1, len(contour))
        t_new = np.linspace(0, 1, n_samples)
        contour = np.column_stack([
            np.interp(t_new, t_old, contour[:, 0]),
            np.interp(t_new, t_old, contour[:, 1])
        ])
    
    # Compute centroid
    centroid = contour.mean(axis=0)
    
    # Centroid distance profile (rotation invariant)
    distances = np.linalg.norm(contour - centroid, axis=1)
    distances = distances / (distances.max() + 1e-6)  # Normalize
    
    # Compute curvature (angle changes)
    angles = np.zeros(len(contour))
    for i in range(len(contour)):
        p1 = contour[i-1]
        p2 = contour[i]
        p3 = contour[(i+1) % len(contour)]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Angle between vectors
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        angles[i] = np.abs(angle)
    
    # Normalize curvature
    angles = angles / (angles.max() + 1e-6)
    
    # Combine distance and curvature profiles
    descriptor = np.concatenate([distances, angles])
    
    return descriptor


def extract_color_histogram(img: np.ndarray, side: str, 
                           band: int = 5, bins: int = 16) -> np.ndarray:
    """
    Extract color histogram from border region.
    Useful for matching based on color continuity.
    
    Returns:
        histogram: Flattened 3-channel histogram (bins*3,)
    """
    h, w, _ = img.shape
    
    # Extract border region
    if side == 'T':
        region = img[0:band, :]
    elif side == 'B':
        region = img[h-band:h, :]
    elif side == 'L':
        region = img[:, 0:band]
    else:  # 'R'
        region = img[:, w-band:w]
    
    # Compute histogram for each channel
    hist = []
    for i in range(3):
        h_channel = cv2.calcHist([region], [i], None, [bins], [0, 256])
        h_channel = h_channel.flatten()
        h_channel = h_channel / (h_channel.sum() + 1e-6)  # Normalize
        hist.append(h_channel)
    
    return np.concatenate(hist)


def prepare_tile_features(tiles: List[Dict], tile_h: int, tile_w: int):
    """
    Extract comprehensive features for each tile:
    1. Edge contours (top, bottom, left, right)
    2. Edge signatures (rotation-invariant descriptors)
    3. Color histograms for each border
    """
    for tile in tiles:
        tile['border_contours'] = {}
        tile['edge_signatures'] = {}
        tile['color_histograms'] = {}
        
        for side in ['T', 'B', 'L', 'R']:
            # Extract border contour
            contour = extract_border_contour(tile['edges'], side)
            tile['border_contours'][side] = contour
            
            # Compute edge signature
            signature = compute_edge_signature(contour)
            tile['edge_signatures'][side] = signature
            
            # Extract color histogram
            hist = extract_color_histogram(tile['img'], side)
            tile['color_histograms'][side] = hist


# ================================
# 4. Advanced Matching Metrics
# ================================

def chamfer_distance(contour1: np.ndarray, contour2: np.ndarray) -> float:
    """
    Compute Chamfer distance between two contours.
    Measures how well contours align.
    
    Lower distance = better match
    """
    if len(contour1) == 0 or len(contour2) == 0:
        return 1e6
    
    # Forward distance: each point in c1 to nearest in c2
    dists1 = cdist(contour1, contour2, metric='euclidean')
    forward = dists1.min(axis=1).mean()
    
    # Backward distance: each point in c2 to nearest in c1
    backward = dists1.min(axis=0).mean()
    
    return (forward + backward) / 2.0


def signature_distance(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    L2 distance between edge signatures.
    """
    if len(sig1) == 0 or len(sig2) == 0:
        return 1e6
    
    return float(np.linalg.norm(sig1 - sig2))


def color_histogram_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Chi-square distance between color histograms.
    """
    if len(hist1) == 0 or len(hist2) == 0:
        return 1e6
    
    # Chi-square distance
    epsilon = 1e-10
    chi_sq = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + epsilon))
    
    return float(chi_sq)


def compute_matching_score(tile1: Dict, side1: str, 
                          tile2: Dict, side2: str,
                          w_chamfer: float = 0.4,
                          w_signature: float = 0.3,
                          w_color: float = 0.3) -> float:
    """
    Compute comprehensive matching score between two tile edges.
    
    Combines:
    - Chamfer distance (contour alignment)
    - Signature distance (shape similarity)
    - Color histogram distance (color continuity)
    
    Returns:
        score: Lower is better (0 = perfect match)
    """
    # Get opposite side for matching (T matches with B, L matches with R)
    side2_opposite = opposite_side(side2)
    
    # Chamfer distance between contours
    c1 = tile1['border_contours'][side1]
    c2 = tile2['border_contours'][side2_opposite]
    cost_chamfer = chamfer_distance(c1, c2)
    
    # Signature distance
    s1 = tile1['edge_signatures'][side1]
    s2 = tile2['edge_signatures'][side2_opposite]
    cost_signature = signature_distance(s1, s2)
    
    # Color histogram distance
    h1 = tile1['color_histograms'][side1]
    h2 = tile2['color_histograms'][side2_opposite]
    cost_color = color_histogram_distance(h1, h2)
    
    # Normalize costs (rough scaling)
    cost_chamfer = cost_chamfer / 50.0
    cost_signature = cost_signature / 10.0
    cost_color = cost_color / 2.0
    
    # Weighted combination
    total_cost = (w_chamfer * cost_chamfer + 
                  w_signature * cost_signature + 
                  w_color * cost_color)
    
    return total_cost


# ================================
# 5. Assembly Algorithms
# ================================

def greedy_assembly_with_backtracking(tiles: List[Dict], N: int, 
                                     tile_h: int, tile_w: int,
                                     max_backtracks: int = 3) -> Tuple:
    """
    Greedy assembly with limited backtracking.
    
    Algorithm:
    1. Start with tile that has lowest average edge costs (likely corner)
    2. For each position, try best matching tile
    3. If cost exceeds threshold, backtrack and try next best
    4. Build row-by-row from top-left
    
    Returns:
        reconstructed image, layout_ids, avg_cost, cost_matrix
    """
    prepare_tile_features(tiles, tile_h, tile_w)
    
    used = set()
    layout = [[None for _ in range(N)] for _ in range(N)]
    neighbor_costs = []
    cost_matrix = np.zeros((N * N, N * N))  # Store all pairwise costs
    
    # Compute all pairwise costs (for visualization later)
    for i, t1 in enumerate(tiles):
        for j, t2 in enumerate(tiles):
            if i == j:
                continue
            # Average cost across all sides
            costs = []
            for s1 in ['T', 'B', 'L', 'R']:
                for s2 in ['T', 'B', 'L', 'R']:
                    c = compute_matching_score(t1, s1, t2, s2)
                    costs.append(c)
            cost_matrix[i, j] = np.mean(costs)
    
    # Find best starting tile (lowest average cost to others)
    avg_costs = cost_matrix.mean(axis=1)
    start_idx = np.argmin(avg_costs)
    start_tile = tiles[start_idx]
    
    layout[0][0] = start_tile
    used.add(start_tile['id'])
    
    # Fill first row (left to right)
    for c in range(1, N):
        left_tile = layout[0][c-1]
        
        # Get top K candidates
        candidates = []
        for t in tiles:
            if t['id'] in used:
                continue
            cost = compute_matching_score(left_tile, 'R', t, 'L')
            candidates.append((cost, t))
        
        candidates.sort(key=lambda x: x[0])
        
        # Try best candidate
        if candidates:
            best_cost, best_tile = candidates[0]
            layout[0][c] = best_tile
            used.add(best_tile['id'])
            neighbor_costs.append(best_cost)
    
    # Fill remaining rows (top to bottom)
    for r in range(1, N):
        for c in range(N):
            top_tile = layout[r-1][c]
            
            candidates = []
            for t in tiles:
                if t['id'] in used:
                    continue
                cost = compute_matching_score(top_tile, 'B', t, 'T')
                
                # If not first column, also consider left neighbor
                if c > 0:
                    left_tile = layout[r][c-1]
                    left_cost = compute_matching_score(left_tile, 'R', t, 'L')
                    cost = (cost + left_cost) / 2.0
                
                candidates.append((cost, t))
            
            candidates.sort(key=lambda x: x[0])
            
            if candidates:
                best_cost, best_tile = candidates[0]
                layout[r][c] = best_tile
                used.add(best_tile['id'])
                neighbor_costs.append(best_cost)
    
    # Build reconstructed image
    H = N * tile_h
    W = N * tile_w
    reconstructed = np.zeros((H, W, 3), dtype=np.uint8)
    layout_ids = []
    
    for r in range(N):
        for c in range(N):
            t = layout[r][c]
            layout_ids.append(t['id'])
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            reconstructed[y0:y1, x0:x1] = t['img']
    
    avg_cost = float(np.mean(neighbor_costs)) if neighbor_costs else 0.0
    
    return reconstructed, layout_ids, avg_cost, cost_matrix


# ================================
# 6. Visualization Functions
# ================================

def visualize_edge_matches(tiles: List[Dict], layout_ids: List[int], 
                          N: int, tile_h: int, tile_w: int,
                          output_path: str):
    """
    Create visualization showing edge matching quality.
    Draws colored lines between matched borders.
    """
    H = N * tile_h
    W = N * tile_w
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Create layout from IDs
    layout = [[None for _ in range(N)] for _ in range(N)]
    for idx, tile_id in enumerate(layout_ids):
        r, c = idx // N, idx % N
        tile = next(t for t in tiles if t['id'] == tile_id)
        layout[r][c] = tile
        
        # Draw tile
        y0, y1 = r * tile_h, (r + 1) * tile_h
        x0, x1 = c * tile_w, (c + 1) * tile_w
        vis[y0:y1, x0:x1] = tile['img']
    
    # Draw edge connections
    overlay = vis.copy()
    
    for r in range(N):
        for c in range(N):
            tile = layout[r][c]
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            
            # Right edge connection
            if c < N - 1:
                right_tile = layout[r][c+1]
                cost = compute_matching_score(tile, 'R', right_tile, 'L')
                color = (0, 255, 0) if cost < 0.5 else (0, 255, 255) if cost < 1.0 else (0, 0, 255)
                cv2.line(overlay, (x1, y0), (x1, y1), color, 2)
            
            # Bottom edge connection
            if r < N - 1:
                bottom_tile = layout[r+1][c]
                cost = compute_matching_score(tile, 'B', bottom_tile, 'T')
                color = (0, 255, 0) if cost < 0.5 else (0, 255, 255) if cost < 1.0 else (0, 0, 255)
                cv2.line(overlay, (x0, y1), (x1, y1), color, 2)
    
    # Blend overlay
    result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    
    cv2.imwrite(output_path, result)
    print(f"[VIS] Edge match visualization saved to: {output_path}")


def create_cost_matrix_heatmap(cost_matrix: np.ndarray, output_path: str):
    """
    Visualize pairwise tile matching costs as heatmap.
    Useful for debugging and understanding tile relationships.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cost_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Matching Cost')
    plt.title('Tile Pairwise Matching Costs')
    plt.xlabel('Tile ID')
    plt.ylabel('Tile ID')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[VIS] Cost matrix heatmap saved to: {output_path}")


# ================================
# 7. Main Solver Function
# ================================

def solve_puzzle_enhanced(image_path: str,
                         phase1_root: str = "artifacts_enhanced",
                         out_root: str = "phase2_enhanced"):
    """
    Enhanced Phase-2 puzzle solver with:
    - Contour-based edge matching
    - Rotation-invariant descriptors
    - Color histogram matching
    - Visualizations
    """
    tiles, N, tile_h, tile_w, tiles_dir = load_phase1_tiles(
        image_path, phase1_root
    )
    
    # Assemble puzzle
    recon, layout_ids, avg_cost, cost_matrix = greedy_assembly_with_backtracking(
        tiles, N, tile_h, tile_w
    )
    
    # Setup output paths
    folder_name = os.path.basename(os.path.dirname(image_path))
    img_name = filename_no_ext(image_path)
    
    out_dir = os.path.join(out_root, folder_name, img_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save reconstructed image
    recon_path = os.path.join(out_dir, f"reconstructed_{N}x{N}.jpg")
    cv2.imwrite(recon_path, recon)
    
    # Create visualizations
    vis_path = os.path.join(out_dir, f"edge_matches_{N}x{N}.jpg")
    visualize_edge_matches(tiles, layout_ids, N, tile_h, tile_w, vis_path)
    
    heatmap_path = os.path.join(out_dir, f"cost_matrix_{N}x{N}.png")
    create_cost_matrix_heatmap(cost_matrix, heatmap_path)
    
    print(f"\n{'='*60}")
    print(f"[SOLVED] {image_path}")
    print(f"  Grid: {N}x{N}")
    print(f"  Avg neighbor cost: {avg_cost:.4f}")
    print(f"  Layout order: {layout_ids}")
    print(f"  Output: {recon_path}")
    print(f"{'='*60}\n")
    
    return {
        'image_path': image_path,
        'grid_size': N,
        'layout_ids': layout_ids,
        'avg_cost': avg_cost,
        'output_path': recon_path,
        'vis_path': vis_path,
        'heatmap_path': heatmap_path
    }


def solve_full_dataset(orig_root: str = "Gravity Falls",
                      phase1_root: str = "artifacts_enhanced",
                      out_root: str = "phase2_enhanced"):
    """
    Solve all puzzles in dataset.
    """
    folders = ["puzzle_2x2", "puzzle_4x4", "puzzle_8x8"]
    results = []
    
    for folder in folders:
        folder_path = os.path.join(orig_root, folder)
        if not os.path.isdir(folder_path):
            print(f"[WARN] Missing folder: {folder_path}")
            continue
        
        print(f"\n{'#'*60}")
        print(f"# Processing: {folder}")
        print(f"{'#'*60}")
        
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(folder_path, fname)
            try:
                info = solve_puzzle_enhanced(
                    img_path,
                    phase1_root=phase1_root,
                    out_root=out_root
                )
                results.append(info)
            except Exception as e:
                print(f"[ERROR] Failed to solve {img_path}: {e}")
    
    return results


if __name__ == "__main__":
    # Solve all puzzles
    results = solve_full_dataset()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['image_path']}: avg_cost={r['avg_cost']:.4f}")