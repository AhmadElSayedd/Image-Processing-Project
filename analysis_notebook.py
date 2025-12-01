"""
Puzzle Assembly Analysis & Metrics
====================================

This script provides comprehensive analysis tools for evaluating
puzzle assembly quality and understanding algorithm behavior.

Features:
1. Accuracy metrics (if ground truth available)
2. Matching quality visualization
3. Edge compatibility analysis
4. Performance comparison between methods
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json


# ================================
# 1. Accuracy Metrics
# ================================

def compute_assembly_accuracy(predicted_layout: List[int], 
                             ground_truth_layout: List[int]) -> Dict:
    """
    Compute accuracy metrics for puzzle assembly.
    
    Metrics:
    - Exact match accuracy: % of tiles in correct positions
    - Neighbor accuracy: % of tiles with correct neighbors
    - Row/column accuracy: % correct within rows/columns
    
    Args:
        predicted_layout: List of tile IDs in assembled order
        ground_truth_layout: List of correct tile IDs
        
    Returns:
        Dictionary of accuracy metrics
    """
    N = int(np.sqrt(len(predicted_layout)))
    total_tiles = N * N
    
    # Exact position accuracy
    correct_positions = sum(
        1 for p, g in zip(predicted_layout, ground_truth_layout) if p == g
    )
    exact_accuracy = correct_positions / total_tiles
    
    # Neighbor accuracy (4-connectivity)
    correct_neighbors = 0
    total_neighbors = 0
    
    pred_grid = np.array(predicted_layout).reshape(N, N)
    gt_grid = np.array(ground_truth_layout).reshape(N, N)
    
    for r in range(N):
        for c in range(N):
            # Check right neighbor
            if c < N - 1:
                total_neighbors += 1
                pred_pair = (pred_grid[r, c], pred_grid[r, c+1])
                # Check if this pair exists adjacently in ground truth
                if check_adjacent_in_gt(pred_pair, gt_grid, 'horizontal'):
                    correct_neighbors += 1
            
            # Check bottom neighbor
            if r < N - 1:
                total_neighbors += 1
                pred_pair = (pred_grid[r, c], pred_grid[r+1, c])
                if check_adjacent_in_gt(pred_pair, gt_grid, 'vertical'):
                    correct_neighbors += 1
    
    neighbor_accuracy = correct_neighbors / total_neighbors if total_neighbors > 0 else 0
    
    # Row accuracy: % of rows with all correct tiles (order may differ)
    row_accuracy = 0
    for r in range(N):
        pred_row = set(pred_grid[r, :])
        gt_row = set(gt_grid[r, :])
        if pred_row == gt_row:
            row_accuracy += 1
    row_accuracy /= N
    
    # Column accuracy
    col_accuracy = 0
    for c in range(N):
        pred_col = set(pred_grid[:, c])
        gt_col = set(gt_grid[:, c])
        if pred_col == gt_col:
            col_accuracy += 1
    col_accuracy /= N
    
    return {
        'exact_accuracy': exact_accuracy,
        'neighbor_accuracy': neighbor_accuracy,
        'row_accuracy': row_accuracy,
        'col_accuracy': col_accuracy,
        'correct_positions': correct_positions,
        'total_tiles': total_tiles
    }


def check_adjacent_in_gt(pair: Tuple[int, int], gt_grid: np.ndarray, 
                        direction: str) -> bool:
    """
    Check if a pair of tiles is adjacent in ground truth grid.
    """
    N = gt_grid.shape[0]
    tile1, tile2 = pair
    
    # Find positions of both tiles in ground truth
    pos1 = np.argwhere(gt_grid == tile1)
    pos2 = np.argwhere(gt_grid == tile2)
    
    if len(pos1) == 0 or len(pos2) == 0:
        return False
    
    r1, c1 = pos1[0]
    r2, c2 = pos2[0]
    
    if direction == 'horizontal':
        return r1 == r2 and abs(c1 - c2) == 1
    else:  # vertical
        return c1 == c2 and abs(r1 - r2) == 1


# ================================
# 2. Visual Quality Assessment
# ================================

def compute_seam_quality(assembled_img: np.ndarray, N: int) -> Dict:
    """
    Analyze visual quality of tile seams in assembled image.
    
    Measures:
    - Color discontinuity at seams
    - Edge alignment at seams
    - Texture consistency
    
    Lower scores = better assembly quality
    """
    h, w, _ = assembled_img.shape
    tile_h, tile_w = h // N, w // N
    
    # Convert to LAB for perceptual color difference
    lab = cv2.cvtColor(assembled_img, cv2.COLOR_BGR2LAB)
    
    horizontal_seam_costs = []
    vertical_seam_costs = []
    
    # Analyze horizontal seams (between rows)
    for r in range(1, N):
        seam_y = r * tile_h
        # Compare pixels just above and below seam
        above = lab[seam_y-1:seam_y, :]
        below = lab[seam_y:seam_y+1, :]
        
        # Color difference
        if above.shape[0] > 0 and below.shape[0] > 0:
            diff = np.abs(above.astype(float) - below.astype(float))
            cost = np.mean(diff)
            horizontal_seam_costs.append(cost)
    
    # Analyze vertical seams (between columns)
    for c in range(1, N):
        seam_x = c * tile_w
        left = lab[:, seam_x-1:seam_x]
        right = lab[:, seam_x:seam_x+1]
        
        if left.shape[1] > 0 and right.shape[1] > 0:
            diff = np.abs(left.astype(float) - right.astype(float))
            cost = np.mean(diff)
            vertical_seam_costs.append(cost)
    
    return {
        'avg_horizontal_seam_cost': np.mean(horizontal_seam_costs) if horizontal_seam_costs else 0,
        'avg_vertical_seam_cost': np.mean(vertical_seam_costs) if vertical_seam_costs else 0,
        'max_horizontal_seam_cost': np.max(horizontal_seam_costs) if horizontal_seam_costs else 0,
        'max_vertical_seam_cost': np.max(vertical_seam_costs) if vertical_seam_costs else 0,
        'avg_seam_cost': np.mean(horizontal_seam_costs + vertical_seam_costs) if horizontal_seam_costs + vertical_seam_costs else 0
    }


def visualize_seam_quality(assembled_img: np.ndarray, N: int, 
                          output_path: str):
    """
    Create heatmap visualization of seam quality.
    Red = poor match, Green = good match
    """
    h, w, _ = assembled_img.shape
    tile_h, tile_w = h // N, w // N
    
    lab = cv2.cvtColor(assembled_img, cv2.COLOR_BGR2LAB)
    vis = assembled_img.copy()
    
    # Overlay for seam quality
    overlay = np.zeros_like(vis)
    
    # Horizontal seams
    for r in range(1, N):
        seam_y = r * tile_h
        above = lab[seam_y-2:seam_y, :]
        below = lab[seam_y:seam_y+2, :]
        
        if above.shape[0] > 0 and below.shape[0] > 0:
            diff = np.abs(above.astype(float) - below.astype(float)).mean(axis=2)
            avg_diff = diff.mean(axis=0)
            
            # Map to color (green=good, red=bad)
            for x in range(w):
                val = min(avg_diff[x], 100) / 100.0
                color = (0, int(255 * (1-val)), int(255 * val))
                cv2.line(overlay, (x, seam_y-1), (x, seam_y+1), color, 2)
    
    # Vertical seams
    for c in range(1, N):
        seam_x = c * tile_w
        left = lab[:, seam_x-2:seam_x]
        right = lab[:, seam_x:seam_x+2]
        
        if left.shape[1] > 0 and right.shape[1] > 0:
            diff = np.abs(left.astype(float) - right.astype(float)).mean(axis=2)
            avg_diff = diff.mean(axis=1)
            
            for y in range(h):
                val = min(avg_diff[y], 100) / 100.0
                color = (0, int(255 * (1-val)), int(255 * val))
                cv2.line(overlay, (seam_x-1, y), (seam_x+1, y), color, 2)
    
    result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    cv2.imwrite(output_path, result)
    print(f"[VIS] Seam quality visualization: {output_path}")


# ================================
# 3. Edge Compatibility Matrix
# ================================

def create_compatibility_graph(tiles: List[Dict], N: int, 
                              output_path: str):
    """
    Create graph showing which tiles are most compatible.
    Useful for understanding puzzle structure.
    """
    from phase2_enhanced import prepare_tile_features, compute_matching_score
    
    # Get tile dimensions from first tile
    tile_h, tile_w = tiles[0]['img'].shape[:2]
    
    # Prepare features if not already done
    if 'border_contours' not in tiles[0]:
        prepare_tile_features(tiles, tile_h, tile_w)
    
    num_tiles = len(tiles)
    
    # Compute compatibility matrix for each side
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    sides = ['T', 'B', 'L', 'R']
    side_names = ['Top', 'Bottom', 'Left', 'Right']
    
    for idx, (side, name) in enumerate(zip(sides, side_names)):
        ax = axes[idx // 2, idx % 2]
        
        # Compute pairwise costs for this side
        cost_matrix = np.zeros((num_tiles, num_tiles))
        for i, t1 in enumerate(tiles):
            for j, t2 in enumerate(tiles):
                if i == j:
                    cost_matrix[i, j] = np.inf
                else:
                    cost = compute_matching_score(t1, side, t2, side)
                    cost_matrix[i, j] = cost
        
        # Plot
        im = ax.imshow(cost_matrix, cmap='hot_r', interpolation='nearest')
        ax.set_title(f'{name} Edge Compatibility')
        ax.set_xlabel('Tile ID')
        ax.set_ylabel('Tile ID')
        plt.colorbar(im, ax=ax, label='Matching Cost')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[VIS] Compatibility graph: {output_path}")


# ================================
# 4. Comparison Report
# ================================

def generate_comparison_report(results: List[Dict], 
                              output_path: str):
    """
    Generate comprehensive report comparing multiple assemblies.
    """
    report = []
    report.append("="*80)
    report.append("PUZZLE ASSEMBLY ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    # Group by grid size
    by_grid = {}
    for r in results:
        grid_size = r.get('grid_size', 0)
        if grid_size not in by_grid:
            by_grid[grid_size] = []
        by_grid[grid_size].append(r)
    
    for grid_size in sorted(by_grid.keys()):
        puzzles = by_grid[grid_size]
        report.append(f"\n{grid_size}x{grid_size} Puzzles ({len(puzzles)} total)")
        report.append("-"*80)
        
        avg_costs = [p.get('avg_cost', 0) for p in puzzles]
        
        report.append(f"  Average matching cost: {np.mean(avg_costs):.4f}")
        report.append(f"  Std deviation: {np.std(avg_costs):.4f}")
        report.append(f"  Min cost: {np.min(avg_costs):.4f}")
        report.append(f"  Max cost: {np.max(avg_costs):.4f}")
        report.append("")
        
        # Individual puzzles
        for p in puzzles:
            img_name = os.path.basename(p.get('image_path', ''))
            cost = p.get('avg_cost', 0)
            report.append(f"    {img_name:30s} cost={cost:.4f}")
    
    report.append("\n" + "="*80)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"[REPORT] Saved to: {output_path}")
    print(report_text)


# ================================
# 5. Main Analysis Pipeline
# ================================

def analyze_puzzle_results(phase2_root: str = "phase2_enhanced",
                          output_dir: str = "analysis_results"):
    """
    Comprehensive analysis of all assembled puzzles.
    """
    ensure_dir(output_dir)
    
    # Collect all results
    results = []
    
    for folder in ['puzzle_2x2', 'puzzle_4x4', 'puzzle_8x8']:
        folder_path = os.path.join(phase2_root, folder)
        if not os.path.isdir(folder_path):
            continue
        
        for puzzle_name in os.listdir(folder_path):
            puzzle_dir = os.path.join(folder_path, puzzle_name)
            if not os.path.isdir(puzzle_dir):
                continue
            
            # Find reconstructed image
            recon_files = [f for f in os.listdir(puzzle_dir) 
                          if f.startswith('reconstructed')]
            if not recon_files:
                continue
            
            recon_path = os.path.join(puzzle_dir, recon_files[0])
            assembled_img = cv2.imread(recon_path)
            
            if assembled_img is None:
                continue
            
            # Extract grid size from filename
            grid_size = int(folder.split('_')[1].split('x')[0])
            
            # Compute seam quality
            seam_metrics = compute_seam_quality(assembled_img, grid_size)
            
            # Visualize seam quality
            seam_vis_path = os.path.join(output_dir, 
                                        f"seam_quality_{folder}_{puzzle_name}.jpg")
            visualize_seam_quality(assembled_img, grid_size, seam_vis_path)
            
            results.append({
                'puzzle': puzzle_name,
                'folder': folder,
                'grid_size': grid_size,
                'seam_metrics': seam_metrics,
                'recon_path': recon_path
            })
    
    # Generate summary statistics
    summary_path = os.path.join(output_dir, "summary_statistics.txt")
    with open(summary_path, 'w') as f:
        f.write("SEAM QUALITY ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        for grid_size in [2, 4, 8]:
            puzzles = [r for r in results if r['grid_size'] == grid_size]
            if not puzzles:
                continue
            
            avg_costs = [p['seam_metrics']['avg_seam_cost'] for p in puzzles]
            
            f.write(f"{grid_size}x{grid_size} Puzzles:\n")
            f.write(f"  Avg seam cost: {np.mean(avg_costs):.2f}\n")
            f.write(f"  Std dev: {np.std(avg_costs):.2f}\n")
            f.write(f"  Range: [{np.min(avg_costs):.2f}, {np.max(avg_costs):.2f}]\n")
            f.write("\n")
    
    print(f"[ANALYSIS] Complete. Results saved to: {output_dir}")
    return results


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    # Run full analysis
    print("Starting puzzle assembly analysis...")
    results = analyze_puzzle_results()
    print(f"\nAnalyzed {len(results)} puzzles.")