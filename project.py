"""
Jigsaw Puzzle Solver - Works with Preprocessed Tiles

This solver loads preprocessed tiles and assembles puzzles using:
- Simulated Annealing for 2x2 (high accuracy)
- Greedy Best-Buddy for 4x4 and 8x8 (scalable)
"""

import os
import glob
import re
import math
import random
import sys
import itertools
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, desc="": x

# ======================================
#   CONFIGURATION
# ======================================
DATASET_DIR = r"C:\Github Repositories\Image-Processing-Project\Gravity Falls"
PREPROCESSED_TILES_DIR = os.path.join(DATASET_DIR, "tiles")
ASSEMBLED_OUTPUT_DIR = os.path.join(DATASET_DIR, "assembled_final")
CORRECT_DIR = os.path.join(DATASET_DIR, "correct")

TILE_EXT = ".png"
PUZZLE_EXT = ".jpg"

# Edge directions
TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3

# ======================================
#   TILE CLASS & FEATURES
# ======================================
class Tile:
    def __init__(self, tile_id: int, img_bgr: np.ndarray):
        self.id = tile_id
        self.img = img_bgr
        self.edge_feats = self._compute_all_edges()

    def _compute_all_edges(self):
        return {side: compute_edge_feature(self.img, side) 
                for side in (TOP, RIGHT, BOTTOM, LEFT)}


def compute_edge_feature(img_bgr: np.ndarray, side: int, strip_width: int = 6) -> np.ndarray:
    """Extract multi-channel edge feature (Lab + gradient) from one side of tile."""
    h, w = img_bgr.shape[:2]
    strip_width = max(2, min(strip_width, h // 4, w // 4))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)

    # Gradient on L channel
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    G = cv2.magnitude(gx, gy)

    # Extract strips based on side
    if side == TOP:
        strips = [ch[0:strip_width, :].mean(axis=0) for ch in (L, A, B, G)]
    elif side == BOTTOM:
        strips = [ch[h-strip_width:h, :].mean(axis=0) for ch in (L, A, B, G)]
    elif side == LEFT:
        strips = [ch[:, 0:strip_width].mean(axis=1) for ch in (L, A, B, G)]
    else:  # RIGHT
        strips = [ch[:, w-strip_width:w].mean(axis=1) for ch in (L, A, B, G)]

    # Normalize and concatenate
    normalized = []
    for s in strips:
        s = s.astype(np.float32)
        std = s.std()
        normalized.append((s - s.mean()) / (std + 1e-6) if std > 1e-6 else s * 0.0)
    
    return np.concatenate(normalized).astype(np.float32)


def load_tiles(folder: str):
    """Load all tiles from folder and create Tile objects."""
    files = sorted([f for f in os.listdir(folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    if not files:
        raise RuntimeError(f"No images in {folder}")

    tiles = []
    for idx, fname in enumerate(files):
        img = cv2.imread(os.path.join(folder, fname))
        if img is not None:
            tiles.append(Tile(idx, img))

    print(f"[INFO] Loaded {len(tiles)} tiles")
    return tiles


# ======================================
#   SIMULATED ANNEALING SOLVER (2x2)
# ======================================
def edge_cost(f1: np.ndarray, f2: np.ndarray) -> float:
    """Squared L2 distance between edge features."""
    diff = f1 - f2
    return float(np.dot(diff, diff))


def build_compatibility(tiles):
    """Precompute all edge matching costs."""
    costs = {}
    for t1 in tiles:
        for t2 in tiles:
            if t1.id == t2.id:
                continue
            costs[(t1.id, RIGHT, t2.id, LEFT)] = edge_cost(t1.edge_feats[RIGHT], t2.edge_feats[LEFT])
            costs[(t1.id, LEFT, t2.id, RIGHT)] = edge_cost(t1.edge_feats[LEFT], t2.edge_feats[RIGHT])
            costs[(t1.id, BOTTOM, t2.id, TOP)] = edge_cost(t1.edge_feats[BOTTOM], t2.edge_feats[TOP])
            costs[(t1.id, TOP, t2.id, BOTTOM)] = edge_cost(t1.edge_feats[TOP], t2.edge_feats[BOTTOM])
    return costs


def assignment_cost(perm, N: int, costs: dict) -> float:
    """Calculate total cost of a tile arrangement."""
    total = 0.0
    for r in range(N):
        for c in range(N):
            tid = perm[r * N + c]
            if c + 1 < N:
                total += costs[(tid, RIGHT, perm[r * N + c + 1], LEFT)]
            if r + 1 < N:
                total += costs[(tid, BOTTOM, perm[(r + 1) * N + c], TOP)]
    return total


def simulated_annealing_solve(tiles, N: int, costs: dict, 
                              restarts: int = 25, iters: int = 2000):
    """Solve puzzle using simulated annealing with multiple restarts."""
    random.seed(0)
    tile_ids = [t.id for t in tiles]
    
    best_perm, best_cost = None, float("inf")

    for r in range(restarts):
        perm = tile_ids[:]
        random.shuffle(perm)
        cur_cost = assignment_cost(perm, N, costs)
        restart_best_perm, restart_best_cost = perm[:], cur_cost

        T0, T_end = 5.0, 0.1

        for it in range(iters):
            i, j = random.sample(range(N * N), 2)
            new_perm = perm[:]
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            new_cost = assignment_cost(new_perm, N, costs)
            
            T = T0 * (T_end / T0) ** (it / max(1, iters - 1))
            
            if new_cost < cur_cost or random.random() < math.exp(-(new_cost - cur_cost) / max(T, 1e-6)):
                perm, cur_cost = new_perm, new_cost
                if cur_cost < restart_best_cost:
                    restart_best_perm, restart_best_cost = perm[:], cur_cost

        print(f"[INFO] SA Restart {r+1}/{restarts}: cost = {restart_best_cost:.2f}")
        if restart_best_cost < best_cost:
            best_perm, best_cost = restart_best_perm, restart_best_cost

    return best_perm


# ======================================
#   GREEDY ASSEMBLER (4x4, 8x8)
# ======================================
class PuzzleAssembler:
    def __init__(self, tiles_folder: str, n: int):
        self.n = n
        self.tiles = []

        for idx, path in enumerate(sorted(glob.glob(os.path.join(tiles_folder, f"*{TILE_EXT}")))):
            img = cv2.imread(path)
            if img is not None:
                self.tiles.append({
                    "id": idx,
                    "img": img,
                    "lab": cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
                })

    def _edge_cost(self, t1, t2, direction: int) -> float:
        """Compute edge matching cost (0=horizontal, 1=vertical)."""
        CROP = 1
        if direction == 0:  # horizontal
            e1, e2 = t1["lab"][CROP:-CROP, -1, :], t2["lab"][CROP:-CROP, 0, :]
        else:  # vertical
            e1, e2 = t1["lab"][-1, CROP:-CROP, :], t2["lab"][0, CROP:-CROP, :]
        
        diff = e1 - e2
        return float(np.mean(np.sqrt(np.sum(diff * diff, axis=1))))

    def _solve_2x2(self):
        """Brute force for 2×2."""
        if len(self.tiles) != 4:
            return None

        best_grid, best_score = None, float("inf")
        
        for perm in itertools.permutations(range(4)):
            score = (self._edge_cost(self.tiles[perm[0]], self.tiles[perm[1]], 0) +
                    self._edge_cost(self.tiles[perm[2]], self.tiles[perm[3]], 0) +
                    self._edge_cost(self.tiles[perm[0]], self.tiles[perm[2]], 1) +
                    self._edge_cost(self.tiles[perm[1]], self.tiles[perm[3]], 1))
            
            if score < best_score:
                best_score = score
                best_grid = [[perm[0], perm[1]], [perm[2], perm[3]]]
        
        return best_grid

    def _solve_greedy(self):
        """Greedy solver with soft best-buddy constraints."""
        nt = len(self.tiles)
        if nt != self.n * self.n:
            return None

        # Precompute costs
        costs = np.full((nt, nt, 2), np.inf, dtype=np.float32)
        for i in range(nt):
            for j in range(nt):
                if i != j:
                    costs[i, j, 0] = self._edge_cost(self.tiles[i], self.tiles[j], 0)
                    costs[i, j, 1] = self._edge_cost(self.tiles[i], self.tiles[j], 1)

        # Best neighbors
        best_right = np.argmin(costs[:, :, 0], axis=1)
        best_left = np.argmin(costs[:, :, 0], axis=0)
        best_down = np.argmin(costs[:, :, 1], axis=1)
        best_up = np.argmin(costs[:, :, 1], axis=0)

        best_grid, best_score = None, float("inf")

        for start in range(nt):
            grid = [[None] * self.n for _ in range(self.n)]
            used = {start}
            grid[0][0] = start
            total_score = 0.0
            failed = False

            for r in range(self.n):
                for c in range(self.n):
                    if r == 0 and c == 0:
                        continue

                    left_idx = grid[r][c-1] if c > 0 else None
                    up_idx = grid[r-1][c] if r > 0 else None

                    best_cand, best_local = None, float("inf")

                    for cand in range(nt):
                        if cand in used:
                            continue

                        cost_sum, count = 0.0, 0
                        if left_idx is not None:
                            cost_sum += costs[left_idx, cand, 0]
                            count += 1
                        if up_idx is not None:
                            cost_sum += costs[up_idx, cand, 1]
                            count += 1

                        if count == 0:
                            continue

                        avg_cost = cost_sum / count

                        # Best-buddy bonus
                        is_buddy = True
                        if left_idx is not None and not (best_right[left_idx] == cand and best_left[cand] == left_idx):
                            is_buddy = False
                        if up_idx is not None and not (best_down[up_idx] == cand and best_up[cand] == up_idx):
                            is_buddy = False

                        final_cost = avg_cost * 0.1 if is_buddy else avg_cost

                        if final_cost < best_local:
                            best_local = final_cost
                            best_cand = cand

                    if best_cand is None:
                        failed = True
                        break

                    grid[r][c] = best_cand
                    used.add(best_cand)
                    total_score += best_local

                if failed:
                    break

            if not failed and total_score < best_score:
                best_score = total_score
                best_grid = grid

        return best_grid

    def solve(self):
        return self._solve_2x2() if self.n == 2 else self._solve_greedy()

    def reconstruct_image(self, grid):
        """Rebuild puzzle from grid."""
        if grid is None:
            return None
        rows = [np.hstack([self.tiles[grid[r][c]]["img"] for c in range(self.n)]) 
                for r in range(self.n)]
        return np.vstack(rows)


# ======================================
#   EVALUATION
# ======================================
def evaluate_accuracy():
    print("\n[EVAL] Evaluating accuracy...")
    stats = {}
    
    puzzle_types = ["2x2", "4x4", "8x8"]

    for grid_str in puzzle_types:
        n = int(grid_str[0])
        puzzle_folder = f"puzzle_{grid_str}"
        puzzle_path = os.path.join(DATASET_DIR, puzzle_folder)
        solved_dir = os.path.join(ASSEMBLED_OUTPUT_DIR, grid_str)

        if not os.path.isdir(puzzle_path):
            continue

        total = len(glob.glob(os.path.join(puzzle_path, f"*{PUZZLE_EXT}")))
        correct = 0

        if os.path.isdir(solved_dir):
            for gt_path in glob.glob(os.path.join(CORRECT_DIR, "*.png")):
                gt_id = int(re.findall(r"\d+", os.path.basename(gt_path))[-1])
                
                match_path = None
                for solved_path in glob.glob(os.path.join(solved_dir, "*.png")):
                    if int(re.findall(r"\d+", os.path.basename(solved_path))[-1]) == gt_id:
                        match_path = solved_path
                        break

                if match_path:
                    gt_img = cv2.imread(gt_path)
                    solved_img = cv2.imread(match_path)
                    if gt_img is not None and solved_img is not None:
                        if gt_img.shape != solved_img.shape:
                            gt_img = cv2.resize(gt_img, (solved_img.shape[1], solved_img.shape[0]))
                        mse = np.mean((gt_img.astype("float32") - solved_img.astype("float32")) ** 2)
                        if mse < 2500.0:
                            correct += 1

        acc = (correct / total * 100.0) if total > 0 else 0.0
        stats[puzzle_folder] = acc
        print(f"  {puzzle_folder}: {correct}/{total} ({acc:.2f}%)")

    # Plot results
    if stats:
        plt.figure(figsize=(8, 5))
        labels, values = list(stats.keys()), list(stats.values())
        bars = plt.bar(labels, values)
        plt.ylim(0, 105)
        plt.ylabel("Accuracy (%)")
        plt.title("Puzzle Solver Accuracy (Preprocessed)")
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, val+1, f"{val:.1f}%", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join(DATASET_DIR, "final_score.png"))
        plt.close()


# ======================================
#   MAIN SOLVING PIPELINE
# ======================================
def solve_all_puzzles():
    print("=== Solving Puzzles (Using Preprocessed Tiles) ===")

    # Clean output directory
    if os.path.isdir(ASSEMBLED_OUTPUT_DIR):
        shutil.rmtree(ASSEMBLED_OUTPUT_DIR)
    os.makedirs(ASSEMBLED_OUTPUT_DIR, exist_ok=True)

    grid_sizes = ["2x2", "4x4", "8x8"]

    for grid_str in grid_sizes:
        n = int(grid_str[0])
        tiles_base = os.path.join(PREPROCESSED_TILES_DIR, grid_str)
        
        if not os.path.isdir(tiles_base):
            print(f"[WARN] No preprocessed tiles found for {grid_str}")
            continue

        # Find all puzzle directories
        puzzle_dirs = [d for d in os.listdir(tiles_base) 
                      if os.path.isdir(os.path.join(tiles_base, d))]
        
        print(f"\n[INFO] {grid_str}: found {len(puzzle_dirs)} puzzles")

        out_subdir = os.path.join(ASSEMBLED_OUTPUT_DIR, grid_str)
        os.makedirs(out_subdir, exist_ok=True)

        for puzzle_name in tqdm(puzzle_dirs, desc=f"Solving {grid_str}"):
            tiles_folder = os.path.join(tiles_base, puzzle_name)

            try:
                # Solve based on grid size
                if n == 2:
                    tiles = load_tiles(tiles_folder)
                    costs = build_compatibility(tiles)
                    perm = simulated_annealing_solve(tiles, n, costs)
                    
                    # Reconstruct from SA solution
                    tile_dict = {t.id: t for t in tiles}
                    h, w = tiles[0].img.shape[:2]
                    solved_img = np.zeros((n*h, n*w, 3), dtype=np.uint8)
                    for r in range(n):
                        for c in range(n):
                            tid = perm[r*n + c]
                            solved_img[r*h:(r+1)*h, c*w:(c+1)*w] = tile_dict[tid].img
                else:
                    assembler = PuzzleAssembler(tiles_folder, n)
                    grid = assembler.solve()
                    solved_img = assembler.reconstruct_image(grid)

                if solved_img is not None:
                    output_path = os.path.join(out_subdir, f"{puzzle_name}.png")
                    cv2.imwrite(output_path, solved_img)
                else:
                    print(f"[FAIL] {puzzle_name}")

            except Exception as e:
                print(f"[ERROR] {puzzle_name}: {e}")

    print("\n=== Solving Complete ===")


def main():
    global DATASET_DIR, PREPROCESSED_TILES_DIR, ASSEMBLED_OUTPUT_DIR, CORRECT_DIR
    
    if len(sys.argv) > 1:
        DATASET_DIR = sys.argv[1]
        PREPROCESSED_TILES_DIR = os.path.join(DATASET_DIR, "preprocessed_tiles")
        ASSEMBLED_OUTPUT_DIR = os.path.join(DATASET_DIR, "assembled_final")
        CORRECT_DIR = os.path.join(DATASET_DIR, "correct")

    # Check if preprocessed tiles exist
    if not os.path.isdir(PREPROCESSED_TILES_DIR):
        print(f"[ERROR] Preprocessed tiles not found at: {PREPROCESSED_TILES_DIR}")
        print("Please run the preprocessing script first:")
        print("  python preprocessing.py")
        sys.exit(1)

    solve_all_puzzles()

    if os.path.isdir(CORRECT_DIR):
        evaluate_accuracy()


if __name__ == "__main__":
    main()