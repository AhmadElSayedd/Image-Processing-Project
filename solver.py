# solver.py
import math
import random
from typing import Dict, Tuple, List

import numpy as np
from tiles import TOP, RIGHT, BOTTOM, LEFT, load_tiles, Tile

# (tid1, side1, tid2, side2) -> cost
CostKey = Tuple[int, int, int, int]
CostDict = Dict[CostKey, float]


def edge_cost(f1: np.ndarray, f2: np.ndarray) -> float:
    """Squared L2 distance between two edge feature vectors."""
    diff = f1 - f2
    return float(np.dot(diff, diff))


def build_compatibility(tiles: List[Tile]) -> CostDict:
    """
    Compute directional edge matching cost between every ordered pair of tiles.

    We don't assume rotations. For every (t1, t2):
      - RIGHT of t1 with LEFT of t2
      - LEFT  of t1 with RIGHT of t2
      - BOTTOM of t1 with TOP of t2
      - TOP    of t1 with BOTTOM of t2
    """
    costs: CostDict = {}
    print("[INFO] Building compatibility matrix ...")

    for t1 in tiles:
        for t2 in tiles:
            if t1.id == t2.id:
                continue

            # Right of t1 next to Left of t2
            c_rl = edge_cost(t1.edge_feats[RIGHT], t2.edge_feats[LEFT])
            costs[(t1.id, RIGHT, t2.id, LEFT)] = c_rl

            # Left of t1 next to Right of t2
            c_lr = edge_cost(t1.edge_feats[LEFT], t2.edge_feats[RIGHT])
            costs[(t1.id, LEFT, t2.id, RIGHT)] = c_lr

            # Bottom of t1 next to Top of t2
            c_bt = edge_cost(t1.edge_feats[BOTTOM], t2.edge_feats[TOP])
            costs[(t1.id, BOTTOM, t2.id, TOP)] = c_bt

            # Top of t1 next to Bottom of t2
            c_tb = edge_cost(t1.edge_feats[TOP], t2.edge_feats[BOTTOM])
            costs[(t1.id, TOP, t2.id, BOTTOM)] = c_tb

    print("[INFO] Compatibility matrix built.")
    return costs


def assignment_cost(perm: List[int], N: int, costs: CostDict) -> float:
    """
    Cost of a full board assignment defined by perm (length N*N).
    perm[k] = tile_id at flattened index k (row-major).

    Only adjacency costs (horiz & vert) are counted.
    """
    total = 0.0
    # board[r][c] = perm[r*N + c]
    for r in range(N):
        for c in range(N):
            tid = perm[r * N + c]

            # Horizontal neighbor (right)
            if c + 1 < N:
                tid_r = perm[r * N + (c + 1)]
                total += costs[(tid, RIGHT, tid_r, LEFT)]

            # Vertical neighbor (bottom)
            if r + 1 < N:
                tid_b = perm[(r + 1) * N + c]
                total += costs[(tid, BOTTOM, tid_b, TOP)]

    return total


def simulated_annealing_solve(tiles: List[Tile],
                              N: int,
                              costs: CostDict,
                              restarts: int = 80,
                              iters_per_restart: int = 4000) -> List[int]:
    """
    Global optimization over permutations using simulated annealing
    with multiple random restarts.

    Returns: best permutation of tile ids (length N*N).
    """
    random.seed(0)
    np.random.seed(0)

    tile_ids = [t.id for t in tiles]
    M = len(tile_ids)
    assert M == N * N, f"Expected {N*N} tiles, got {M}"

    global_best_perm = None
    global_best_cost = float("inf")

    for r in range(restarts):
        perm = tile_ids[:]  # copy
        random.shuffle(perm)
        cur_cost = assignment_cost(perm, N, costs)

        best_perm = perm[:]
        best_cost = cur_cost

        T0 = 5.0
        T_end = 0.1

        for it in range(iters_per_restart):
            i, j = random.sample(range(M), 2)
            new_perm = perm[:]
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]

            new_cost = assignment_cost(new_perm, N, costs)
            delta = new_cost - cur_cost

            # Exponential cooling schedule
            t = it / max(1, (iters_per_restart - 1))
            T = T0 * (T_end / T0) ** t

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-6)):
                perm = new_perm
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_perm = perm[:]

        print(f"[INFO] Restart {r+1}/{restarts}: best cost = {best_cost:.2f}")

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_perm = best_perm

    print(f"[INFO] Global best cost = {global_best_cost:.2f}")
    return global_best_perm


def solve_puzzle(tiles_folder: str, N: int):
    """
    High-level API: load tiles, build costs, solve with SA.
    Returns:
        tiles (list[Tile]),
        perm  (list of tile ids in row-major order)
    """
    tiles = load_tiles(tiles_folder)
    costs = build_compatibility(tiles)
    perm = simulated_annealing_solve(tiles, N, costs)
    return tiles, perm
