import os
import glob
import cv2
import numpy as np

from .config import tile_ext


class PuzzleAssembler:
    def __init__(self, tiles_folder: str, n: int):
        self.n = n
        self.tiles = []

        # Load all tiles in this folder
        tile_paths = sorted(glob.glob(os.path.join(tiles_folder, f"*{tile_ext}")))
        for idx, path in enumerate(tile_paths):
            img = cv2.imread(path)
            if img is None:
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
            self.tiles.append({
                "id": idx,
                "img": img,
                "lab": lab,
            })

    def _edge_cost(self, t1, t2, direction: int) -> float:
        """Compute matching cost between two tiles along one side.

        direction = 0 → t1 right edge vs t2 left edge (horizontal neighbor)
        direction = 1 → t1 bottom edge vs t2 top edge (vertical neighbor)
        """
        CROP = 1  # ignore 1px border to reduce noise

        if direction == 0:  # horizontal
            e1 = t1["lab"][CROP:-CROP, -1, :]
            e2 = t2["lab"][CROP:-CROP, 0, :]
        else:  # vertical
            e1 = t1["lab"][-1, CROP:-CROP, :]
            e2 = t2["lab"][0, CROP:-CROP, :]

        diff = e1 - e2
        return float(np.mean(np.sqrt(np.sum(diff * diff, axis=1))))

    def _solve_2x2_bruteforce(self):
        """Exhaustive search for best 2×2 arrangement (4! = 24 permutations)."""
        import itertools

        if len(self.tiles) != 4:
            return None

        best_grid = None
        best_score = float("inf")
        indices = list(range(4))

        for perm in itertools.permutations(indices):
            # layout:
            # [ p0 p1 ]
            # [ p2 p3 ]
            score = 0.0
            score += self._edge_cost(self.tiles[perm[0]], self.tiles[perm[1]], direction=0)
            score += self._edge_cost(self.tiles[perm[2]], self.tiles[perm[3]], direction=0)
            score += self._edge_cost(self.tiles[perm[0]], self.tiles[perm[2]], direction=1)
            score += self._edge_cost(self.tiles[perm[1]], self.tiles[perm[3]], direction=1)

            if score < best_score:
                best_score = score
                best_grid = [[perm[0], perm[1]],
                             [perm[2], perm[3]]]

        return best_grid

    def _solve_greedy_soft_buddies(self):
        """Greedy solver for 4×4 and 8×8 using soft best-buddy constraints."""
        nt = len(self.tiles)
        if nt != self.n * self.n:
            return None

        # 1) Precompute costs for every ordered pair (i, j) in both directions
        costs = np.full((nt, nt, 2), np.inf, dtype=np.float32)
        for i in range(nt):
            for j in range(nt):
                if i == j:
                    continue
                costs[i, j, 0] = self._edge_cost(self.tiles[i], self.tiles[j], direction=0)  # right
                costs[i, j, 1] = self._edge_cost(self.tiles[i], self.tiles[j], direction=1)  # down

        # 2) Best neighbor indices
        best_right = np.argmin(costs[:, :, 0], axis=1)  # who is best to the right of i
        best_left  = np.argmin(costs[:, :, 0], axis=0)  # who is best to the left of j
        best_down  = np.argmin(costs[:, :, 1], axis=1)  # who is best below i
        best_up    = np.argmin(costs[:, :, 1], axis=0)  # who is best above j

        best_grid = None
        best_score = float("inf")

        # 3) Try each tile as starting point of (0,0)
        for start in range(nt):
            grid = [[None for _ in range(self.n)] for _ in range(self.n)]
            used = set([start])
            grid[0][0] = start
            total_score = 0.0
            failed = False

            for r in range(self.n):
                for c in range(self.n):
                    if r == 0 and c == 0:
                        continue

                    left_idx = grid[r][c - 1] if c > 0 else None
                    up_idx   = grid[r - 1][c] if r > 0 else None

                    best_candidate = None
                    best_local = float("inf")

                    for cand in range(nt):
                        if cand in used:
                            continue

                        cost_sum = 0.0
                        count = 0

                        if left_idx is not None:
                            cost_sum += float(costs[left_idx, cand, 0])
                            count += 1
                        if up_idx is not None:
                            cost_sum += float(costs[up_idx, cand, 1])
                            count += 1

                        if count == 0:
                            continue

                        avg_cost = cost_sum / count

                        # soft best-buddy bonus
                        is_buddy = True
                        if left_idx is not None:
                            if not (best_right[left_idx] == cand and best_left[cand] == left_idx):
                                is_buddy = False
                        if up_idx is not None:
                            if not (best_down[up_idx] == cand and best_up[cand] == up_idx):
                                is_buddy = False

                        final_cost = avg_cost * 0.1 if is_buddy else avg_cost

                        if final_cost < best_local:
                            best_local = final_cost
                            best_candidate = cand

                    if best_candidate is None:
                        failed = True
                        break

                    grid[r][c] = best_candidate
                    used.add(best_candidate)
                    total_score += best_local

                if failed:
                    break

            if not failed and total_score < best_score:
                best_score = total_score
                best_grid = grid

        return best_grid

    def solve(self):
        if self.n == 2:
            return self._solve_2x2_bruteforce()
        else:
            return self._solve_greedy_soft_buddies()

    def reconstruct_image(self, grid):
        """Rebuild full puzzle image from grid of tile indices."""
        if grid is None:
            return None

        rows = []
        for r in range(self.n):
            row_tiles = [self.tiles[grid[r][c]]["img"] for c in range(self.n)]
            rows.append(np.hstack(row_tiles))

        full_img = np.vstack(rows)
        return full_img
