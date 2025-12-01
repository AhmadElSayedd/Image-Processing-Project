import os
import numpy as np
from phase2_enhanced import solve_full_dataset_from_phase1


def summarize_results(results):
    if not results:
        print("[WARN] No results to summarize.")
        return

    # Group by grid size
    by_grid = {}
    for r in results:
        N = r["grid_size"]
        by_grid.setdefault(N, []).append(r["avg_neighbor_cost"])

    print("\n============================")
    print(" PHASE-2 PERFORMANCE METRICS")
    print("============================\n")

    all_costs = []

    for N in sorted(by_grid.keys()):
        costs = np.array(by_grid[N], dtype=np.float32)
        all_costs.extend(costs.tolist())
        print(f"Grid {N}x{N}:")
        print(f"  # puzzles: {len(costs)}")
        print(f"  avg neighbor cost: {costs.mean():.2f}")
        print(f"  min neighbor cost: {costs.min():.2f}")
        print(f"  max neighbor cost: {costs.max():.2f}")
        print()

    if all_costs:
        all_costs = np.array(all_costs, dtype=np.float32)
        print("Overall:")
        print(f"  Total puzzles: {len(all_costs)}")
        print(f"  Global avg neighbor cost: {all_costs.mean():.2f}")
        print(f"  Global min neighbor cost: {all_costs.min():.2f}")
        print(f"  Global max neighbor cost: {all_costs.max():.2f}")
    print()


if __name__ == "__main__":
    # This will internally call solve_puzzle_from_phase1 for each puzzle
    results = solve_full_dataset_from_phase1()
    summarize_results(results)
