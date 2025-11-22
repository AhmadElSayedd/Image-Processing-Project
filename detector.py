import cv2
import numpy as np

# ----------------------------
# 1) Utilities: slicing + edges
# ----------------------------

def slice_tiles(img, N):
    H, W = img.shape[:2]
    th, tw = H // N, W // N
    tiles = []
    for r in range(N):
        for c in range(N):
            tile = img[r*th:(r+1)*th, c*tw:(c+1)*tw]
            tiles.append(tile)
    return tiles

def get_edges(tile_lab, k=4):
    # tile_lab: (h,w,3)
    top    = tile_lab[:k, :, :]
    bottom = tile_lab[-k:, :, :]
    left   = tile_lab[:, :k, :]
    right  = tile_lab[:, -k:, :]
    return top, right, bottom, left  # order: 0,1,2,3

def edge_descriptor(edge_strip):
    """
    Make edge strip robust:
    - slight blur
    - normalize per-channel (zero-mean / unit-std)
    """
    e = cv2.GaussianBlur(edge_strip, (3,3), 0).astype(np.float32)
    mu = e.mean(axis=(0,1), keepdims=True)
    sd = e.std(axis=(0,1), keepdims=True) + 1e-6
    e = (e - mu) / sd
    return e

def edge_dist(eA, eB):
    """SSD distance on normalized Lab strips."""
    diff = eA - eB
    return float(np.mean(diff * diff))

# ----------------------------
# 2) Compatibility scoring for one N
# ----------------------------

def compute_compatibility_score(tiles, k=4):
    """
    Returns a grid-consistency score for a given tiling.
    Uses:
      - mutual best buddies
      - ratio test best/second-best
      - global clarity of matches
    """

    n_tiles = len(tiles)
    # convert to Lab for illumination robustness
    tiles_lab = [cv2.cvtColor(t, cv2.COLOR_BGR2LAB) for t in tiles]

    # precompute normalized edge descriptors
    edges = []
    for t in tiles_lab:
        raw = get_edges(t, k=k)
        desc = tuple(edge_descriptor(x) for x in raw)
        edges.append(desc)

    opposite = {0:2, 1:3, 2:0, 3:1}

    # best match index and distances, also second-best
    best_idx = np.full((n_tiles, 4), -1, dtype=int)
    best_d   = np.full((n_tiles, 4), np.inf, dtype=np.float32)
    second_d = np.full((n_tiles, 4), np.inf, dtype=np.float32)

    # compare every tile edge against every other tile opposite edge
    for i in range(n_tiles):
        for d in range(4):
            e_i = edges[i][d]
            od = opposite[d]
            for j in range(n_tiles):
                if i == j: 
                    continue
                e_j = edges[j][od]
                dij = edge_dist(e_i, e_j)

                if dij < best_d[i, d]:
                    second_d[i, d] = best_d[i, d]
                    best_d[i, d] = dij
                    best_idx[i, d] = j
                elif dij < second_d[i, d]:
                    second_d[i, d] = dij

    # ---- Mutual best-buddy count + ratio-test confidence ----
    mutual_confident = 0
    mutual_total = 0
    confident_total = 0
    ratios = []

    for i in range(n_tiles):
        for d in range(4):
            j = best_idx[i, d]
            if j == -1:
                continue
            od = opposite[d]

            # ratio test: best must be significantly better than second best
            # smaller ratio = clearer match
            r = best_d[i, d] / (second_d[i, d] + 1e-9)
            ratios.append(r)
            if r < 0.75:
                confident_total += 1

            # mutual best-buddy?
            if best_idx[j, od] == i:
                mutual_total += 1
                if r < 0.75:
                    mutual_confident += 1

    total_edges = n_tiles * 4

    # ---- Build score components ----
    # 1) Mutual confident buddies rate
    mutual_rate = mutual_confident / (total_edges + 1e-9)

    # 2) Confidence (how many edges pass ratio test)
    conf_rate = confident_total / (total_edges + 1e-9)

    # 3) Clarity: median ratio (lower is better)
    med_ratio = np.median(ratios) if len(ratios) else 1.0
    clarity = 1.0 / (med_ratio + 1e-6)

    # Combined score (weights tuned to avoid 2x2 bias)
    score = (0.55 * mutual_rate) + (0.25 * conf_rate) + (0.20 * clarity)

    return score, {
        "mutual_rate": mutual_rate,
        "conf_rate": conf_rate,
        "clarity": clarity,
        "med_ratio": med_ratio
    }

# ----------------------------
# 3) Final grid-size inference
# ----------------------------

def detect_grid_size_unknown(img, candidates=(2,4,8), k=4, debug=False):
    """
    Infer grid size WITHOUT knowing it.
    Tries each candidate N, scores compatibility, picks best.

    Includes a mild size prior so 2x2 can't win unless clearly better.
    """

    scores = {}
    details = {}

    for N in candidates:
        tiles = slice_tiles(img, N)
        s, d = compute_compatibility_score(tiles, k=k)

        # Mild prior favoring larger N only slightly
        # (does NOT force it, just breaks ties fairly)
        prior = (N / 2.0) ** 0.35
        s_final = s * prior

        scores[N] = s_final
        details[N] = (s, s_final, d)

        if debug:
            print(f"\nN={N}")
            print(f"  raw score     = {s:.4f}")
            print(f"  prior         = {prior:.3f}")
            print(f"  final score   = {s_final:.4f}")
            print(f"  components    = {d}")

    bestN = max(scores, key=scores.get)

    if debug:
        print("\nFinal scores:", {N: round(scores[N],4) for N in scores})
        print("Chosen grid size:", bestN)

    return bestN


img = cv2.imread("Gravity Falls/puzzle_2x2/0.jpg")
print(detect_grid_size_unknown(img, debug=True))