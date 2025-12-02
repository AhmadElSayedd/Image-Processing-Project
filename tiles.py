# tiles.py
import os
import cv2
import numpy as np

# Directions (no rotation)
TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
DIR_NAMES = {TOP: "TOP", RIGHT: "RIGHT", BOTTOM: "BOTTOM", LEFT: "LEFT"}


class Tile:
    """
    Represents one puzzle tile.
    - id: integer index
    - img: BGR uint8 image
    - edge_feats: dict[side] -> 1D feature vector
    """

    def __init__(self, tile_id: int, img_bgr: np.ndarray):
        self.id = tile_id
        self.img = img_bgr
        self.edge_feats = {}
        self._compute_all_edges()

    def _compute_all_edges(self):
        for side in (TOP, RIGHT, BOTTOM, LEFT):
            self.edge_feats[side] = compute_edge_feature(self.img, side)


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    mean = v.mean()
    std = v.std()
    if std < 1e-6:
        return v * 0.0
    return (v - mean) / (std + 1e-6)


def compute_edge_feature(img_bgr: np.ndarray, side: int, strip_width: int = 6) -> np.ndarray:
    """
    Multi-channel edge feature along one side of the tile.

    Steps:
      1) Convert to Lab (better perceptual color).
      2) Take a small strip at the border (top/bottom or left/right).
      3) Average across strip thickness -> a 1D profile along the edge.
      4) Concatenate L,a,b + gradient magnitude profile.
      5) Normalize.

    Returns: 1D np.array (float32)
    """
    h, w, _ = img_bgr.shape
    strip_width = max(2, min(strip_width, h // 4, w // 4))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)

    # Gradient on L channel
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    G = cv2.magnitude(gx, gy)

    if side == TOP:
        s = slice(0, strip_width)
        Lp = L[s, :].mean(axis=0)
        Ap = A[s, :].mean(axis=0)
        Bp = B[s, :].mean(axis=0)
        Gp = G[s, :].mean(axis=0)
    elif side == BOTTOM:
        s = slice(h - strip_width, h)
        Lp = L[s, :].mean(axis=0)
        Ap = A[s, :].mean(axis=0)
        Bp = B[s, :].mean(axis=0)
        Gp = G[s, :].mean(axis=0)
    elif side == LEFT:
        s = slice(0, strip_width)
        Lp = L[:, s].mean(axis=1)
        Ap = A[:, s].mean(axis=1)
        Bp = B[:, s].mean(axis=1)
        Gp = G[:, s].mean(axis=1)
    elif side == RIGHT:
        s = slice(w - strip_width, w)
        Lp = L[:, s].mean(axis=1)
        Ap = A[:, s].mean(axis=1)
        Bp = B[:, s].mean(axis=1)
        Gp = G[:, s].mean(axis=1)
    else:
        raise ValueError("Invalid side")

    # Normalize each profile and concatenate
    Lp = _normalize_vec(Lp)
    Ap = _normalize_vec(Ap)
    Bp = _normalize_vec(Bp)
    Gp = _normalize_vec(Gp)

    feat = np.concatenate([Lp, Ap, Bp, Gp]).astype(np.float32)
    return feat


def load_tiles(folder: str):
    """
    Load all tiles (PNG/JPG) from folder.
    Assign ids in sorted filename order.
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    if not files:
        raise RuntimeError(f"No image files found in {folder}")

    files.sort()  # ensures deterministic mapping filename -> id
    tiles = []

    for idx, fname in enumerate(files):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read image: {path}")
            continue
        tiles.append(Tile(tile_id=idx, img_bgr=img))

    if not tiles:
        raise RuntimeError(f"Failed to load any valid tiles from {folder}")

    print(f"[INFO] Loaded {len(tiles)} tiles from {folder}")
    return tiles
