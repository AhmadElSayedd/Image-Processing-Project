# src/features.py
import os
import cv2
import numpy as np
from glob import glob
from typing import Dict

from config import TILES_2_DIR, TILES_4_DIR, TILES_8_DIR, VISUALS_DIR, ensure_dirs


def extract_edge_strips(tile: np.ndarray,
                        strip_width: int = 3) -> Dict[str, np.ndarray]:
    """
    Given a tile (H x W x 3), extract 4 border strips:
    - 'top', 'right', 'bottom', 'left'
    Each strip is a small region near the border.
    """
    h, w = tile.shape[:2]
    w_strip = strip_width

    # Make sure strip width is not too big
    w_strip = min(w_strip, h // 2, w // 2)

    top = tile[0:w_strip, :, :]           # strip along first rows
    bottom = tile[h-w_strip:h, :, :]      # strip along last rows
    left = tile[:, 0:w_strip, :]          # strip along first columns
    right = tile[:, w-w_strip:w, :]       # strip along last columns

    return {
        "top": top,
        "right": right,
        "bottom": bottom,
        "left": left,
    }


def visualize_tile_edges(tile_path: str,
                         strip_width: int = 3,
                         out_path: str = None):
    """
    Create a simple visualization showing the tile and the 4 extracted strips.
    Good for the Milestone 1 report.
    """
    import matplotlib.pyplot as plt

    tile = cv2.imread(tile_path)
    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    edges = extract_edge_strips(tile, strip_width=strip_width)

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 3, 2)
    plt.title("Tile")
    plt.imshow(tile_rgb)
    plt.axis("off")

    # Show strips
    edge_names = ["top", "right", "bottom", "left"]
    positions = [4, 3, 5, 1]  # somewhat around the central tile

    for name, pos in zip(edge_names, positions):
        strip = edges[name]
        strip_rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 3, pos)
        plt.title(name)
        plt.imshow(strip_rgb)
        plt.axis("off")

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.show()

def run_feature_extraction():
    ensure_dirs()
    # Only extracting features for 2x2, 4x4, 8x8 tiles
    # You can add logic for debug visual too if needed
    example_tiles = (
        glob(os.path.join(TILES_4_DIR, "*.png")) +
        glob(os.path.join(TILES_4_DIR, "*.jpg"))
    )
    if example_tiles:
        example = example_tiles[0]
        out = os.path.join(VISUALS_DIR, "tile_edges_example.png")
        visualize_tile_edges(example, strip_width=3, out_path=out)

if __name__ == "__main__":
    ensure_dirs()

    # Pick an example tile to visualize
    # You can change this to any existing tile path.
    example_tiles = (
        glob(os.path.join(TILES_4_DIR, "*.png")) +
        glob(os.path.join(TILES_4_DIR, "*.jpg"))
    )
    if example_tiles:
        example = example_tiles[0]
        out = os.path.join(VISUALS_DIR, "tile_edges_example.png")
        visualize_tile_edges(example, strip_width=3, out_path=out)

