import os
import cv2
import numpy as np

TILE_SIZE = 128  # chosen resolution


def extract_tiles(image_path, n, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize to fixed target resolution
    target_size = (n * TILE_SIZE, n * TILE_SIZE)
    img_resized = cv2.resize(img, target_size)

    # Cut equally
    tiles = []
    idx = 0
    for r in range(n):
        for c in range(n):
            tile = img_resized[r*TILE_SIZE:(r+1)*TILE_SIZE,
                               c*TILE_SIZE:(c+1)*TILE_SIZE]
            tile_path = os.path.join(out_dir, f"tile_{idx}.png")
            cv2.imwrite(tile_path, tile)
            tiles.append(tile)
            idx += 1

    return tiles
