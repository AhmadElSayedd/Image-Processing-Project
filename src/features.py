import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from config import (
    TILES_ENH_2_DIR, TILES_ENH_4_DIR, TILES_ENH_8_DIR,
    VISUALS_TILES_EDGES_DIR, ensure_dirs
)

def extract_edges(tile, sw=4):
    h,w = tile.shape[:2]
    sw = min(sw, h//4, w//4)
    return {
        "top": tile[0:sw,:,:],
        "bottom": tile[h-sw:h,:,:],
        "left": tile[:,0:sw,:],
        "right": tile[:,w-sw:w,:]
    }

def visualize(tile_path, out_path, sw=4):
    import matplotlib.pyplot as plt
    tile = cv2.imread(tile_path)
    if tile is None: return
    edges = extract_edges(tile, sw)
    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

    names=["top","right","bottom","left"]
    pos=[4,3,5,1]

    plt.figure(figsize=(8,4))
    plt.subplot(2,3,2); plt.imshow(tile_rgb); plt.axis("off")
    plt.title(os.path.basename(tile_path))

    for n,p in zip(names,pos):
        strip = cv2.cvtColor(edges[n], cv2.COLOR_BGR2RGB)
        plt.subplot(2,3,p); plt.imshow(strip); plt.axis("off"); plt.title(n)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def run_edge_visuals():
    ensure_dirs()
    folders=[
        (TILES_ENH_2_DIR,"2x2"),
        (TILES_ENH_4_DIR,"4x4"),
        (TILES_ENH_8_DIR,"8x8"),
    ]
    for f,label in folders:
        tiles = sorted(glob(os.path.join(f,"*.png")))
        if not tiles: continue
        save_dir = os.path.join(VISUALS_TILES_EDGES_DIR, label)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n[Edges] {label}")
        for t in tqdm(tiles[:6]):
            out = os.path.join(save_dir,
                                os.path.basename(t).replace(".png","_edges.png"))
            visualize(t, out)

if __name__=="__main__":
    run_edge_visuals()
