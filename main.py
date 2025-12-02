import os
import cv2

DATASET = r"C:\Github Repositories\Image-Processing-Project\Gravity Falls"
OUTPUT_ROOT = os.path.join(DATASET, "tiles")

# Map folder -> required grid size
PUZZLE_FOLDERS = {
    "puzzle_2x2": 2,
    "puzzle_4x4": 4,
    "puzzle_8x8": 8
}

def extract_and_save_tiles(img_path, base_name, N):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Failed to read: {img_path}")
        return

    H, W, _ = img.shape
    tile_h = H // N
    tile_w = W // N

    save_root = os.path.join(OUTPUT_ROOT, f"{N}x{N}", f"{N}x{N}_{base_name}")
    os.makedirs(save_root, exist_ok=True)

    print(f"[INFO] Processing {base_name} ({N}x{N}) → Folder: {save_root}")

    t = 0
    for r in range(N):
        for c in range(N):
            y1, y2 = r*tile_h, (r+1)*tile_h
            x1, x2 = c*tile_w, (c+1)*tile_w
            tile = img[y1:y2, x1:x2]

            cv2.imwrite(os.path.join(save_root, f"{t}.png"), tile)
            t += 1


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Create subfolders 2x2, 4x4, 8x8
    for folder, N in PUZZLE_FOLDERS.items():
        os.makedirs(os.path.join(OUTPUT_ROOT, f"{N}x{N}"), exist_ok=True)

    for folder, N in PUZZLE_FOLDERS.items():
        puzzle_path = os.path.join(DATASET, folder)
        images = [f for f in os.listdir(puzzle_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]

        print(f"\n--- Found {len(images)} images in {folder} ---")

        for idx, img_name in enumerate(images):
            base_name = os.path.splitext(img_name)[0]
            extract_and_save_tiles(os.path.join(puzzle_path, img_name), base_name, N)

    print("\n🎯 DONE! Tiles generated successfully ✔")


if __name__ == "__main__":
    main()
