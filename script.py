import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# def detect_peaks_robust(profile):
#     # Smooth profile to remove noise
#     smooth = cv2.GaussianBlur(profile.reshape(1,-1), (1,51), 0).ravel()

#     # First derivative (edge intensity)
#     derivative = np.gradient(smooth)

#     # Normalize
#     derivative = (derivative - derivative.min()) / (derivative.max() - derivative.min())

#     # Peak detection with strong conditions
#     peaks, _ = find_peaks(
#         derivative,
#         distance=len(profile)//4,
#         prominence=0.2*np.max(derivative),
#         width=10
#     )
    
#     return peaks, smooth, derivative


import cv2
import numpy as np

def detect_grid_size(img, debug=False):
    """
    Detect puzzle grid size (2,4,8) by measuring Sobel edge energy
    at the positions where tile boundaries *would* be.

    Strongest energy = correct grid size.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # Sobel (stronger is better for boundary testing)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    sobel = np.abs(sobel)

    # Candidate grid sizes
    candidates = [2, 4, 8]

    scores = {}  # store boundary energy per candidate

    for N in candidates:
        tile_h = H / N
        tile_w = W / N

        vertical_energy = 0
        horizontal_energy = 0

        # --- Vertical boundaries (constant x)
        for i in range(1, N):
            x = int(tile_w * i)
            # Sum sobel energy on the entire vertical line
            vertical_energy += np.sum(sobel[:, x-1:x+1])

        # --- Horizontal boundaries (constant y)
        for i in range(1, N):
            y = int(tile_h * i)
            # Sum sobel energy on the entire horizontal line
            horizontal_energy += np.sum(sobel[y-1:y+1, :])

        # Total score
        scores[N] = vertical_energy + horizontal_energy

    # Pick the N with the HIGHEST boundary energy
    best_grid = max(scores, key=scores.get)

    if debug:
        print("Boundary scores:", scores)
        print("Chosen grid size:", best_grid)

    return best_grid




def extract_tiles(img, grid_size, out_dir="tiles", draw_grid=True, debug=False):
	"""
	Extracts tiles from the puzzle image and saves them to disk.
	Optionally draws a grid overlay for debugging.
	Args:
		img: Input BGR image (numpy array)
		grid_size: int (2, 4, or 8)
		out_dir: Directory to save tiles
		draw_grid: If True, saves a grid overlay image
		debug: If True, shows the grid overlay
	Returns:
		tile_paths: List of saved tile file paths
		tile_height: int
		tile_width: int
	"""
	H, W = img.shape[:2]
	tile_height = H // grid_size
	tile_width = W // grid_size
	os.makedirs(out_dir, exist_ok=True)
	tile_paths = []
	for row in range(grid_size):
		for col in range(grid_size):
			y0 = row * tile_height
			x0 = col * tile_width
			tile = img[y0:y0+tile_height, x0:x0+tile_width]
			tile_path = os.path.join(out_dir, f"tile_{row}_{col}.png")
			cv2.imwrite(tile_path, tile)
			tile_paths.append(tile_path)

	# Draw grid overlay
	if draw_grid:
		overlay = img.copy()
		color = (0, 255, 0)
		thickness = max(1, min(tile_height, tile_width)//30)
		for i in range(1, grid_size):
			# Horizontal lines
			y = i * tile_height
			cv2.line(overlay, (0, y), (W, y), color, thickness)
			# Vertical lines
			x = i * tile_width
			cv2.line(overlay, (x, 0), (x, H), color, thickness)
		grid_path = os.path.join(out_dir, "grid_overlay.png")
		cv2.imwrite(grid_path, overlay)
		if debug:
			plt.figure(figsize=(6,6))
			plt.title("Grid Overlay")
			plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
			plt.axis('off')
			plt.show()

	return tile_paths, tile_height, tile_width

def main():
	import argparse
	parser = argparse.ArgumentParser(description="Automatic Jigsaw Puzzle Grid Detection and Tile Extraction")
	parser.add_argument("--input", type=str, required=True, help="Path to input puzzle image")
	parser.add_argument("--out_dir", type=str, default="tiles", help="Directory to save extracted tiles")
	parser.add_argument("--debug", action="store_true", help="Show debug visualizations")
	args = parser.parse_args()

	img = cv2.imread(args.input)
	if img is None:
		print(f"Error: Could not load image {args.input}")
		return

	grid_size = detect_grid_size(img, debug=args.debug)
	print(f"Detected grid size: {grid_size}x{grid_size}")

	tile_paths, tile_height, tile_width = extract_tiles(img, grid_size, out_dir=args.out_dir, draw_grid=True, debug=args.debug)
	print(f"Extracted {len(tile_paths)} tiles.")
	print(f"Tile size: {tile_height}x{tile_width} pixels.")
	print(f"Tiles saved to: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
	main()
