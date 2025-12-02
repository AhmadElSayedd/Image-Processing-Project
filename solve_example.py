# solve_example.py
import cv2
import numpy as np

from solver import solve_puzzle
from tiles import TOP, RIGHT, BOTTOM, LEFT  # not strictly needed here, but kept for clarity


def build_solution_image(tiles, perm, N):
    """
    Assemble the solved board into one big BGR image.
    tiles: list[Tile]
    perm: list of tile ids in row-major order
    """
    id_to_tile = {t.id: t for t in tiles}
    h, w, _ = tiles[0].img.shape

    board = np.zeros((N * h, N * w, 3), dtype=np.uint8)

    for r in range(N):
        for c in range(N):
            tid = perm[r * N + c]
            tile_img = id_to_tile[tid].img
            y1, y2 = r * h, (r + 1) * h
            x1, x2 = c * w, (c + 1) * w
            board[y1:y2, x1:x2, :] = tile_img

    return board


if __name__ == "__main__":
    # >>>> SET YOUR PATH & N HERE <<<<
    tiles_folder = r"C:\Github Repositories\Image-Processing-Project\Gravity Falls\tiles\4x4\4x4_40"
    N = 4

    tiles, perm = solve_puzzle(tiles_folder, N)
    solved_img = build_solution_image(tiles, perm, N)

    cv2.imshow("Solved Puzzle", solved_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
