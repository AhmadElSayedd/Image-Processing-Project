import os

# base dataset directory on this laptop
dataset_dir = r"C:\Github Repositories\Image-Processing-Project\Gravity Falls"

# output paths for pipeline artifacts
tiles_output_dir    = os.path.join(dataset_dir, "tiles_generated")
assembly_output_dir = os.path.join(dataset_dir, "assembled_results")
visual_output_dir   = os.path.join(dataset_dir, "visual_reports")

# correct target images
correct_dir = os.path.join(dataset_dir, "correct")

# puzzle folders to iterate on
puzzle_dirs = ["puzzle_2x2", "puzzle_4x4", "puzzle_8x8"]

# file extensions used
puzzle_ext = ".jpg"
tile_ext   = ".png"

# tile naming format (format A)
tile_name_pattern = "tile_{row}_{col}"
