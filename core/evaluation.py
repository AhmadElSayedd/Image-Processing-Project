import os
import glob
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .config import (
    dataset_dir,
    assembly_output_dir,
    correct_dir,
    puzzle_dirs,
    puzzle_ext,
)


def _extract_id(name: str) -> int:
    digits = re.findall(r"\d+", name)
    return int(digits[-1]) if digits else -1


def evaluate_accuracy():
    print("\n[EVAL] Evaluating reconstruction accuracy...")

    stats = {}

    for puzzle_folder in puzzle_dirs:
        puzzle_path = os.path.join(dataset_dir, puzzle_folder)
        if not os.path.isdir(puzzle_path):
            continue

        # expected grid string, e.g., "2x2"
        grid_str = puzzle_folder.split("_")[1]
        solved_dir = os.path.join(assembly_output_dir, f"detected_{grid_str}")

        # count total input puzzles of that type
        total_puzzles = len(glob.glob(os.path.join(puzzle_path, f"*{puzzle_ext}")))
        correct_count = 0

        if os.path.isdir(solved_dir):
            # iterate over all ground-truth images
            for gt_path in glob.glob(os.path.join(correct_dir, "*.png")):
                gt_name = os.path.basename(gt_path)
                gt_id = _extract_id(gt_name)

                # find corresponding solved image for this id
                match_path = None
                for solved_path in glob.glob(os.path.join(solved_dir, "*.png")):
                    solved_name = os.path.basename(solved_path)
                    if _extract_id(solved_name) == gt_id:
                        match_path = solved_path
                        break

                if match_path is None:
                    continue

                gt_img = cv2.imread(gt_path)
                solved_img = cv2.imread(match_path)

                if gt_img is None or solved_img is None:
                    continue

                # resize ground truth to solved size for fair comparison
                if gt_img.shape != solved_img.shape:
                    gt_img = cv2.resize(gt_img, (solved_img.shape[1], solved_img.shape[0]))

                # mean squared error
                diff = gt_img.astype("float32") - solved_img.astype("float32")
                mse = float(np.mean(diff * diff))

                # threshold from original code
                if mse < 2500.0:
                    correct_count += 1

        acc = (correct_count / total_puzzles * 100.0) if total_puzzles > 0 else 0.0
        stats[puzzle_folder] = acc
        print(f"  {puzzle_folder}: {correct_count}/{total_puzzles} ({acc:.2f}%)")

    # bar plot of accuracies
    if stats:
        labels = list(stats.keys())
        values = [stats[k] for k in labels]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values)
        plt.ylim(0, 105)
        plt.ylabel("Accuracy (%)")
        plt.title("Final Project Accuracy")

        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%",
                     ha="center", va="bottom")

        fig_path = os.path.join(dataset_dir, "final_score.png")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

        print(f"\n[EVAL] Accuracy plot saved to: {fig_path}")
