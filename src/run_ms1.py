# run_ms1.py
import os
from config import *
from preprocess import run_preprocessing
from segment_tiles import run_segmentation
from features import run_feature_extraction


def run_ms1():
    print("\n==========================")
    print(" PHASE 1: FULL PIPELINE")
    print("==========================")

    print("\n[1/3] Preprocessing images...")
    run_preprocessing()

    print("\n[2/3] Segmenting tiles...")
    run_segmentation()

    print("\n[3/3] Extracting edge features...")
    run_feature_extraction()

    print("\n🎯 Milestone 1 completed successfully!")
    print(f"Artifacts saved under:\n - {TILES_2_DIR}\n - {TILES_4_DIR}\n - {TILES_8_DIR}")
    print("\nDescriptors and visualizations saved under:", VISUALS_DIR)


if __name__ == "__main__":
    run_ms1()
