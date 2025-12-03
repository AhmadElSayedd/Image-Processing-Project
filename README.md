# Image Processing Project - Milestone 1

## 📌 Overview

This project implements an automated puzzle image processing pipeline for the **Milestone 1 (Phase 1)** deliverable. The system preprocesses puzzle images, segments them into tiles, extracts edge features, and automatically detects grid sizes.

## 🌿 Branch Information

- **Branch Name:** `phase1`
- **Milestone:** Milestone 1 - Image Preprocessing & Tile Segmentation
- **Status:** Complete ✅

## 🚀 Quick Start

### Running the Complete Pipeline

To execute the entire Milestone 1 pipeline, simply run:

```bash
python src/run_ms1.py
```

This command will:

1. Preprocess all puzzle images (enhancement & filtering)
2. Segment images into tiles (2×2, 4×4, 8×8 grids)
3. Extract and visualize edge features for each tile

### Output Structure

After running, artifacts will be organized as follows:

```
artifacts/
├── preprocessed/           # Enhanced images
│   ├── puzzle_2x2/
│   ├── puzzle_4x4/
│   └── puzzle_8x8/
├── tiles/                  # Segmented tiles
│   ├── 2x2/
│   ├── 4x4/
│   └── 8x8/
├── tiles_enhanced/         # Enhanced tiles (for reporting)
├── tiles_contours/         # Contour-extracted tiles
└── visuals/                # Feature visualizations
    ├── tiles_before_after/
    └── tiles_edges/
```

---

## 📁 File Structure & Descriptions

### Core Pipeline Files

#### `src/run_ms1.py`

**Purpose:** Main entry point for Milestone 1 execution.

**What it does:**

- Orchestrates the complete Phase 1 pipeline
- Calls preprocessing, segmentation, and feature extraction in sequence
- Provides progress updates and completion summary
- **Usage:** `python src/run_ms1.py`

---

#### `src/config.py`

**Purpose:** Centralized configuration and path management.

**What it does:**

- Defines all project directory paths (input/output)
- Sets up folder structure for artifacts:
  - Preprocessed images
  - Tiles (regular, enhanced, contours)
  - Visualizations and reports
- Provides `ensure_dirs()` utility to create required folders
- Makes path management consistent across all modules

**Key Directories:**

- `DATA_DIR`: Input puzzle images (`Gravity Falls/`)
- `ARTIFACTS_DIR`: All processing outputs
- `PREPROC_DIR`: Enhanced images after preprocessing
- `TILES_DIR`: Segmented tiles organized by grid size
- `VISUALS_DIR`: Feature visualizations and comparisons

---

#### `src/preprocess.py`

**Purpose:** Image enhancement and preprocessing operations.

**What it does:**

- **Luminance Equalization:** Normalizes brightness using YCrCb color space
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Enhances local contrast in LAB color space
- **Sharpening:** Applies unsharp masking for edge clarity
- **Contour Extraction:** Detects edges using Canny edge detection
- Processes all puzzle images from `puzzle_2x2/`, `puzzle_4x4/`, `puzzle_8x8/` folders
- Saves enhanced versions to `artifacts/preprocessed/`

**Main Functions:**

- `preprocess(img)`: Applies full enhancement pipeline
- `extract_contours(tile)`: Extracts edge contours from tiles
- `run_preprocessing()`: Batch processes all puzzle folders

---

#### `src/segment_tiles.py`

**Purpose:** Tile segmentation from preprocessed images.

**What it does:**

- Divides each puzzle image into N×N equal-sized tiles
- Handles 2×2, 4×4, and 8×8 grid configurations
- Maintains tile position metadata (row, column indices)
- Saves individual tiles with naming convention: `{image_id}_r{row}_c{col}.png`
- Processes preprocessed images and generates organized tile outputs

**Main Functions:**

- `segment_image_into_tiles(img_path, output_dir, N)`: Segments single image
- `segment_folder(preproc_dir, tiles_dir, N)`: Batch segments entire folder
- `run_segmentation()`: Processes all grid sizes (2×2, 4×4, 8×8)

---

#### `src/features.py`

**Purpose:** Edge feature extraction and visualization.

**What it does:**

- Extracts border regions from each tile (top, bottom, left, right)
- Creates visual representations of edge features
- Generates matplotlib-based visualizations showing:
  - Central tile image
  - Four border strips positioned around it
- Saves descriptive visualizations to `artifacts/visuals/tiles_edges/`
- Useful for analyzing tile boundaries and matching potential

**Main Functions:**

- `extract_edges(tile, sw=4)`: Extracts 4-pixel wide border strips
- `visualize(tile_path, out_path)`: Creates edge visualization figure
- `run_edge_visuals()`: Batch processes all enhanced tiles

---

#### `src/auto_grid.py`

**Purpose:** Automatic grid size detection using gradient projection.

**What it does:**

- **Gradient Projection Algorithm:** Analyzes color gradient patterns to detect grid lines
- Computes LAB color space gradients in both X and Y directions
- Projects gradients to identify periodic peaks (grid boundaries)
- Automatically classifies images as 2×2, 4×4, or 8×8 puzzles
- Validates detection accuracy against ground truth
- Generates overlay visualizations showing detected grid lines
- Saves correctness reports and misclassified images

**How Grid Detection Works:**

1. Converts image to LAB color space for better gradient separation
2. Computes horizontal and vertical gradient profiles
3. Normalizes gradient intensity relative to median
4. Evaluates gradient peaks at expected grid positions
5. Scores each grid size hypothesis (2×2, 4×4, 8×8)
6. Selects grid size with highest confidence score

**Main Functions:**

- `gradient_projection_scores(img_bgr, n_cuts)`: Computes grid score for N×N hypothesis
- `detect_grid_size(img_bgr)`: Returns detected grid size (2, 4, or 8)
- `save_grid_overlay(img_bgr, ...)`: Draws detected grid on image
- `run_auto_grid_eval_and_save()`: Evaluates accuracy across dataset

**Outputs:**

- Grid overlay images (correct/incorrect folders)
- Accuracy logs: `{N}x{N}_correct.txt` and `{N}x{N}_wrong.txt`
- Saves to: `artifacts/tiles_auto/`

**Usage:**

```bash
python src/auto_grid.py
```

---

## 🎯 Pipeline Workflow

### Step 1: Preprocessing (`preprocess.py`)

- Load raw puzzle images
- Apply luminance equalization, CLAHE, and sharpening
- Save enhanced images to `artifacts/preprocessed/`

### Step 2: Tile Segmentation (`segment_tiles.py`)

- Load preprocessed images
- Segment into N×N tiles based on folder (2×2, 4×4, 8×8)
- Save individual tiles to `artifacts/tiles/{N}x{N}/`

### Step 3: Feature Extraction (`features.py`)

- Load segmented tiles
- Extract edge border strips (4 pixels wide)
- Generate visualization showing tile with its 4 borders
- Save to `artifacts/visuals/tiles_edges/`

### Optional: Auto Grid Detection (`auto_grid.py`)

- Test gradient projection algorithm on preprocessed images
- Detect grid size automatically
- Validate against expected grid sizes
- Generate accuracy reports and overlays

---

## 📊 Expected Outputs

### Preprocessed Images

- **Location:** `artifacts/preprocessed/puzzle_{2x2|4x4|8x8}/`
- **Format:** Enhanced JPG/PNG images
- **Enhancement:** Better contrast, sharper edges, normalized lighting

### Tile Segments

- **Location:** `artifacts/tiles/{2x2|4x4|8x8}/`
- **Format:** `{imageID}_r{row}_c{col}.png`
- **Example:** `56_r0_c1.png` (image 56, row 0, column 1)

### Edge Visualizations

- **Location:** `artifacts/visuals/tiles_edges/`
- **Format:** PNG figures showing tile + 4 border strips
- **Purpose:** Visual inspection of tile boundaries for matching

### Auto Grid Detection

- **Location:** `artifacts/tiles_auto/`
- **Contents:**
  - Grid overlays (correct/wrong folders)
  - Accuracy text logs
  - Misclassification reports

---

## 🛠 Requirements

Install dependencies:

```bash
pip install opencv-python numpy matplotlib tqdm
```

Or use:

```bash
pip install -r requirements.txt
```

---

## 📂 Input Data Structure

Expected input folder structure:

```
Gravity Falls/
├── correct/           # Ground truth images (optional)
├── puzzle_2x2/        # 2×2 scrambled puzzles
├── puzzle_4x4/        # 4×4 scrambled puzzles
└── puzzle_8x8/        # 8×8 scrambled puzzles
```

---

## 🔍 Key Features

✅ **Automated Enhancement:** Adaptive contrast, sharpening, luminance normalization  
✅ **Flexible Grid Support:** Handles 2×2, 4×4, and 8×8 puzzle configurations  
✅ **Edge Feature Extraction:** Border-based descriptors for tile matching  
✅ **Auto Grid Detection:** Gradient projection algorithm with validation  
✅ **Organized Outputs:** Clean artifact structure with descriptive naming  
✅ **Progress Tracking:** tqdm progress bars for batch operations

---

## 📝 Notes

- All processing is **non-destructive** (original images unchanged)
- Artifacts are organized by grid size for easy navigation
- Edge visualizations help identify good tile boundary features
- Auto grid detection provides confidence scores for validation
- Pipeline is modular—each step can be run independently

---

## 👥 Team Information

**Project:** Image Processing - Puzzle Solver  
**Milestone:** 1 (Phase 1)  
**Branch:** `phase1`  
**Repository:** Image-Processing-Project

---

## 📌 Summary

**Milestone 1** focuses on preprocessing and feature preparation for puzzle solving:

1. ✅ Image enhancement pipeline
2. ✅ Tile segmentation (2×2, 4×4, 8×8)
3. ✅ Edge feature extraction
4. ✅ Automatic grid size detection
5. ✅ Visual validation tools

**Next Steps (Milestone 2):** Tile matching, puzzle reconstruction, and accuracy evaluation.

---

**Run the complete pipeline with:**

```bash
python src/run_ms1.py
```
