import cv2
import numpy as np
import os
import glob
import sys
import time

# --- IMPORT LOGIC ---
# We try importing 'main_slicer' first as per your code, 
# but fallback to 'milestone1_pipeline' just in case.

from main_slicer import PuzzlePreprocessor, detect_grid_size_robust
print("✔ Imported logic from 'main_slicer.py'")


def run_batch_processing(base_input_dir="Gravity Falls", base_output_dir="Milestone1_Results"):
    """
    Runs the full pipeline on all images with:
    - Clean console output (only errors/progress bar)
    - Accuracy statistics (detection vs expected)
    - Organized file saving
    """
    
    # Map folder names to expected grid sizes
    folders_map = {
        "puzzle_2x2": 2,
        "puzzle_4x4": 4,
        "puzzle_8x8": 8
    }
    
    # Stats storage
    stats = {
        "puzzle_2x2": {"total": 0, "correct_detect": 0, "processed": 0, "errors": 0},
        "puzzle_4x4": {"total": 0, "correct_detect": 0, "processed": 0, "errors": 0},
        "puzzle_8x8": {"total": 0, "correct_detect": 0, "processed": 0, "errors": 0}
    }
    
    print(f"{'='*80}")
    print(f"{'BATCH PIPELINE PROCESSING':^80}")
    print(f"{'='*80}")
    print(f"Input:  {base_input_dir}")
    print(f"Output: {base_output_dir}")
    print("-" * 80)

    total_images_all = 0

    for folder, expected_size in folders_map.items():
        input_path = os.path.join(base_input_dir, folder)
        
        if not os.path.exists(input_path):
            print(f"⚠ Skipping '{folder}' (Folder not found)")
            continue

        image_files = sorted(glob.glob(os.path.join(input_path, "*.jpg")))
        total_files = len(image_files)
        total_images_all += total_files
        
        stats[folder]["total"] = total_files
        
        if total_files == 0:
            print(f"⚠ Skipping '{folder}' (No images found)")
            continue

        print(f"\n📂 Processing {folder} ({total_files} images)...")
        
        # Progress Bar setup
        bar_width = 40
        
        for i, img_path in enumerate(image_files):
            filename = os.path.basename(img_path)
            
            # Progress Bar UI
            progress = (i + 1) / total_files
            filled = int(bar_width * progress)
            bar = "█" * filled + "-" * (bar_width - filled)
            sys.stdout.write(f"\r   [{bar}] {i+1}/{total_files}")
            sys.stdout.flush()

            try:
                # 1. Setup Output Folder
                image_name_no_ext = os.path.splitext(filename)[0]
                output_folder = os.path.join(base_output_dir, folder, image_name_no_ext)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # 2. Auto-Detect Grid Size
                detected_N = detect_grid_size_robust(img_path)
                
                # Update Accuracy Stats
                if detected_N == expected_size:
                    stats[folder]["correct_detect"] += 1
                else:
                    # Log detection error silently or to a log file if needed
                    pass 

                # 3. Run Pipeline
                processor = PuzzlePreprocessor(img_path)
                processor.apply_morphology_denoise(kernel_size=3)
                processor.apply_fourier_filter(filter_type='high_pass')
                
                # 4. Slicing & Saving
                pieces = processor.slice_puzzle_grid(detected_N)
                
                # Save Report Artifacts
                processor.save_report_artifacts(output_folder)
                
                # Save Grid Overlay
                grid_overlay = processor.draw_grid_lines(detected_N)
                cv2.imwrite(f"{output_folder}/grid_visualization.jpg", cv2.cvtColor(grid_overlay, cv2.COLOR_RGB2BGR))
                
                # Save Pieces
                pieces_dir = os.path.join(output_folder, "pieces")
                if not os.path.exists(pieces_dir):
                    os.makedirs(pieces_dir)
                    
                for j, p in enumerate(pieces):
                    p_bgr = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{pieces_dir}/piece_{j}.jpg", p_bgr)

                stats[folder]["processed"] += 1

            except Exception as e:
                # Clear progress line to show error clearly
                sys.stdout.write(f"\r{' '*60}\r")
                print(f"   ❌ Error processing {filename}: {e}")
                stats[folder]["errors"] += 1

        print(" ✔ Done")

    # ==========================================
    # FINAL SUMMARY REPORT
    # ==========================================
    print(f"\n{'='*80}")
    print(f"{'PROCESSING SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Category':<15} | {'Total':<8} | {'Processed':<10} | {'Detect Acc':<12} | {'Errors':<8}")
    print("-" * 80)

    grand_total = 0
    grand_processed = 0
    grand_correct = 0

    for folder, data in stats.items():
        if data["total"] == 0: continue
        
        processed = data["processed"]
        acc = (data["correct_detect"] / data["total"]) * 100
        errors = data["errors"]
        
        print(f"{folder:<15} | {data['total']:<8} | {processed:<10} | {acc:.1f}%{'':<7} | {errors:<8}")
        
        grand_total += data["total"]
        grand_processed += processed
        grand_correct += data["correct_detect"]

    print("-" * 80)
    
    if grand_total > 0:
        overall_acc = (grand_correct / grand_total) * 100
        print(f"{'OVERALL':<15} | {grand_total:<8} | {grand_processed:<10} | {overall_acc:.1f}%{'':<7} | {total_images_all - grand_processed:<8}")
    
    print(f"{'='*80}")
    print(f"Artifacts saved in: {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    # Ensure this matches your folder name (Gravity Falls vs Gravity Falls/correct)
    run_batch_processing(base_input_dir="Gravity Falls")