import cv2
import numpy as np
import os
import glob
from collections import defaultdict

def detect_grid_size_robust(image_path, debug=False):
    """
    Detects if a puzzle is 2x2, 4x4, or 8x8 using Gradient Energy Analysis.
    Improved to detect faint lines in 8x8 grids.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- CHANGE 1: Use Sobel instead of Canny ---
    # Sobel captures "how strong is the edge" rather than just "is there an edge?"
    # This helps detect faint cut lines in 8x8 puzzles.
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # Vertical edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) # Horizontal edges
    
    # Calculate absolute gradient magnitude
    abs_grad_x = cv2.convertScaleAbs(sobel_x)
    abs_grad_y = cv2.convertScaleAbs(sobel_y)

    # --- CHANGE 2: Projections based on Gradient Energy ---
    # Sum of vertical edge strength along columns
    v_proj = np.sum(abs_grad_x, axis=0)
    # Sum of horizontal edge strength along rows
    h_proj = np.sum(abs_grad_y, axis=1)

    # Smooth the profiles slightly to remove single-pixel noise
    kernel_size = max(3, int(min(h, w) * 0.005))
    v_proj = np.convolve(v_proj, np.ones(kernel_size)/kernel_size, mode='same')
    h_proj = np.convolve(h_proj, np.ones(kernel_size)/kernel_size, mode='same')

    # Normalize by the median noise level (robust against overall image contrast)
    # We add a small epsilon to avoid division by zero
    v_baseline = np.median(v_proj) + 1e-5
    h_baseline = np.median(h_proj) + 1e-5
    
    v_norm = v_proj / v_baseline
    h_norm = h_proj / h_baseline

    def get_score_at_intervals(projection, N):
        """
        Checks the signal strength at 1/N, 2/N, ... (N-1)/N.
        Returns the percentage of expected peaks that are 'strong'.
        """
        length = len(projection)
        
        # We check every interval k/N
        # For 8x8, we specifically care about the odd intervals (1/8, 3/8, 5/8, 7/8)
        # because 2/8 (1/4) overlaps with 4x4.
        if N == 8:
            ratios = [1/8, 3/8, 5/8, 7/8]
        elif N == 4:
            ratios = [1/4, 3/4]
        else:
            ratios = [1/2]

        window = int(length * 0.02) # Look within 2% error margin

        scores = []
        for r in ratios:
            center = int(length * r)
            start = max(0, center - window)
            end = min(length, center + window)
            
            # Find the max signal in this expected region
            peak_strength = np.max(projection[start:end])
            scores.append(peak_strength)

        # A peak is "valid" if it is significantly higher than the baseline (1.0)
        # 8x8 lines are often weaker, so we use a lower threshold for them
        threshold = 1.5 if N == 8 else 2.0
        
        # Count how many peaks passed the threshold
        pass_rate = sum(1 for s in scores if s > threshold) / len(scores)
        avg_strength = np.mean(scores)
        
        return pass_rate, avg_strength

    # --- CHANGE 3: Hierarchical Scoring ---
    # 1. Check 8x8 unique lines (1/8, 3/8...)
    pass_v_8, str_v_8 = get_score_at_intervals(v_norm, 8)
    pass_h_8, str_h_8 = get_score_at_intervals(h_norm, 8)
    
    # 2. Check 4x4 unique lines (1/4, 3/4)
    pass_v_4, str_v_4 = get_score_at_intervals(v_norm, 4)
    pass_h_4, str_h_4 = get_score_at_intervals(h_norm, 4)

    detected_size = 2 # Fallback

    # Decision Logic
    # To be 8x8, we need strong evidence at the 1/8 marks.
    # We require > 50% of the 8x8 lines to be visible.
    is_8x8 = (pass_v_8 >= 0.5 and pass_h_8 >= 0.5) or (pass_v_8 + pass_h_8 >= 1.5)
    
    # To be 4x4, we need evidence at 1/4 marks, but NOT at 1/8 marks
    is_4x4 = (pass_v_4 >= 0.5 and pass_h_4 >= 0.5) or (pass_v_4 + pass_h_4 >= 1.5)

    if is_8x8:
        detected_size = 8
    elif is_4x4:
        detected_size = 4
    else:
        detected_size = 2

    return detected_size


def test_all_puzzles(base_dir="Gravity Falls", verbose=False):
    """
    Test all puzzle images in each folder and generate comprehensive statistics.
    
    Args:
        base_dir: Base directory containing puzzle folders
        verbose: If True, print details for each image
    """
    # Define test configurations
    puzzle_types = {
        "puzzle_2x2": 2,
        "puzzle_4x4": 4,
        "puzzle_8x8": 8,
    }
    
    # Results storage
    results = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})
    
    print("=" * 100)
    print(f"{'COMPREHENSIVE PUZZLE GRID DETECTION TEST':^100}")
    print("=" * 100)
    print()
    
    # Test each puzzle type
    for folder_name, expected_size in puzzle_types.items():
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"⚠ Warning: Folder '{folder_path}' not found. Skipping...")
            continue
        
        # Get all .jpg files in the folder
        image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        
        if not image_files:
            print(f"⚠ Warning: No .jpg files found in '{folder_path}'. Skipping...")
            continue
        
        print(f"\n{'='*100}")
        print(f"Testing: {folder_name} (Expected: {expected_size}x{expected_size})")
        print(f"Total Images: {len(image_files)}")
        print(f"{'='*100}")
        
        correct_count = 0
        failed_images = []
        
        for image_file in image_files:
            image_name = os.path.basename(image_file)
            
            try:
                detected_size = detect_grid_size_robust(image_file, debug=False)
                
                is_correct = (detected_size == expected_size)
                results[folder_name]["total"] += 1
                
                if is_correct:
                    correct_count += 1
                    results[folder_name]["correct"] += 1
                    if verbose:
                        print(f"  ✓ {image_name}: Detected {detected_size}x{detected_size}")
                else:
                    failed_images.append((image_name, detected_size))
                    results[folder_name]["errors"].append({
                        "file": image_name,
                        "expected": expected_size,
                        "detected": detected_size
                    })
                    print(f"  ✗ {image_name}: Detected {detected_size}x{detected_size} (Expected {expected_size}x{expected_size})")
                    
            except Exception as e:
                results[folder_name]["total"] += 1
                results[folder_name]["errors"].append({
                    "file": image_name,
                    "expected": expected_size,
                    "error": str(e)
                })
                print(f"  ✗ {image_name}: ERROR - {e}")
        
        # Print summary for this folder
        accuracy = (correct_count / len(image_files) * 100) if len(image_files) > 0 else 0
        print(f"\n{folder_name} Results: {correct_count}/{len(image_files)} correct ({accuracy:.1f}%)")
        
        if failed_images and not verbose:
            print(f"Failed Images: {len(failed_images)}")
            if len(failed_images) <= 10:
                for img_name, detected in failed_images[:10]:
                    print(f"  - {img_name}: Detected {detected}x{detected}")
            else:
                print(f"  (showing first 10 of {len(failed_images)} failures)")
                for img_name, detected in failed_images[:10]:
                    print(f"  - {img_name}: Detected {detected}x{detected}")
    
    # Print overall summary
    print("\n" + "=" * 100)
    print(f"{'OVERALL SUMMARY':^100}")
    print("=" * 100)
    
    total_correct = sum(r["correct"] for r in results.values())
    total_images = sum(r["total"] for r in results.values())
    overall_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
    
    print(f"\nTotal Images Tested: {total_images}")
    print(f"Total Correct: {total_correct}")
    print(f"Total Failed: {total_images - total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    
    print("\n" + "-" * 100)
    print(f"{'Folder':<20} {'Total':<10} {'Correct':<10} {'Failed':<10} {'Accuracy':<10}")
    print("-" * 100)
    
    for folder_name, expected_size in puzzle_types.items():
        if folder_name in results:
            r = results[folder_name]
            total = r["total"]
            correct = r["correct"]
            failed = total - correct
            acc = (correct / total * 100) if total > 0 else 0
            print(f"{folder_name:<20} {total:<10} {correct:<10} {failed:<10} {acc:.1f}%")
    
    print("=" * 100)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test grid detection on all puzzle images")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Print details for each image (including successful ones)")
    parser.add_argument("--base-dir", "-d", default="Gravity Falls",
                       help="Base directory containing puzzle folders (default: 'Gravity Falls')")
    
    args = parser.parse_args()
    
    results = test_all_puzzles(base_dir=args.base_dir, verbose=args.verbose)
