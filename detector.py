import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        valid_peaks = 0
        total_checks = 0
        
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

    if debug:
        print(f"--- Debug: {image_path} ---")
        print(f"8x8 Check: Vertical Pass={pass_v_8:.2f}, Horizontal Pass={pass_h_8:.2f}")
        print(f"4x4 Check: Vertical Pass={pass_v_4:.2f}, Horizontal Pass={pass_h_4:.2f}")
        print(f"Result: {detected_size}x{detected_size}")
        
        # Visual Debug
        plt.figure(figsize=(12, 4))
        plt.subplot(1,2,1)
        plt.plot(v_norm)
        plt.title(f"Vertical Profile (Norm) - Detected {detected_size}")
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        for r in [1/8, 3/8, 5/8, 7/8]: plt.axvline(x=w*r, color='g', alpha=0.3)
        for r in [1/4, 3/4]: plt.axvline(x=w*r, color='r', alpha=0.5)
        
        plt.subplot(1,2,2)
        plt.imshow(abs_grad_x, cmap='gray')
        plt.title("Vertical Edge Strength (Sobel)")
        plt.axis('off')
        plt.show()

    return detected_size