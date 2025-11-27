import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from detector import detect_grid_size_robust
# ==========================================
# 2. MAIN CLASS: PUZZLE PREPROCESSOR
# ==========================================
class PuzzlePreprocessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_bgr = cv2.imread(image_path)
        if self.original_bgr is None:
            raise ValueError(f"Could not load image at {image_path}")
        self.original_rgb = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2GRAY)
        
        # State variables
        self.denoised_image = None
        self.enhanced_image = None
        self.magnitude_spectrum = None
        self.pieces = []

    def apply_morphology_denoise(self, kernel_size=3):
        # MORPH_OPEN (remove salt), MORPH_CLOSE (remove pepper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        opening = cv2.morphologyEx(self.gray, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        self.denoised_image = closing
        return self.denoised_image

    def apply_fourier_filter(self, filter_type='high_pass', radius=20):
        img_to_process = self.denoised_image if self.denoised_image is not None else self.gray
        
        # DFT
        dft = np.fft.fft2(img_to_process)
        dft_shift = np.fft.fftshift(dft)
        
        # Save Magnitude Spectrum for Report (Normalized 0-255)
        mag = 20 * np.log(np.abs(dft_shift) + 1)
        self.magnitude_spectrum = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        rows, cols = img_to_process.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        
        if filter_type == 'high_pass':
            mask = np.ones((rows, cols), np.uint8)
            mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0
            # Keep DC
            mask[crow-2:crow+2, ccol-2:ccol+2] = 1
        elif filter_type == 'low_pass':
            mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1

        fshift = dft_shift * mask
        img_back = np.fft.ifft2(np.fft.ifftshift(fshift))
        img_back = np.abs(img_back)
        
        self.enhanced_image = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return self.enhanced_image, self.magnitude_spectrum

    def draw_grid_lines(self, grid_size):
        """
        Draws green grid lines over the original image to show cuts.
        Returns the annotated image.
        """
        overlay = self.original_rgb.copy()
        h, w, _ = overlay.shape
        step_h = h // grid_size
        step_w = w // grid_size

        # Draw Vertical Lines
        for x in range(1, grid_size):
            cv2.line(overlay, (x * step_w, 0), (x * step_w, h), (0, 255, 0), 2)
            
        # Draw Horizontal Lines
        for y in range(1, grid_size):
            cv2.line(overlay, (0, y * step_h), (w, y * step_h), (0, 255, 0), 2)
            
        return overlay

    def slice_puzzle_grid(self, grid_size):
        img_h, img_w, _ = self.original_rgb.shape
        piece_h = img_h // grid_size
        piece_w = img_w // grid_size
        
        self.pieces = []
        for y in range(grid_size):
            for x in range(grid_size):
                start_y, end_y = y * piece_h, (y + 1) * piece_h
                start_x, end_x = x * piece_w, (x + 1) * piece_w
                piece = self.original_rgb[start_y:end_y, start_x:end_x]
                self.pieces.append(piece)
        return self.pieces

    def save_report_artifacts(self, output_dir):
        """
        Saves all intermediate stages for your report.
        """
        artifacts_dir = os.path.join(output_dir, "report_steps")
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
            
        # 1. Grayscale
        cv2.imwrite(f"{artifacts_dir}/1_grayscale.jpg", self.gray)
        
        # 2. Denoised (Morphology)
        if self.denoised_image is not None:
            cv2.imwrite(f"{artifacts_dir}/2_denoised.jpg", self.denoised_image)
            
        # 3. Frequency Spectrum
        if self.magnitude_spectrum is not None:
            cv2.imwrite(f"{artifacts_dir}/3_fourier_spectrum.jpg", self.magnitude_spectrum)
            
        # 4. Enhanced (High Pass)
        if self.enhanced_image is not None:
            cv2.imwrite(f"{artifacts_dir}/4_enhanced_edges.jpg", self.enhanced_image)
            
        print(f"✔ Saved report images to: {artifacts_dir}")

    def visualize_pipeline(self, magnitude_spectrum):
        plt.figure(figsize=(12, 8))
        
        titles = ['1. Original', '2. Denoised', '3. Fourier Spectrum', '4. Enhanced (HPF)']
        images = [self.original_rgb, self.denoised_image, magnitude_spectrum, self.enhanced_image]
        cmaps = [None, 'gray', 'gray', 'gray']
        
        for i in range(4):
            plt.subplot(2, 3, i+1)
            plt.imshow(images[i], cmap=cmaps[i])
            plt.title(titles[i])
            plt.axis('off')
            
        # Preview Slices
        plt.subplot(2, 3, 5)
        if self.pieces:
            h, w, c = self.pieces[0].shape
            preview = np.zeros((h*2, w*2, c), dtype=np.uint8)
            preview[0:h, 0:w] = self.pieces[0]
            if len(self.pieces) > 1: preview[0:h, w:w*2] = self.pieces[1]
            if len(self.pieces) > 2: preview[h:h*2, 0:w] = self.pieces[2]
            if len(self.pieces) > 3: preview[h:h*2, w:w*2] = self.pieces[3]
            plt.imshow(preview)
            plt.title(f'5. Slices (Grid {len(self.pieces)**0.5:.0f}x{len(self.pieces)**0.5:.0f})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# ==========================================
# 3. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # --- USER SETTINGS ---
    # Just change the filename, the grid size will be auto-detected!
    IMAGE_FILE = "Gravity Falls/puzzle_4x4/56.jpg" 
    
    if os.path.exists(IMAGE_FILE):
        print(f"=== Processing: {IMAGE_FILE} ===")
        
        # 1. AUTO-DETECT GRID SIZE
        print(">>> detecting grid size...")
        detected_N = detect_grid_size_robust(IMAGE_FILE)
        print(f"✔ Detected Grid Size: {detected_N}x{detected_N}")
        
        # 2. INITIALIZE PIPELINE
        processor = PuzzlePreprocessor(IMAGE_FILE)
        
        # 3. APPLY STEPS
        print(">>> Applying Morphology (Denoising)...")
        processor.apply_morphology_denoise(kernel_size=3)
        
        print(">>> Applying Fourier Transform (Enhancement)...")
        enhanced, spectrum = processor.apply_fourier_filter(filter_type='high_pass')
        
        print(f">>> Slicing image into {detected_N}x{detected_N}...")
        pieces = processor.slice_puzzle_grid(detected_N)
        
        # 4. SAVE ARTIFACTS
        output_dir = "processed_slices"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # --- NEW: Save Report Images (Steps 1-4) ---
        processor.save_report_artifacts(output_dir)

        # --- NEW: Save Grid Overlay Image ---
        grid_overlay = processor.draw_grid_lines(detected_N)
        cv2.imwrite(f"{output_dir}/grid_overlay.jpg", cv2.cvtColor(grid_overlay, cv2.COLOR_RGB2BGR))
        print(f"✔ Saved Grid Overlay visualization.")

        # Save individual pieces
        # Clear old files in that folder first (optional)
        files = os.listdir(output_dir)
        for f in files:
            if f.startswith("piece_"): os.remove(os.path.join(output_dir, f))

        for i, p in enumerate(pieces):
            p_bgr = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{output_dir}/piece_{i}.jpg", p_bgr)
            
        print(f"✔ Saved {len(pieces)} pieces to '{output_dir}/'")
        
        # 5. VISUALIZE
        processor.visualize_pipeline(spectrum)
        
    else:
        print(f"❌ Error: File not found at {IMAGE_FILE}")