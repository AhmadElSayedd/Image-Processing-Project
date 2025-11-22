import os
import cv2
from script import detect_grid_size

print(">>> test_puzzles.py started")

BASE_DIR = "Gravity Falls"

TEST_FOLDERS = {
    "puzzle_2x2": 2,
    "puzzle_4x4": 4,
    "puzzle_8x8": 8
}

def test_folder(folder_name, expected_size):
    print(f"\n--- Checking folder: {folder_name} ---")

    folder_path = os.path.join(BASE_DIR, folder_name)
    print("Full path:", folder_path)

    if not os.path.exists(folder_path):
        print("❌ ERROR: Folder not found!")
        return 0, 0  # no correct, no total

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(images)} images.")

    total = len(images)
    correct = 0

    for file in images:
        print(f"Processing {file}...")
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[ERROR] Cannot load: {file}")
            continue

        detected = detect_grid_size(img, debug=False)
        print(f"Result: {file} -> {detected}")

        if detected == expected_size:
            correct += 1

    # Print accuracy
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n✔ Accuracy for {folder_name}: {accuracy:.2f}% "
              f"({correct}/{total} correct)")
    else:
        print("\n⚠ No images to test.")

    return correct, total


def run_all_tests():
    print(">>> Running all tests...")

    for folder, expected in TEST_FOLDERS.items():
        test_folder(folder, expected)

    print("\n>>> Finished all tests.")


if __name__ == "__main__":
    run_all_tests()
