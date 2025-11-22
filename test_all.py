import os
import cv2
from detector import detect_grid_size_unknown

BASE_DIR = "Gravity Falls"

TEST_FOLDERS = {
    "puzzle_2x2": 2,
    "puzzle_4x4": 4,
    "puzzle_8x8": 8
}

def test_folder(folder_name, expected_size):
    print(f"\n=== Testing folder: {folder_name} (expected {expected_size}) ===")

    folder_path = os.path.join(BASE_DIR, folder_name)
    if not os.path.exists(folder_path):
        print("❌ Folder not found:", folder_path)
        return 0, 0

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) == 0:
        print("⚠ No images found in:", folder_path)
        return 0, 0

    correct = 0
    total = len(images)

    for file in images:
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[ERROR] Could not load {file}")
            continue

        detected = detect_grid_size_unknown(img, debug=False)
        ok = detected == expected_size

        print(f"{file} -> detected {detected}  "
              f"{'✔ CORRECT' if ok else '❌ WRONG'}")

        if ok:
            correct += 1

    print(f"\nAccuracy for {folder_name}: "
          f"{correct}/{total} = {100*correct/total:.2f}%")

    return correct, total


def run_all_tests():
    print("\n===========================")
    print(" RUNNING GRID SIZE TESTS")
    print("===========================\n")

    grand_correct = 0
    grand_total = 0

    for folder, expected in TEST_FOLDERS.items():
        c, t = test_folder(folder, expected)
        grand_correct += c
        grand_total += t

    print("\n===========================")
    print("     FINAL SUMMARY")
    print("===========================\n")
    print(f"TOTAL: {grand_correct}/{grand_total} "
          f"= {100*grand_correct/grand_total:.2f}%\n")


if __name__ == "__main__":
    run_all_tests()
