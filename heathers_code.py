import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm

# ===== CONFIG =====
test_csv = './dataset_v2/test.csv'
image_root = './dataset_v2/root/test'
output_root = './blurred_test_outputs'

kernel_sizes = [0, 11, 21, 31, 41]   # 0 = original
level_names = ['Original', 'Blur1', 'Blur2', 'Blur3', 'Blur4']

# ===== FUNCTIONS =====
def crop_image(image, coords):
    """Crop image based on 8 coordinates, clipping to image bounds."""
    h, w = image.shape[:2]
    x_coords = coords[::2]
    y_coords = coords[1::2]
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

    # Clip to image dimensions
    x_min, x_max = max(0, x_min), min(w, x_max)
    y_min, y_max = max(0, y_min), min(h, y_max)

    # Ensure non-empty crop
    if x_max <= x_min or y_max <= y_min:
        return None

    return image[y_min:y_max, x_min:x_max]

# ===== MAIN FUNCTION =====
def main():
    # Load CSV
    df = pd.read_csv(test_csv)
    df.columns = df.columns.str.strip()  # remove any whitespace

    # Coordinate columns
    coord_cols = ['p1_x', 'p_1y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y']

    # Check columns exist
    for col in coord_cols:
        if col not in df.columns:
            raise ValueError(f"git aCSV missing column: {col}")

    # Create output folders
    os.makedirs(output_root, exist_ok=True)
    for name in level_names:
        os.makedirs(os.path.join(output_root, name), exist_ok=True)

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing test images"):
        image_id = row['image_id']
        image_path = os.path.join(image_root, f"{int(image_id)}.jpg")

        # Read image
        image = cv2.imread(image_path)

        # Crop
        coords = row[coord_cols].values.astype(float)
        crop = crop_image(image, coords)

        # Apply blur levels and save
        for ksize, name in zip(kernel_sizes, level_names):
            if ksize == 0:
                blurred = crop
            else:
                blurred = cv2.GaussianBlur(crop, (ksize, ksize), 0)

            out_path = os.path.join(output_root, name, f"{int(image_id)}.jpg")
            cv2.imwrite(out_path, blurred)

    print("\n✅ All images processed! Cropped + blurred images saved in:", output_root)

# ===== RUN SCRIPT =====
if __name__ == "__main__":
    main()
