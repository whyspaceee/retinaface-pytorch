import cv2
import numpy as np
import random
import os
import utils
from typing import Tuple

# --- Import functions from your saved file ---
# Make sure augmentation_utils.py is in the same directory
# or adjust the import path accordingly.
# Also ensure matrix_iof is available either via utils.box_utils
# or added directly into augmentation_utils.py
try:
    from utils.transform import (
        _crop,
        distort_image,
        horizontal_flip,
        pad_image_to_square,
        resize_subtract_mean,

    )
except ImportError as e:
    print(f"Error importing from augmentation_utils.py: {e}")
    print("Please ensure the file exists and matrix_iof is defined/imported within it.")
    exit()

# --- Configuration ---
INPUT_IMAGE_PATH = 'data/widerface/train/images/35--Basketball/35_Basketball_Basketball_35_68.jpg' # <<< CHANGE THIS
OUTPUT_DIR = 'augmentation_steps_output'
IMAGE_SIZE = 640  # Target size for the model
BGR_MEAN = (104, 117, 123) # Example BGR mean values

# --- Example Ground Truth (Replace with actual data if available) ---
# Format: [xmin, ymin, xmax, ymax, lmk1_x, lmk1_y, ..., lmk5_x, lmk5_y, label]
# IMPORTANT: Coordinates must be absolute pixel values for the *original* input image.
#            Label is often -1 for faces to ignore, 1 for faces to keep.
# Example for a hypothetical face in the image:
INITIAL_TARGETS = np.array([
    # Box: xmin, ymin, xmax, ymax; Landmarks: lmk1_x, lmk1_y, ..., lmk5_x, lmk5_y; Label
    [232, 475, 258, 513, 237.75, 489.661, 248.116, 486.527, 243.295, 493.518, 243.054, 504.125, 249.562, 502.196, 1],
    [282, 329, 305, 359, 288.125, 337.188, 298.625, 337.188, 290.375, 340.0,   288.688, 347.125, 299.188, 347.312, 1],
    [315, 204, 336, 239, -1.0,    -1.0,    -1.0,    -1.0,    -1.0,    -1.0,    -1.0,    -1.0,    -1.0,    -1.0,    1],
    [446, 344, 469, 373, 450.96,  352.924, 457.366, 356.036, 451.143, 357.134, 450.96,  363.174, 454.254, 364.272, 1],
    [382, 386, 402, 409, -1.0,    -1.0,    -1.0,    -1.0,    -1.0,    -1.0,    -1.0,    -1.0,    -1.0,    -1.0,    1]
], dtype=np.float32)

# --- Helper function for visualization ---
import cv2
import numpy as np
import os

# Assuming BGR_MEAN and OUTPUT_DIR are defined globally or passed appropriately
# Updated visualize_step function
def visualize_step(step_name: str, image: np.ndarray, targets: np.ndarray = None, extra_info: str = ""):
    """Saves the image and optionally draws boxes/landmarks. Handles different image formats and coordinate types."""
    vis_image = image.copy() # Start with a copy

    # --- Format Handling (CHW Float -> HWC uint8) ---
    if len(vis_image.shape) == 3 and vis_image.shape[0] == 3 and vis_image.dtype == np.float32:
        # Handle CHW float image from resize_subtract_mean
        print(f"[{step_name}] Handling CHW float: Reversing mean, transposing, ensuring contiguity.")
        mean_reshaped = np.array(BGR_MEAN, dtype=np.float32).reshape(3, 1, 1)
        vis_image += mean_reshaped
        vis_image = np.clip(vis_image, 0, 255)
        vis_image = vis_image.transpose(1, 2, 0)
        vis_image = vis_image.copy() # Ensure C-contiguity
        vis_image = vis_image.astype(np.uint8)
    elif vis_image.dtype == np.float32:
        # Handle other float images (assume HWC)
        print(f"[{step_name}] Handling HWC float: Reversing mean.")
        vis_image += np.array(BGR_MEAN, dtype=np.float32)
        vis_image = np.clip(vis_image, 0, 255)
        vis_image = vis_image.astype(np.uint8)
    elif vis_image.dtype != np.uint8:
        # Convert any other non-uint8 types
        vis_image = vis_image.astype(np.uint8)

    # --- Drawing ---
    # Ensure image is drawable (HWC, uint8)
    if not (len(vis_image.shape) == 3 and vis_image.shape[2] == 3 and vis_image.dtype == np.uint8):
        print(f"[{step_name}] Warning: Image shape {vis_image.shape} or dtype {vis_image.dtype} not suitable for drawing. Skipping drawing.")
    elif targets is not None and targets.shape[0] > 0:
        height, width, _ = vis_image.shape # Use dimensions of the image being drawn on

        draw_boxes = targets[:, :4].copy()
        draw_landmarks = targets[:, 4:-1].copy()

        # Check if coordinates appear normalized (0-1 range)
        is_normalized = (draw_boxes.size > 0 and np.all(draw_boxes >= 0.0) and np.all(draw_boxes <= 1.0) and
                         draw_landmarks.size > 0 and np.all(draw_landmarks >= -1.0) and np.all(draw_landmarks <= 1.0)) # Allow -1 for landmarks in check

        if is_normalized:
            print(f"[{step_name}] Denormalizing coordinates (detected 0-1 range) using image W={width}, H={height}.")
            # Denormalize non-negative landmarks only
            lmk_mask = draw_landmarks >= 0
            draw_boxes[:, 0::2] *= width   # xmin, xmax
            draw_boxes[:, 1::2] *= height  # ymin, ymax
            draw_landmarks[:, 0::2][lmk_mask[:, 0::2]] *= width  # lmk_x
            draw_landmarks[:, 1::2][lmk_mask[:, 1::2]] *= height # lmk_y
        # else: Assume coordinates are absolute pixels relative to the current vis_image

        # Draw boxes and landmarks
        for i in range(targets.shape[0]):
            box = draw_boxes[i].astype(int)
            pt1 = (max(0, box[0]), max(0, box[1]))
            pt2 = (min(width - 1, box[2]), min(height - 1, box[3]))
            cv2.rectangle(vis_image, pt1, pt2, (0, 255, 0), 2) # Green box

            if draw_landmarks.shape[1] == 10:
                lmks = draw_landmarks[i].reshape(-1, 2) # Keep as float for check
                for lmk in lmks:
                    # *** Check for missing landmarks BEFORE converting to int ***
                    if lmk[0] < 0 or lmk[1] < 0:
                        continue # Skip drawing this landmark if coords are negative

                    # Convert valid landmark to int and draw
                    lmk_int = lmk.astype(int)
                    center = (min(width - 1, max(0, lmk_int[0])), min(height - 1, max(0, lmk_int[1])))
                    cv2.circle(vis_image, center, 3, (0, 0, 255), -1) # Red landmark points

    # --- Saving ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, f"{step_name}.png")
    try:
        cv2.imwrite(filename, vis_image)
        print(f"Saved: {filename} {extra_info}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        print(f"Image shape: {vis_image.shape}, dtype: {vis_image.dtype}")

# --- Main Execution ---
if __name__ == "__main__":
    # Load the image
    image = cv2.imread(INPUT_IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image at {INPUT_IMAGE_PATH}")
        exit()

    print(f"Loaded image: {INPUT_IMAGE_PATH} with shape {image.shape}")
    print(f"Initial targets:\n{INITIAL_TARGETS}")

    # Make copies to avoid modifying the originals
    current_image = image.copy()
    current_targets = INITIAL_TARGETS.copy()
    boxes = current_targets[:, :4].copy()
    labels = current_targets[:, -1].copy()
    landmarks = current_targets[:, 4:-1].copy()

    # --- Step 0: Original Image ---
    visualize_step("0_original", current_image, current_targets)

    # --- Step 1: Crop ---
    # Note: _crop randomly selects a crop. Run multiple times for different results.
    # The function returns original if no suitable crop is found after 250 tries.
    print("\n--- Running Step 1: Crop ---")
    image_after_crop, boxes_after_crop, labels_after_crop, landmarks_after_crop, pad_flag = _crop(
        current_image, boxes, labels, landmarks, IMAGE_SIZE
    )
    targets_after_crop = np.hstack((boxes_after_crop, landmarks_after_crop, np.expand_dims(labels_after_crop, 1)))
    visualize_step("1_cropped", image_after_crop, targets_after_crop, f"Requires Padding: {pad_flag}")
    current_image = image_after_crop
    # Update boxes, labels, landmarks for the next steps *only* if crop was successful
    # If pad_flag is True, it means cropping failed, and we use the original image/targets later for padding.
    if not pad_flag:
        boxes = boxes_after_crop
        labels = labels_after_crop
        landmarks = landmarks_after_crop
        current_targets = targets_after_crop
    else:
        print("Crop attempt failed or skipped, proceeding with original/padded image.")
        # Reset to original image dimensions before padding if crop failed
        current_image = image.copy()
        boxes = INITIAL_TARGETS[:, :4].copy()
        labels = INITIAL_TARGETS[:, -1].copy()
        landmarks = INITIAL_TARGETS[:, 4:-1].copy()
        current_targets = INITIAL_TARGETS.copy()


    # --- Step 2: Distort ---
    print("\n--- Running Step 2: Distort ---")
    image_after_distort = distort_image(current_image)
    # Targets (boxes, landmarks) are not changed by distortion
    visualize_step("2_distorted", image_after_distort, current_targets)
    current_image = image_after_distort

    # --- Step 3: Pad ---
    print("\n--- Running Step 3: Pad ---")
    image_after_pad = pad_image_to_square(current_image, BGR_MEAN, pad_flag)
    # Targets (boxes, landmarks) coordinates don't change relative to the image content,
    # but the image canvas size might change. Visualization uses the current targets.
    visualize_step("3_padded", image_after_pad, current_targets, f"Padding Applied: {pad_flag}")
    current_image = image_after_pad

    # --- Step 4: Horizontal Flip ---
    # Note: Flip happens randomly (default p=0.5). Run multiple times for different results.
    print("\n--- Running Step 4: Horizontal Flip ---")
    image_after_flip, boxes_after_flip, landmarks_after_flip = horizontal_flip(
        current_image, boxes, landmarks # Pass the potentially updated boxes/landmarks
    )
    # Check if flip actually happened by comparing images (or boxes)
    flipped = not np.array_equal(current_image, image_after_flip)
    targets_after_flip = np.hstack((boxes_after_flip, landmarks_after_flip, np.expand_dims(labels, 1)))
    visualize_step("4_flipped", image_after_flip, targets_after_flip, f"Flipped: {flipped}")
    current_image = image_after_flip
    # Update boxes, landmarks, targets for next steps
    boxes = boxes_after_flip
    landmarks = landmarks_after_flip
    current_targets = targets_after_flip


    # --- Step 5: Resize and Subtract Mean ---
    print("\n--- Running Step 5: Resize & Subtract Mean ---")
    # Get dimensions *before* resize for scaling calculation AND final normalization
    h_orig, w_orig, _ = current_image.shape
    image_before_resize = current_image.copy()
    # IMPORTANT: Keep the targets relative to the image *before* resize for final normalization
    targets_before_resize = current_targets.copy()

    # Perform the resize operation
    interpolation_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interpolation = random.choice(interpolation_methods)
    image_resized = cv2.resize(image_before_resize, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interpolation)

    # --- Create SCALED coordinates specifically for VISUALIZING on the resized image ---
    targets_scaled_for_vis = targets_before_resize.copy()
    if w_orig > 0 and h_orig > 0: # Avoid division by zero
        scale_x = IMAGE_SIZE / w_orig
        scale_y = IMAGE_SIZE / h_orig
        print(f"[Step 5 Vis] Scaling coordinates for visualization: sx={scale_x:.3f}, sy={scale_y:.3f}")
        # Scale boxes
        targets_scaled_for_vis[:, 0:4:2] *= scale_x # xmin, xmax
        targets_scaled_for_vis[:, 1:4:2] *= scale_y # ymin, ymax
        # Scale landmarks
        targets_scaled_for_vis[:, 4:-1:2] *= scale_x # lmk_x
        targets_scaled_for_vis[:, 5:-1:2] *= scale_y # lmk_y
    else:
        print("[Step 5 Vis] Warning: Original image dimensions are zero, cannot scale targets.")

    # Visualize the resized image (BEFORE mean subtraction) using the SCALED coordinates
    visualize_step("5a_resized_only", image_resized.copy(), targets_scaled_for_vis, f"Interpolation: {interpolation}")

    # Apply the full resize_subtract_mean logic (returns CHW float image)
    final_processed_image = resize_subtract_mean(image_before_resize, IMAGE_SIZE, BGR_MEAN)

    # Visualize the final processed image (AFTER mean subtraction) using the SAME SCALED coordinates
    # visualize_step will handle reversing the mean/transposing the image AND drawing the scaled coordinates
    visualize_step("5b_resized_mean_subtracted", final_processed_image.copy(), targets_scaled_for_vis, "(Visually Adjusted)")

    print(f"Final processed image shape (C, H, W): {final_processed_image.shape}, dtype: {final_processed_image.dtype}")


    # --- Step 6: Normalize Targets ---
    print("\n--- Running Step 6: Normalize Targets ---")
    # Normalize the targets that were relative to the image dimensions *before* resize (w_orig, h_orig)
    final_targets = targets_before_resize.copy() # Start from coordinates before resize
    if w_orig > 0 and h_orig > 0:
        final_targets[:, 0::2] /= w_orig # Normalize x by original width
        final_targets[:, 1::2] /= h_orig # Normalize y by original height
        print(f"Normalized targets using original dimensions W={w_orig}, H={h_orig}")
    else:
         print("[Step 6] Warning: Original image dimensions are zero, cannot normalize targets.")

    print(f"Final normalized targets (first 5 rows):\n{final_targets[:5, :]}")

    # Optional: Visualize final *normalized* targets on the *resized* image.
    # visualize_step should detect these normalized (0-1) coords and scale them using IMAGE_SIZE
    # visualize_step("6_final_normalized_targets", image_resized.copy(), final_targets.copy())

    print(f"\nAugmentation steps saved in directory: {OUTPUT_DIR}")

