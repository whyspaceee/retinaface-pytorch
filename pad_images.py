import os
import cv2
import numpy as np
import random
import argparse
from tqdm import tqdm
import math

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Randomly select images, resize maintaining aspect ratio based on longer side, "
                    "pad onto a canvas for multiple target sizes, and save."
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        help='Directory containing the original image files (e.g., AFLW2000).',
        default='./data/HeadPoseImageDatabase/Person09'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Root directory to save the structured padded output images (subdirs like scaled_48, original will be created).',
        default='./data/padded_images/hpid'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=8,
        help='Number of random images to process for each size.'
    )
    parser.add_argument(
        '--canvas-size',
        type=int,
        default=640,
        help='The square size (width and height) of the final canvas for padding.'
    )
    # No --scaled-size argument here, as we iterate through predefined sizes
    return parser.parse_args()

def resize_maintaining_aspect_ratio(img, target_longer_side):
    """
    Resizes an image so its longer side matches target_longer_side, maintaining aspect ratio.
    """
    orig_h, orig_w = img.shape[:2]

    if orig_w == 0 or orig_h == 0:
        # Avoid division by zero for invalid images
        return None

    if orig_w >= orig_h: # Wider or square image
        scale_ratio = target_longer_side / orig_w
        new_w = target_longer_side
        new_h = int(orig_h * scale_ratio)
    else: # Taller image
        scale_ratio = target_longer_side / orig_h
        new_h = target_longer_side
        new_w = int(orig_w * scale_ratio)

    # Ensure dimensions are at least 1
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # Choose interpolation based on whether we are shrinking or enlarging
    if scale_ratio < 1.0:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    return resized_img

def process_and_pad_image(image_path, output_path, target_size_config, canvas_size):
    """
    Reads an image, processes it based on target_size_config (resize or original),
    pads it onto a centered canvas (resizing down if needed to fit), and saves it.
    target_size_config can be an integer (target longer side) or the string 'original'.
    """
    try:
        # Read the original image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return False

        # --- Determine the image to embed ---
        if target_size_config == 'original':
            img_to_embed = img.copy()
        elif isinstance(target_size_config, int):
            img_to_embed = resize_maintaining_aspect_ratio(img, target_size_config)
            if img_to_embed is None:
                 print(f"Warning: Could not resize image {image_path} (invalid dimensions?). Skipping.")
                 return False
        else:
            print(f"Warning: Invalid target_size_config '{target_size_config}' for {image_path}. Skipping.")
            return False

        # --- Check if the image to embed fits the canvas ---
        embed_h, embed_w = img_to_embed.shape[:2]

        if embed_h > canvas_size or embed_w > canvas_size:
            print(f"  Info: Image '{os.path.basename(image_path)}' (size {embed_w}x{embed_h} for config '{target_size_config}')"
                  f" is larger than canvas ({canvas_size}x{canvas_size}). Resizing down to fit.")
            # Calculate the necessary downscale ratio
            downscale_ratio = min(canvas_size / embed_h, canvas_size / embed_w)
            final_w = max(1, int(embed_w * downscale_ratio))
            final_h = max(1, int(embed_h * downscale_ratio))
            img_to_embed = cv2.resize(img_to_embed, (final_w, final_h), interpolation=cv2.INTER_AREA)
            embed_h, embed_w = img_to_embed.shape[:2] # Update dimensions

        # --- Create canvas and embed ---
        # Ensure the canvas has the same number of channels and data type
        num_channels = img_to_embed.shape[2] if img_to_embed.ndim == 3 else 1
        canvas_shape = (canvas_size, canvas_size, num_channels) if num_channels > 1 else (canvas_size, canvas_size)
        canvas = np.zeros(canvas_shape, dtype=img_to_embed.dtype)

        # Calculate offsets to center the potentially resized image on the canvas
        offset_y = (canvas_size - embed_h) // 2
        offset_x = (canvas_size - embed_w) // 2

        # Embed the image onto the canvas
        if num_channels > 1:
             canvas[offset_y:offset_y + embed_h, offset_x:offset_x + embed_w, :] = img_to_embed
        else:
             canvas[offset_y:offset_y + embed_h, offset_x:offset_x + embed_w] = img_to_embed


        # Save the resulting padded image
        cv2.imwrite(output_path, canvas)
        return True

    except Exception as e:
        print(f"Error processing image {image_path} for config '{target_size_config}': {e}")
        return False

if __name__ == "__main__":
    args = parse_arguments()

    # --- Validate inputs and Setup ---
    if not os.path.isdir(args.images_dir):
        print(f"Error: Image directory not found: {args.images_dir}")
        exit(1)

    # Define the target size configurations to iterate through
    target_size_configs = [48,56, 64, 128, 'original']

    # --- Get list of all available image files ---
    try:
        all_files = os.listdir(args.images_dir)
        # Filter for common image extensions
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    except OSError as e:
        print(f"Error reading image directory {args.images_dir}: {e}")
        exit(1)

    if not image_files:
        print(f"Error: No image files found in {args.images_dir}")
        exit(1)

    # --- Select Random Images (do this *once* before looping through sizes) ---
    num_available = len(image_files)
    num_to_select = min(args.num_images, num_available) # Ensure we don't request more than available
    if num_to_select < args.num_images:
        print(f"Warning: Requested {args.num_images} images, but only found {num_available}. Processing {num_to_select}.")

    selected_files = random.sample(image_files, num_to_select)
    print(f"Randomly selected {num_to_select} images to process for each size configuration.")
    print("-" * 30)

    # --- Process for each target size configuration ---
    total_processed_count = 0
    # Outer loop with tqdm for sizes
    for config in tqdm(target_size_configs, desc="Overall Size Configs"):
        size_str = f"scaled_{config}" if isinstance(config, int) else str(config)
        current_output_dir = os.path.join(args.output_dir, size_str)

        print(f"\nProcessing for configuration: '{config}' -> {current_output_dir}")
        os.makedirs(current_output_dir, exist_ok=True)

        processed_count_for_size = 0
        # Inner loop with tqdm for images within this size config
        for filename in tqdm(selected_files, desc=f"Images for '{size_str}'", leave=False):
            input_path = os.path.join(args.images_dir, filename)
            # Use a consistent output filename structure
            output_filename = f"{os.path.splitext(filename)[0]}_padded_{size_str}{os.path.splitext(filename)[1]}"
            output_path = os.path.join(current_output_dir, output_filename)

            if process_and_pad_image(input_path, output_path, config, args.canvas_size):
                processed_count_for_size += 1

        print(f"Finished processing for '{config}'. {processed_count_for_size}/{num_to_select} images saved.")
        total_processed_count += processed_count_for_size
        print("-" * 30)


    print(f"\nProcessing complete. A total of {total_processed_count} image processing operations were attempted across {len(target_size_configs)} configurations.")
    print(f"Output saved in subdirectories under: {args.output_dir}")