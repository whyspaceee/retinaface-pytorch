import argparse
import os
import cv2
import numpy as np
import torch
import re
import torchvision.ops as tvops
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from config import get_config
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landmarks
import csv
from collections import defaultdict
import time # <--- Added import

# Initialize storage for per tilt-pan results
eval_results = defaultdict(lambda: {
    "total_images": 0,
    "true_positives": 0,
    "false_positives": 0,
    "false_negatives": 0,
})

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Ultra-Optimized RetinaFace Inference with Batched Processing"
    )
    # Model and device options
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights/Resnet34_Final.pth', # Example path, adjust if needed
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='mobilenetv2', # Changed default to match weights example
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network'
    )
    # Detection settings
    parser.add_argument('--conf-threshold', type=float, default=0.9, help='Confidence threshold')
    parser.add_argument('--pre-nms-topk', type=int, default=5000, help='Max detections before NMS')
    parser.add_argument('--nms-threshold', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('--post-nms-topk', type=int, default=750, help='Max detections after NMS')
    # Output options
    parser.add_argument('-s', '--save-image', action='store_true', help='Save successful detection result images (separate from failures)')
    parser.add_argument('-v', '--vis-threshold', type=float, default=0.6, help='Visualization threshold (unused in current eval logic)')
    # Image input (root directory for evaluation)
    parser.add_argument('--image-path', type=str, default='data/HeadPoseImageDatabase', help='Root directory of the Head Pose Image Database') # Changed default
    # Rescale option: if set, images are resized to a small size and embedded into a fixed 640x640 canvas
    parser.add_argument('--rescale', action='store_true', help='Rescale images to a fixed canvas')
    # TorchScript flag
    parser.add_argument('--jit', action='store_true', help='Convert model to TorchScript for faster inference')
    # Batch size for batched inference (works only with --rescale)
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for inference')
    return parser.parse_args()

def compute_iou_vectorized(gt_box, pred_boxes):
    """
    Compute IoU between one ground-truth box and multiple predicted boxes.
    Boxes are in [x_min, y_min, x_max, y_max] format.
    """
    xA = np.maximum(gt_box[0], pred_boxes[:, 0])
    yA = np.maximum(gt_box[1], pred_boxes[:, 1])
    xB = np.minimum(gt_box[2], pred_boxes[:, 2])
    yB = np.minimum(gt_box[3], pred_boxes[:, 3])

    # Ensure width and height are non-negative
    interW = np.maximum(0, xB - xA) # No need for +1 if comparing areas directly
    interH = np.maximum(0, yB - yA) # No need for +1 if comparing areas directly
    interArea = interW * interH

    boxAArea = np.maximum(0, gt_box[2] - gt_box[0]) * np.maximum(0, gt_box[3] - gt_box[1])
    boxesArea = np.maximum(0, pred_boxes[:, 2] - pred_boxes[:, 0]) * np.maximum(0, pred_boxes[:, 3] - pred_boxes[:, 1])

    # Handle division by zero
    denominator = boxAArea + boxesArea - interArea
    iou = np.divide(interArea, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator!=0)

    return iou


@torch.no_grad()
def batched_inference(model, images_tensor, device):
    """Runs batched inference and measures time.""" # <--- Docstring updated
    model.eval()
    loc, conf, landmarks = None, None, None
    elapsed_time = 0.0

    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # --- Start timed region ---
        with torch.amp.autocast("cuda"): # Use AMP context manager
            loc, conf, landmarks = model(images_tensor)
        # --- End timed region ---
        end_event.record()
        torch.cuda.synchronize() # Wait for operations to complete
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0 # Time in seconds
    else: # CPU or other devices
        start_time = time.perf_counter()
        # --- Start timed region ---
        loc, conf, landmarks = model(images_tensor)
        # --- End timed region ---
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time # Time in seconds

    return loc, conf, landmarks, elapsed_time # <--- Return elapsed time

def batched_detectRetinaface(params, model, device, images, cfg, prior_cache):
    """
    Process a batch of images (numpy array of shape [B, H, W, 3]).
    Returns a list of detection arrays (one per image) and the inference time for the batch.
    """
    B, H, W, _ = images.shape
    rgb_mean = np.array([104, 117, 123], dtype=np.float32)
    images_processed = images.astype(np.float32) - rgb_mean  # vectorized subtraction
    images_tensor = torch.from_numpy(images_processed).permute(0, 3, 1, 2).to(device)

    # --- Get model output and inference time ---
    loc, conf, landmarks, batch_inference_time = batched_inference(model, images_tensor, device) # <--- Capture time

    detections_list = []
    for i in range(B):
        loc_i = loc[i].unsqueeze(0)   # shape: (1, num_priors, 4)
        conf_i = conf[i].unsqueeze(0) # shape: (1, num_priors, 2)
        # Decode boxes using the cached priors
        boxes = decode(loc_i.squeeze(0), prior_cache, cfg['variance'])
        bbox_scale = torch.tensor([W, H] * 2, device=device)
        boxes = (boxes * bbox_scale).cpu().numpy()
        scores = conf_i.squeeze(0).cpu().numpy()[:, 1]

        # Filter and sort
        inds = scores > params.conf_threshold
        boxes = boxes[inds]
        scores_filtered = scores[inds]
        order = scores_filtered.argsort()[::-1][:params.pre_nms_topk]
        boxes = boxes[order]
        scores_filtered = scores_filtered[order]

        # Apply NMS using torchvision (fast C++/CUDA implementation)
        if boxes.shape[0] > 0:
            boxes_tensor = torch.from_numpy(boxes).to(device)
            scores_tensor = torch.from_numpy(scores_filtered).to(device)
            keep = tvops.nms(boxes_tensor, scores_tensor, params.nms_threshold)
            keep = keep.cpu().numpy()
            # Combine boxes and scores BEFORE applying post_nms_topk
            detections = np.hstack((boxes, scores_filtered[:, None])).astype(np.float32)[keep]
            detections = detections[:params.post_nms_topk]
        else:
            detections = np.empty((0, 5), dtype=np.float32)
        # Ensure detections are integers for consistency and potential drawing
        detections_list.append(detections.astype(int))

    return detections_list, batch_inference_time # <--- Return time

def convert_to_bbox(center_x, center_y, width, height):
    x_min = center_x - (width / 2)
    y_min = center_y - (height / 2)
    x_max = center_x + (width / 2)
    y_max = center_y + (height / 2)
    return int(x_min), int(y_min), int(x_max), int(y_max)

def extract_tilt_pan(text):
    pattern = r'.*?([+-]\d+)([+-]\d+)(?:\.jpg)?$'
    match = re.search(pattern, text)
    if match:
        tilt = int(match.group(1))
        pan = int(match.group(2))
        return tilt, pan
    else:
        # Return None if not found, allowing processing to continue
        return None, None # Modified to not raise error

def load_image_and_annotation(person_dir, groundtruth, scaled_size, fixed_canvas_size):
    """
    Loads an image and its annotation, rescales it to scaled_size, and embeds it in a fixed canvas.
    Also transforms the ground-truth coordinates and extracts tilt/pan from the image name.
    """
    mat_path = os.path.join(person_dir, groundtruth)
    base_name = os.path.splitext(groundtruth)[0]
    image_name = f"{base_name}.jpg"
    image_full_path = os.path.join(person_dir, image_name)

    if not os.path.exists(image_full_path):
        # print(f"Image file not found: {image_full_path}") # Optional: uncomment for debugging
        return None

    try:
        with open(mat_path, "r") as f:
            lines = f.readlines()
        # Assuming lines[3-6] contain x, y, w, h relative to original image
        x = int(lines[3])
        y = int(lines[4])
        w = int(lines[5])
        h = int(lines[6])
    except Exception as e:
        print(f"Error reading annotation {mat_path}: {e}")
        return None

    img = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not load image: {image_name}")
        return None
    orig_h, orig_w = img.shape[:2]

    # Resize image to scaled_size (maintaining aspect ratio for embedding might be better,
    # but following script's logic of direct resize)
    resized_img = cv2.resize(img, (scaled_size, scaled_size))

    # Embed resized image in a fixed canvas (e.g., 640x640)
    canvas = np.zeros((fixed_canvas_size[1], fixed_canvas_size[0], 3), dtype=img.dtype)
    offset_x = (fixed_canvas_size[0] - scaled_size) // 2
    offset_y = (fixed_canvas_size[1] - scaled_size) // 2
    canvas[offset_y:offset_y+scaled_size, offset_x:offset_x+scaled_size] = resized_img

    # Transform ground-truth bounding box coordinates to the canvas space
    scale_x = scaled_size / orig_w
    scale_y = scaled_size / orig_h
    x_new = int(x * scale_x + offset_x)
    y_new = int(y * scale_y + offset_y)
    w_new = int(w * scale_x)
    h_new = int(h * scale_y)
    x_min, y_min, x_max, y_max = convert_to_bbox(x_new, y_new, w_new, h_new)
    gt_box = [x_min, y_min, x_max, y_max]

    # Attempt to extract tilt and pan from the image name.
    tilt, pan = extract_tilt_pan(image_name)
    # No error print here, just pass None if not found

    return {
        "image_name": image_name,
        "image": canvas, # The image embedded in the canvas
        "gt_box": gt_box, # GT box coordinates relative to the canvas
        "tilt": tilt,
        "pan": pan,
    }

def write_csv_results(results, output_csv, args, scale_value):
    """
    Write the evaluation results (aggregated per tilt and pan) to a CSV file.
    """
    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = ["tilt", "pan", "total_images", "true_positives", "false_positives", "false_negatives",
                      "precision", "recall", "f1_score", "accuracy", "conf_threshold", "scale"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Sort keys for consistency
        for key in sorted(results.keys()):
            tp = results[key]["true_positives"]
            fp = results[key]["false_positives"]
            fn = results[key]["false_negatives"]
            total = results[key]["total_images"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0 # Note: Accuracy definition might vary
            writer.writerow({
                "tilt": key[0],
                "pan": key[1],
                "total_images": total,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "f1_score": f"{f1_score:.4f}",
                "accuracy": f"{accuracy:.4f}", # Including accuracy for completeness
                "conf_threshold": args.conf_threshold,
                "scale": scale_value
            })

def convert_aflw_to_wider_detection(root_dir, output_root, args, model, device, cfg):
    """
    Converts AFLW annotations to WIDER FACE detection format and evaluates detections.
    When --rescale is enabled, images are processed in batches.
    Also aggregates evaluation results per tilt and pan and outputs a CSV file.
    Reports average inference time.
    Saves failed detection images (False Negatives) to './gagal'. <<< MODIFIED
    """
    total_images = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_threshold = 0.1 # IoU threshold to consider a detection a match

    # --- Variables for timing ---
    total_inference_time = 0.0
    total_images_timed = 0 # Count images included in timing calculation

    # <<< ADDED: Define and create the failure output directory >>>
    failure_output_dir = "./gagal"
    os.makedirs(failure_output_dir, exist_ok=True)
    print(f"Saving failed detection images (FNs) to: {failure_output_dir}")
    # <<< END ADDED >>>

    # This value is used when rescaling
    scaled_size = 48 # Example scale value, adjust as needed

    if args.rescale:
        fixed_canvas_size = (640, 640) # Example canvas size
        tasks = []
        # Gather image tasks from the dataset
        print("Loading images and annotations for batched processing...")
        all_persons = [p for p in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, p)) and p != "Front"]
        for person in tqdm(all_persons, desc="Persons"):
            person_dir = os.path.join(root_dir, person)
            all_groundtruths = [gt for gt in os.listdir(person_dir) if gt.endswith('.txt')]
            for groundtruth in all_groundtruths: # Removed inner tqdm
                task = load_image_and_annotation(person_dir, groundtruth, scaled_size, fixed_canvas_size)
                if task is not None:
                    tasks.append(task)

        if not tasks:
            print("Error: No valid image/annotation pairs found for processing.")
            return

        batch_size = args.batch_size
        # Precompute the anchor boxes (prior cache) for fixed canvas size
        priorbox = PriorBox(cfg, image_size=fixed_canvas_size)
        prior_cache = priorbox.generate_anchors().to(device)

        print(f"Processing {len(tasks)} images in batches...")
        # Process tasks in batches
        for i in tqdm(range(0, len(tasks), batch_size), desc="Batches"):
            batch_tasks = tasks[i:i+batch_size]
            if not batch_tasks: # Skip if batch is empty
                continue
            images_batch = np.stack([t["image"] for t in batch_tasks], axis=0)

            # --- Perform detection and get batch inference time ---
            detections_list, batch_inf_time = batched_detectRetinaface(args, model, device, images_batch, cfg, prior_cache)
            total_inference_time += batch_inf_time # Accumulate total time
            num_images_in_batch = len(batch_tasks)
            total_images_timed += num_images_in_batch # Accumulate number of images processed

            for j, task in enumerate(batch_tasks):
                gt_box = np.array(task["gt_box"])
                detections = detections_list[j] # Already int type from batched_detectRetinaface
                image_name = task["image_name"]
                image_canvas = task["image"] # Use the padded canvas image

                # --- Evaluation Logic (TP, FP, FN for single GT box per image) ---
                is_tp = False
                current_fp = 0
                current_fn = 0

                if detections.size == 0:
                    # Ground truth exists, no detection -> False Negative
                    current_fn = 1
                else:
                    # Ensure detections have 4 columns for IoU calc
                    pred_boxes = detections[:, :4].astype(np.float32)
                    ious = compute_iou_vectorized(gt_box, pred_boxes)
                    # Check if any detection overlaps significantly with the single GT box
                    match_indices = np.where(ious >= iou_threshold)[0]

                    if len(match_indices) > 0:
                        # At least one detection matches the GT -> True Positive
                        is_tp = True
                        # If multiple detections match the single GT, count extras as FP
                        # All non-matching detections are also FP
                        current_fp = len(detections) - 1 # Count all but the one best match as FP
                    else:
                        # No detection matches the GT -> False Negative
                        current_fn = 1
                        # All detections are false positives as none matched the GT
                        current_fp = len(detections)

                # --- Update overall counts ---
                if is_tp:
                    total_tp += 1
                total_fp += current_fp
                total_fn += current_fn
                total_images += 1 # Increment total image count here

                # --- Update per tilt-pan results if available ---
                tilt = task.get("tilt")
                pan = task.get("pan")
                if tilt is not None and pan is not None:
                    eval_results[(tilt, pan)]["total_images"] += 1
                    if is_tp:
                        eval_results[(tilt, pan)]["true_positives"] += 1
                    eval_results[(tilt, pan)]["false_positives"] += current_fp
                    eval_results[(tilt, pan)]["false_negatives"] += current_fn

                # --- Save regular output image if requested ---
                if args.save_image:
                    img_vis = image_canvas.copy()
                    # GT in Blue (BGR)
                    cv2.rectangle(img_vis, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, 0), 2)
                    # Detections in Green (BGR)
                    for rec in detections: # detections are already int
                        cv2.rectangle(img_vis, (rec[0], rec[1]), (rec[2], rec[3]), (0, 255, 0), 2)
                    output_dir = os.path.join(output_root, "HPID_Rescaled_Output", args.weights, str(scaled_size)) # Changed output dir name
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, image_name)
                    cv2.imwrite(output_path, img_vis)


    else: # Fallback: process images one by one (original behavior)
        print("Processing images individually (no --rescale)...")
        all_persons = [p for p in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, p)) and p != "Front"]
        for person in tqdm(all_persons, desc="Persons"):
            person_dir = os.path.join(root_dir, person)
            all_groundtruths = [gt for gt in os.listdir(person_dir) if gt.endswith('.txt')]
            for groundtruth in tqdm(all_groundtruths, desc=f"Processing {person}", leave=False):
                mat_path = os.path.join(person_dir, groundtruth)
                base_name = os.path.splitext(groundtruth)[0]
                image_name = f"{base_name}.jpg"
                image_path = os.path.join(person_dir, image_name)

                # Extract tilt/pan early to potentially skip if needed later
                tilt, pan = extract_tilt_pan(image_name)

                if not os.path.exists(image_path):
                    continue # Skip missing image

                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue # Skip unloadable image

                # --- Perform detection and get single image inference time ---
                # detectRetinaface needs to be defined or imported if used here
                # Assuming detectRetinaface returns (detections_array, time)
                detections, single_inf_time = detectRetinaface(args, model, device, img, cfg) # Capture time
                total_inference_time += single_inf_time # Accumulate time
                total_images_timed += 1                 # Accumulate count

                try:
                    with open(mat_path, "r") as f:
                        lines = f.readlines()
                    # Assuming lines[3-6] contain x, y, w, h relative to original image
                    x = int(lines[3])
                    y = int(lines[4])
                    w = int(lines[5])
                    h = int(lines[6])
                    x_min, y_min, x_max, y_max = convert_to_bbox(x, y, w, h)
                    gt_box = np.array([x_min, y_min, x_max, y_max]) # Use numpy array
                except Exception as e:
                    print(f"Error reading ground truth {mat_path}: {e}")
                    continue # Skip this image if GT is bad

                # --- Evaluation Logic (TP, FP, FN) - same as batched ---
                is_tp = False
                current_fp = 0
                current_fn = 0

                if detections.size == 0:
                    current_fn = 1
                else:
                    # Ensure detections have 4 columns and are float for IoU
                    pred_boxes = detections[:, :4].astype(np.float32)
                    ious = compute_iou_vectorized(gt_box, pred_boxes)
                    match_indices = np.where(ious >= iou_threshold)[0]
                    if len(match_indices) > 0:
                        is_tp = True
                        current_fp = len(detections) - 1
                    else:
                        current_fn = 1
                        current_fp = len(detections)

                # --- Update overall counts ---
                if is_tp:
                    total_tp += 1
                total_fp += current_fp
                total_fn += current_fn
                total_images += 1 # Increment total image count

                # --- Update per tilt-pan results ---
                if tilt is not None and pan is not None:
                    eval_results[(tilt, pan)]["total_images"] += 1
                    if is_tp:
                        eval_results[(tilt, pan)]["true_positives"] += 1
                    eval_results[(tilt, pan)]["false_positives"] += current_fp
                    eval_results[(tilt, pan)]["false_negatives"] += current_fn

                # --- Save regular output image if requested ---
                if args.save_image:
                    img_vis = img.copy() # Use the original image for visualization here
                    # GT Blue (BGR)
                    cv2.rectangle(img_vis, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, 0), 2)
                     # Detections Green (BGR) - ensure int coords
                    for rec in detections: # detections are already int
                        cv2.rectangle(img_vis, (rec[0], rec[1]), (rec[2], rec[3]), (0, 255, 0), 2)
                    output_dir = os.path.join(output_root, "0--AFLW_Original_Output") # Different output dir
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, image_name) # Use image_name here
                    cv2.imwrite(output_path, img_vis)

                # <<< --- ADDED: Save failure image if it's a False Negative --- >>>
                if current_fn == 1:
                    img_to_save = img.copy() # Use the original image
                    # Draw GT box (Red BGR)
                    cv2.rectangle(img_to_save, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 255), 2)
                    # Draw ALL actual detections (Green BGR) - ensure int coords
                    for rec in detections: # detections are already int
                        cv2.rectangle(img_to_save, (rec[0], rec[1]), (rec[2], rec[3]), (0, 255, 0), 2)
                    save_path = os.path.join(failure_output_dir, image_name)
                    cv2.imwrite(save_path, img_to_save)
                # <<< --- END ADDED --- >>>

    # --- Calculate and print final results ---
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    # --- Calculate Average Inference Time ---
    avg_inference_time_ms = 0
    if total_images_timed > 0:
        avg_inference_time_s = total_inference_time / total_images_timed
        avg_inference_time_ms = avg_inference_time_s * 1000 # Convert to milliseconds

    print("\n--- Overall Evaluation Results ---")
    print(f"Total images processed: {total_images}")
    print(f"True Positives (TP): {total_tp}")
    print(f"False Positives (FP): {total_fp}")
    print(f"False Negatives (FN): {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    # print(f"Accuracy (TP / (TP+FP+FN)): {accuracy:.4f}") # Uncomment if using this accuracy definition
    print(f"Confidence threshold: {args.conf_threshold}")
    current_scale_value = 'original'
    if args.rescale:
        current_scale_value = scaled_size
        print(f"Scale value used for resizing: {scaled_size}")
        print(f"Canvas size: {fixed_canvas_size}")
    # --- Print Average Inference Time ---
    print(f"Avg. Inference Time per Image: {avg_inference_time_ms:.2f} ms ({total_images_timed} images timed)")
    print(f"Device used: {device}")
    print(f"Network: {args.network}")
    if args.jit:
        print("Using TorchScript model.")
    if args.rescale:
        print(f"Using batched inference with batch size: {args.batch_size}")


    # Write the per tilt-pan results to CSV
    csv_scale_tag = current_scale_value if args.rescale else 'orig'
    output_csv = os.path.join(output_root, f"evaluation_results_{args.network}_conf{args.conf_threshold}_scale{csv_scale_tag}.csv")
    write_csv_results(eval_results, output_csv, args, scale_value=current_scale_value)
    print(f"\nPer tilt-pan evaluation CSV saved to: {output_csv}")


# =====================================================
# Single Image Detection Function (Needed for non-rescale mode)
# =====================================================
def detectRetinaface(params, model, device, original_image, cfg):
    """Detects faces, returns detections and inference time.""" # <--- Docstring updated
    rgb_mean = np.array([104, 117, 123], dtype=np.float32)
    image = np.float32(original_image) - rgb_mean
    img_height, img_width, _ = image.shape
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    # --- Get model output and inference time ---
    loc, conf, landmarks, inference_time = inference(model, image, device) # <--- Capture time

    # --- Post-processing ---
    # Use actual image size for PriorBox when not rescaling
    priorbox = PriorBox(cfg, image_size=(img_width, img_height))
    priors = priorbox.generate_anchors().to(device)
    boxes = decode(loc.squeeze(0), priors, cfg['variance'])
    bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
    boxes = (boxes * bbox_scale).cpu().numpy()
    scores = conf.squeeze(0).cpu().numpy()[:, 1]

    inds = scores > params.conf_threshold
    boxes = boxes[inds]
    scores_filtered = scores[inds]

    # Sort by score
    order = scores_filtered.argsort()[::-1][:params.pre_nms_topk] # Apply pre_nms_topk
    boxes = boxes[order]
    scores_filtered = scores_filtered[order]

    # Apply NMS
    if boxes.shape[0] > 0:
        boxes_tensor = torch.from_numpy(boxes).to(device)
        scores_tensor = torch.from_numpy(scores_filtered).to(device)
        keep = tvops.nms(boxes_tensor, scores_tensor, params.nms_threshold)
        keep = keep.cpu().numpy()
         # Combine boxes and scores BEFORE applying post_nms_topk
        detections = np.hstack((boxes, scores_filtered[:, None])).astype(np.float32)[keep]
        detections = detections[:params.post_nms_topk] # Apply post_nms_topk
    else:
        detections = np.empty((0, 5), dtype=np.float32)

    # Return detections as integers and the inference time
    return detections.astype(int), inference_time # <--- Return time

@torch.no_grad()
def inference(model, image, device):
    """Runs single image inference and measures time.""" # <--- Docstring updated
    model.eval()
    loc, conf, landmarks = None, None, None
    elapsed_time = 0.0

    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # --- Start timed region ---
        with torch.amp.autocast("cuda"): # Use AMP for single inference too
            loc, conf, landmarks = model(image)
        # --- End timed region ---
        end_event.record()
        torch.cuda.synchronize() # Wait for GPU operations to complete
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0 # Time in seconds
    else: # CPU
        start_time = time.perf_counter()
        # --- Start timed region ---
        loc, conf, landmarks = model(image)
        # --- End timed region ---
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time # Time in seconds

    return loc, conf, landmarks, elapsed_time # <--- Return elapsed time


if __name__ == "__main__":
    args = parse_arguments()

    # Use the image path argument as the root directory
    root_dir = args.image_path
    output_root = r"output" # Example path for outputs (CSV, optional saved images)
    os.makedirs(output_root, exist_ok=True) # Ensure output directory exists

    cfg = get_config(args.network)
    if cfg is None:
        raise KeyError(f"Config file for {args.network} not found!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load the model
    print(f"Initializing model {args.network}...")
    model = RetinaFace(cfg=cfg)
    # Load weights BEFORE potential JIT compilation
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights} to device {device}...")
        # Use weights_only=True for security if loading untrusted checkpoints
        # Set to False if your checkpoint contains more than just weights (e.g., optimizer state)
        # and you trust the source. For inference, True is generally safer.
        try:
           state_dict = torch.load(args.weights, map_location=device, weights_only=True)
           model.load_state_dict(state_dict)
           print("Model weights loaded successfully!")
        except Exception as e:
            print(f"Error loading weights: {e}. Check the weights file and network compatibility.")
            exit() # Exit if weights can't be loaded
    else:
        print(f"Error: Weights file not found at {args.weights}")
        exit()

    model.to(device)
    model.eval() # Set to eval mode


    # Optional: Convert model to TorchScript for faster inference
    if args.jit:
        print("Attempting to convert model to TorchScript...")
        # Determine dummy input size based on rescale or a default
        # Use the actual canvas size if rescaling, otherwise a typical size
        dummy_input_size = (640, 640) if args.rescale else (480, 640) # Adjust default if needed
        # Create dummy input with batch dimension 1
        dummy_input = torch.randn(1, 3, dummy_input_size[1], dummy_input_size[0], device=device)
        try:
            model = torch.jit.trace(model, dummy_input)
            model.eval() # Ensure JIT model is in eval mode
            print("Model successfully converted to TorchScript!")
        except Exception as e:
            print(f"Warning: TorchScript conversion failed: {e}. Running with standard PyTorch model.")
            args.jit = False # Disable JIT if conversion fails

    # Run conversion and evaluation on the AFLW dataset
    print(f"Starting evaluation on dataset: {root_dir}...")
    convert_aflw_to_wider_detection(root_dir, output_root, args, model, device, cfg)
    print(f"\nProcessing finished. Evaluation results saved in: {output_root}")
    print(f"Failed detection images (FNs) saved in: ./gagal")
