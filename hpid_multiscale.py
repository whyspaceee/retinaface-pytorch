import argparse
import os
import cv2
import numpy as np
import torch
import re
import torchvision.ops as tvops
# from concurrent.futures import ThreadPoolExecutor # No longer needed
from tqdm import tqdm
from config import get_config
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode # Assuming decode_landmarks is not strictly needed based on usage
import csv
from collections import defaultdict
import time # Import time for inference measurement

# Initialize storage for per tilt-pan results
eval_results = defaultdict(lambda: {
    "total_images": 0,
    "true_positives": 0,
    "false_positives": 0,
    "false_negatives": 0,
})

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="RetinaFace Inference (Sequential, Optional Rescale)"
    )
    # Model and device options
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights/Resnet34_Final.pth', # Example, adjust as needed
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='mobilenetv2', # Example, ensure it matches weights
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network'
    )
    # Detection settings
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold for filtering detections')
    parser.add_argument('--pre-nms-topk', type=int, default=5000, help='Max detections before NMS')
    parser.add_argument('--nms-threshold', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('--post-nms-topk', type=int, default=750, help='Max detections after NMS')
    # Output options
    # parser.add_argument('-s', '--save-image', action='store_true', help='Save detection result images') # Images are always saved now
    parser.add_argument('-v', '--vis-threshold', type=float, default=0.6, help='Visualization threshold (currently unused in drawing)') # Note: not used, but kept
    # Image input source directory
    parser.add_argument('--image-dir', type=str, default='data/HeadPoseImageDatabase', help='Root directory of the image dataset')
    # Output directory
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save results and images')
    # Rescale option: if set, images are ONLY resized to scaled-size x scaled-size (no padding)
    parser.add_argument('--rescale', action='store_true', help='Rescale images to scaled-size (no padding/canvas)')
    parser.add_argument('--scaled-size', type=int, default=48, help='Target size (width and height) for rescaling')
    # TorchScript flag
    parser.add_argument('--jit', action='store_true', help='Convert model to TorchScript for potentially faster inference')
    # Removed --batch-size
    return parser.parse_args()

def compute_iou_vectorized(gt_box, pred_boxes):
    """
    Compute IoU between one ground-truth box and multiple predicted boxes.
    Boxes are in [x_min, y_min, x_max, y_max] format.
    Assumes pred_boxes is a 2D numpy array [N, 4].
    """
    if pred_boxes.size == 0:
        return np.array([], dtype=np.float32)

    xA = np.maximum(gt_box[0], pred_boxes[:, 0])
    yA = np.maximum(gt_box[1], pred_boxes[:, 1])
    xB = np.minimum(gt_box[2], pred_boxes[:, 2])
    yB = np.minimum(gt_box[3], pred_boxes[:, 3])

    # Intersection dimensions, ensure non-negative
    interW = np.maximum(0, xB - xA) # Removed +1 assuming coords are pixel boundaries
    interH = np.maximum(0, yB - yA) # Removed +1
    interArea = interW * interH

    boxAArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    boxesArea = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])

    # Union Area calculation, add epsilon to avoid division by zero
    unionArea = boxAArea + boxesArea - interArea
    iou = interArea / (unionArea + 1e-6) # Add epsilon for stability

    return iou


@torch.no_grad()
def inference(model, image_tensor, device):
    """ Performs model inference and measures time. """
    model.eval()
    inf_time = 0 # Default value
    try:
        start_time = time.time()
        if device.type == 'cuda':
             # Use AMP for potential speedup on CUDA
            with torch.cuda.amp.autocast():
                loc, conf, landmarks = model(image_tensor)
        else:
            # Run on CPU or other devices
            loc, conf, landmarks = model(image_tensor)
        end_time = time.time()
        inf_time = end_time - start_time
    except Exception as e:
        print(f"Error during inference: {e}")
        # Return dummy tensors or handle error as appropriate
        # For simplicity, let's return None and handle it upstream
        return None, None, None, 0 # Indicate error with None

    return loc, conf, landmarks, inf_time


def detectRetinaface(params, model, device, processed_tensor, cfg):
    """
    Performs detection on a single preprocessed image tensor.
    Returns detections and inference time.
    """
    # processed_tensor shape expected: [1, C, H, W]
    _, _, img_height, img_width = processed_tensor.shape # Get H, W from tensor

    # Perform inference
    loc, conf, landmarks, inference_time = inference(model, processed_tensor.to(device), device)

    # Handle potential inference errors
    if loc is None:
        return np.empty((0, 5), dtype=np.float32), 0 # Return empty detections and 0 time on error

    # Decode outputs
    # Use the actual dimensions of the processed image for PriorBox
    priorbox = PriorBox(cfg, image_size=(img_width, img_height))
    priors = priorbox.generate_anchors().to(device)

    # Decode Boxes
    boxes = decode(loc.squeeze(0), priors, cfg['variance'])
    bbox_scale = torch.tensor([img_width, img_height, img_width, img_height], device=device) # Corrected scale order
    boxes = (boxes * bbox_scale).cpu().numpy()

    # Decode Scores (Confidence)
    scores = conf.squeeze(0).cpu().numpy()[:, 1] # Probability of 'face' class

    # Filter by Confidence Threshold
    inds = scores > params.conf_threshold
    boxes = boxes[inds]
    scores_filtered = scores[inds]

    # Sort by score and limit (Pre-NMS TopK)
    order = scores_filtered.argsort()[::-1][:params.pre_nms_topk]
    boxes = boxes[order]
    scores_filtered = scores_filtered[order]

    # Apply Non-Maximum Suppression (NMS)
    detections = np.empty((0, 5), dtype=np.float32) # Default empty
    if boxes.shape[0] > 0:
        # Ensure boxes and scores are on the correct device for tvops.nms
        boxes_tensor_nms = torch.from_numpy(boxes).to(device)
        scores_tensor_nms = torch.from_numpy(scores_filtered).to(device)

        keep = tvops.nms(boxes_tensor_nms, scores_tensor_nms, params.nms_threshold)
        keep = keep.cpu().numpy() # Get indices back to CPU

        # Combine boxes and scores for kept detections
        detections = np.hstack((boxes[keep], scores_filtered[keep, None])).astype(np.float32)

        # Limit detections (Post-NMS TopK)
        detections = detections[:params.post_nms_topk]

    # Return detections array [N, 5] (x1, y1, x2, y2, score) and inference time
    # Cast coordinates to int for drawing convenience later if needed, but keep score float
    # Return float coordinates for precision in evaluation, cast later if needed.
    return detections.astype(np.float32), inference_time


def convert_to_bbox(center_x, center_y, width, height):
    """ Converts center coords, width, height to x_min, y_min, x_max, y_max. """
    x_min = center_x - (width / 2)
    y_min = center_y - (height / 2)
    x_max = center_x + (width / 2)
    y_max = center_y + (height / 2)
    # Return as float for potential rescaling precision, cast to int later if needed
    return x_min, y_min, x_max, y_max

def extract_tilt_pan(text):
    """ Extracts tilt/pan integers from filename like '...[+-]tilt[+-]pan.jpg'. """
    pattern = r'.*?([+-]\d+)([+-]\d+)(?:\.jpg)?$'
    match = re.search(pattern, text)
    if match:
        tilt = int(match.group(1))
        pan = int(match.group(2))
        return tilt, pan
    else:
        # Return None or raise error if format is critical
        # print(f"Warning: No tilt/pan found in filename: {text}")
        return None, None
        # raise ValueError(f"No tilt and pan values found in filename: {text}")

def write_csv_results(results, output_csv, args, scale_value):
    """ Writes evaluation results per tilt/pan to a CSV file. """
    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = ["tilt", "pan", "total_images", "true_positives", "false_positives", "false_negatives",
                      "precision", "recall", "f1_score", "accuracy", "conf_threshold", "scale"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Sort keys (tilt, pan tuples) for consistent CSV output
        for key in sorted(results.keys()):
            stats = results[key]
            tp = stats["true_positives"]
            fp = stats["false_positives"]
            fn = stats["false_negatives"]
            total = stats["total_images"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            # Accuracy definition can vary; here using TP / (TP+FP+FN)
            accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

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
                "accuracy": f"{accuracy:.4f}",
                "conf_threshold": args.conf_threshold,
                "scale": scale_value # Use the passed scale value (e.g., "40" or "Original")
            })

# --- Main Processing Function ---
def convert_aflw_to_wider_detection(root_dir, output_root, args, model, device, cfg):
    """
    Processes images sequentially, optionally resizing without padding.
    Evaluates detections, saves ALL annotated images, and calculates avg inference time.
    Aggregates evaluation results per tilt and pan and outputs a CSV file.
    """
    total_images = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_inference_time = 0.0
    inference_count = 0
    iou_threshold = 0.1 # IoU threshold for TP/FP/FN calculation

    # Define the consistent output directory for images
    output_image_dir = os.path.join(output_root, "annotated_images")
    os.makedirs(output_image_dir, exist_ok=True)
    print(f"Saving annotated images to: {output_image_dir}")

    rgb_mean = np.array([104, 117, 123], dtype=np.float32) # Used for preprocessing

    print("Processing images sequentially...")
    persons = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d != "Front"]

    for person in tqdm(persons, desc="Processing Persons"):
        person_dir = os.path.join(root_dir, person)
        groundtruths = [f for f in os.listdir(person_dir) if f.endswith('.txt')]

        for groundtruth in tqdm(groundtruths, desc=f"Files in {person}", leave=False):
            mat_path = os.path.join(person_dir, groundtruth)
            base_name = os.path.splitext(groundtruth)[0]
            image_name = f"{base_name}.jpg"
            image_path = os.path.join(person_dir, image_name)

            # --- Load Image ---
            if not os.path.exists(image_path):
                continue
            img_original = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img_original is None:
                continue

            orig_h, orig_w = img_original.shape[:2]

            # --- Load Ground Truth ---
            try:
                with open(mat_path, "r") as f: lines = f.readlines()
                x_c, y_c, w, h = map(int, [lines[3], lines[4], lines[5], lines[6]])
                gt_x_min, gt_y_min, gt_x_max, gt_y_max = convert_to_bbox(x_c, y_c, w, h)
                gt_box_original = np.array([gt_x_min, gt_y_min, gt_x_max, gt_y_max])
            except Exception as e:
                print(f"Error reading GT {mat_path}: {e}, skipping.")
                continue

            # --- Extract Tilt/Pan ---
            tilt, pan = extract_tilt_pan(image_name)
            # Optionally skip if tilt/pan is required for eval_results grouping
            # if tilt is None or pan is None: continue

            # --- Image Preparation (Rescale or Use Original) ---
            image_to_process = img_original # Default to original
            gt_box_processed = gt_box_original # Default to original GT box

            if args.rescale:
                target_size = args.scaled_size
                # Resize image (simple resize, no padding)
                image_to_process = cv2.resize(img_original, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

                # Rescale GT box coordinates
                scale_x = target_size / orig_w
                scale_y = target_size / orig_h
                gt_x_min_new = gt_box_original[0] * scale_x
                gt_y_min_new = gt_box_original[1] * scale_y
                gt_x_max_new = gt_box_original[2] * scale_x
                gt_y_max_new = gt_box_original[3] * scale_y
                gt_box_processed = np.array([gt_x_min_new, gt_y_min_new, gt_x_max_new, gt_y_max_new])
                # print(f"Orig GT: {gt_box_original}, Scaled GT: {gt_box_processed.astype(int)}") # Debug print

            # --- Preprocessing for Model ---
            img_proc_float = np.float32(image_to_process) - rgb_mean
            img_proc_transposed = img_proc_float.transpose(2, 0, 1)  # HWC -> CHW
            processed_tensor = torch.from_numpy(img_proc_transposed).unsqueeze(0) # Add batch dim [1, C, H, W]

            # --- Perform Detection ---
            detections, inf_time = detectRetinaface(args, model, device, processed_tensor, cfg)
            # Detections are [N, 5] (x1, y1, x2, y2, score) relative to image_to_process dimensions

            if inf_time > 0: # Only count successful inferences
                 total_inference_time += inf_time
                 inference_count += 1

            # --- Evaluation Logic ---
            tp_increment = 0
            fp_increment = 0
            fn_increment = 0

            if detections.size == 0:
                fn_increment = 1 # FN: GT exists, no detections
            else:
                pred_boxes = detections[:, :4] # Get detection boxes [N, 4]
                # Calculate IoU between the (potentially scaled) GT box and all predicted boxes
                ious = compute_iou_vectorized(gt_box_processed, pred_boxes)

                # Check if *any* detection overlaps sufficiently with the GT box
                if np.any(ious >= iou_threshold):
                    tp_increment = 1 # TP: At least one detection matches the GT
                    # Count detections that *don't* match as FPs (IoU < threshold)
                    fp_increment = np.sum(ious < iou_threshold)
                else:
                    fn_increment = 1 # FN: GT exists, but no detection matched it
                    # All detections are FPs since none matched the GT
                    fp_increment = detections.shape[0]

            # Update overall totals
            total_tp += tp_increment
            total_fp += fp_increment
            total_fn += fn_increment

            # Update per tilt-pan results (only if tilt/pan were found)
            if tilt is not None and pan is not None:
                key = (tilt, pan)
                eval_results[key]["total_images"] += 1
                eval_results[key]["true_positives"] += tp_increment
                eval_results[key]["false_positives"] += fp_increment
                eval_results[key]["false_negatives"] += fn_increment

            total_images += 1 # Count processed image

            # --- Always Draw and Save Image ---
            # Draw on a copy of the image that was processed (original or resized)
            img_vis = image_to_process.copy()

            # Draw ground truth box (Blue) - Use processed GT coords, cast to int for drawing
            gt_coords_int = gt_box_processed.astype(int)
            cv2.rectangle(img_vis, (gt_coords_int[0], gt_coords_int[1]), (gt_coords_int[2], gt_coords_int[3]), (255, 0, 0), 2)

            # Draw detection boxes (Green) - Cast coords to int for drawing
            if detections.size > 0:
                for rec in detections:
                    # rec format: [x1, y1, x2, y2, score]
                    det_coords_int = rec[:4].astype(int)
                    cv2.rectangle(img_vis, (det_coords_int[0], det_coords_int[1]), (det_coords_int[2], det_coords_int[3]), (0, 255, 0), 1) # Use thinner line maybe
                    # Optional: Add confidence score text
                    # score = rec[4]
                    # cv2.putText(img_vis, f'{score:.2f}', (det_coords_int[0], det_coords_int[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Save the annotated image
            output_img_path = os.path.join(output_image_dir, image_name)
            cv2.imwrite(output_img_path, img_vis)
            # --- End Drawing and Saving ---


    # --- Final Reporting and CSV Writing ---
    print("-" * 30)
    print("Overall Evaluation Results:")
    print(f"Total images processed: {total_images}")
    print(f"True Positives (TP): {total_tp}")
    print(f"False Positives (FP): {total_fp}")
    print(f"False Negatives (FN): {total_fn}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0 # TP / (TP+FP+FN)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy (TP / All Detections + FN): {accuracy:.4f}")
    print(f"IoU Threshold for Matching: {iou_threshold}")
    print(f"Confidence Threshold for Detections: {args.conf_threshold}")

    # Report Average Inference Time
    avg_inference_time = total_inference_time / inference_count if inference_count > 0 else 0
    print(f"Average Inference Time per Image: {avg_inference_time:.6f} seconds ({inference_count} images measured)")

    # Determine scale value string for reporting and CSV
    scale_val_report = str(args.scaled_size) if args.rescale else "Original"
    print(f"Image Scale Used for Processing: {scale_val_report}")
    print("-" * 30)

    # Write the per tilt-pan results to CSV
    output_csv = os.path.join(output_root, "evaluation_results_per_tilt_pan.csv")
    write_csv_results(eval_results, output_csv, args, scale_value=scale_val_report)
    print(f"Per tilt-pan evaluation CSV saved to: {output_csv}")

# --- Main Execution Block ---
if __name__ == "__main__":
    args = parse_arguments()

    # Setup paths using args
    root_dir = args.image_dir
    output_root = args.output_dir

    # Create output directories if they don't exist
    os.makedirs(output_root, exist_ok=True)

    # Get model configuration
    cfg = get_config(args.network)
    if cfg is None:
        raise KeyError(f"Config file for network '{args.network}' not found!")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize and load the model
    print(f"Loading model weights from: {args.weights}")
    model = RetinaFace(cfg=cfg) # Ensure phase is 'test'
    model.to(device)
    model.eval() # Set model to evaluation mode

    # Make sure the weights file exists
    if not os.path.isfile(args.weights):
         raise FileNotFoundError(f"Weights file not found at {args.weights}")

    # Load state dict (handle potential 'module.' prefix)
    state_dict = torch.load(args.weights, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    for k, v in state_dict.items():
        name = k[7:] if has_module_prefix and k.startswith('module.') else k # remove `module.` prefix
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("Model loaded successfully!")

    # Optional: Convert model to TorchScript
    if args.jit:
        print("Attempting to convert model to TorchScript...")
        # Define dummy input size based on rescale option
        if args.rescale:
             dummy_input_size = (1, 3, args.scaled_size, args.scaled_size)
             print(f"Tracing with dummy input size: {dummy_input_size}")
        else:
             # Using a common size like 640x640 for tracing if not rescaling.
             # NOTE: Tracing with a fixed size might limit the model's ability
             # to handle variable input sizes if not originally designed for it.
             dummy_input_size = (1, 3, 640, 640) # Example default size
             print(f"Warning: JIT tracing without --rescale. Using dummy size {dummy_input_size}. Model must support this size.")

        dummy_input = torch.randn(dummy_input_size, device=device)
        try:
             model = torch.jit.trace(model, dummy_input)
             print("Model converted to TorchScript successfully!")
        except Exception as e:
             print(f"Failed to convert model to TorchScript: {e}")
             print("Proceeding with the original PyTorch model.")


    # Run main processing and evaluation function
    print("Starting dataset processing and evaluation...")
    convert_aflw_to_wider_detection(root_dir, output_root, args, model, device, cfg)

    # Final messages
    print("-" * 30)
    print(f"Processing complete.")
    print(f"Annotated images saved in: {os.path.join(output_root, 'annotated_images')}")
    print(f"Evaluation CSV saved in: {output_root}")
    print("-" * 30)