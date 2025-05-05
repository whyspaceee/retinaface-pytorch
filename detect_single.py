import argparse
import os
import cv2
import numpy as np
import torch
import torchvision.ops as tvops
import time
from config import get_config # Assuming config.py is in the same directory or accessible
from layers.functions.prior_box import PriorBox # Assuming layers/functions/prior_box.py exists
from models.retinaface import RetinaFace # Assuming models/retinaface.py exists
from utils.box_utils import decode # Assuming utils/box_utils.py exists

# --- Helper Functions (from previous script) ---

def convert_to_bbox(center_x, center_y, width, height):
    """Converts center coords, width, height to xmin, ymin, xmax, ymax."""
    x_min = center_x - (width / 2)
    y_min = center_y - (height / 2)
    x_max = center_x + (width / 2)
    y_max = center_y + (height / 2)
    return int(x_min), int(y_min), int(x_max), int(y_max)

@torch.no_grad()
def inference(model, image_tensor, device):
    """Runs single image inference and measures time."""
    model.eval()
    loc, conf, landmarks = None, None, None
    elapsed_time = 0.0

    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.amp.autocast("cuda"): # Use AMP
             loc, conf, landmarks = model(image_tensor)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0 # Time in seconds
    else: # CPU
        start_time = time.perf_counter()
        loc, conf, landmarks = model(image_tensor)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time # Time in seconds

    return loc, conf, landmarks, elapsed_time

def load_annotation(annotation_path):
    """Loads AFLW/HeadPose style annotation (x, y, w, h on specific lines)."""
    try:
        with open(annotation_path, "r") as f:
            lines = f.readlines()
        # Assuming lines[3-6] contain x, y, w, h relative to original image
        x = int(lines[3].strip())
        y = int(lines[4].strip())
        w = int(lines[5].strip())
        h = int(lines[6].strip())
        return x, y, w, h
    except Exception as e:
        print(f"Error reading annotation {annotation_path}: {e}")
        return None

# --- Main Script Logic ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="RetinaFace Single Image Detection")
    # Input/Output
    parser.add_argument('--image', required=True, type=str, help='Path to the input image file')
    parser.add_argument('--annotation', required=True, type=str, help='Path to the annotation file (.txt)')
    parser.add_argument('--output', required=True, type=str, help='Path to save the output image with boxes')
    # Model
    parser.add_argument('-w', '--weights', required=True, type=str, help='Path to trained model weights')
    parser.add_argument('--network', type=str, default='mobilenetv2',
                        choices=['mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
                                 'mobilenetv2', 'resnet50', 'resnet34', 'resnet18'],
                        help='Backbone network')
    # Preprocessing/Inference Options
    parser.add_argument('--rescale', action='store_true', help='Rescale image to 128x128 and pad to 640x640 canvas')
    parser.add_argument('--jit', action='store_true', help='Use TorchScript model (if available/converted)')
    # Detection Hyperparameters
    parser.add_argument('--conf-threshold', type=float, default=0.1, help='Confidence threshold for detection') # Adjusted default slightly
    parser.add_argument('--nms-threshold', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('--pre-nms-topk', type=int, default=5000, help='Keep top-k scores before NMS')
    parser.add_argument('--post-nms-topk', type=int, default=750, help='Keep top-k detections after NMS')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # --- Basic Setup ---
    cfg = get_config(args.network)
    if cfg is None:
        raise KeyError(f"Config file for {args.network} not found!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Initializing model {args.network}...")
    model = RetinaFace(cfg=cfg)

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found at {args.weights}")
        exit()

    print(f"Loading weights from {args.weights}...")
    try:
        state_dict = torch.load(args.weights, map_location=device, weights_only=True) # Safer loading
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}. Check file and network compatibility.")
        exit()

    model.to(device)
    model.eval()

    # --- Optional JIT ---
    if args.jit:
        print("Attempting to use TorchScript (tracing)...")
        # Define dummy input size based on whether rescale is used
        dummy_input_size = (640, 640) if args.rescale else (480, 640) # Example non-rescale size
        dummy_input = torch.randn(1, 3, dummy_input_size[1], dummy_input_size[0], device=device)
        try:
            model = torch.jit.trace(model, dummy_input)
            model.eval()
            print("Model successfully traced to TorchScript.")
        except Exception as e:
            print(f"Warning: TorchScript conversion failed: {e}. Running with standard PyTorch model.")
            # No need to change args.jit, just proceed with the original model

    # --- Load Image and Annotation ---
    if not os.path.exists(args.image):
        print(f"Error: Input image not found at {args.image}")
        exit()
    if not os.path.exists(args.annotation):
        print(f"Error: Annotation file not found at {args.annotation}")
        exit()

    original_img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if original_img is None:
        print(f"Error: Could not load image {args.image}")
        exit()
    orig_h, orig_w = original_img.shape[:2]

    gt_coords = load_annotation(args.annotation)
    if gt_coords is None:
        exit()
    gt_x, gt_y, gt_w, gt_h = gt_coords

    # --- Preprocessing and GT Box Transformation ---
    image_to_process = None
    gt_box_on_output = None # GT box coordinates relative to image_to_process
    prior_image_size = None # Image size used for generating anchor boxes
    output_scale_factors = (1.0, 1.0) # (scale_w, scale_h)

    if args.rescale:
        scaled_size = 48
        fixed_canvas_size = (640, 640) # (W, H)
        print(f"Rescaling image to {scaled_size}x{scaled_size} and padding to {fixed_canvas_size}...")

        resized_img = cv2.resize(original_img, (scaled_size, scaled_size))
        canvas = np.zeros((fixed_canvas_size[1], fixed_canvas_size[0], 3), dtype=original_img.dtype)
        offset_x = (fixed_canvas_size[0] - scaled_size) // 2
        offset_y = (fixed_canvas_size[1] - scaled_size) // 2
        canvas[offset_y:offset_y+scaled_size, offset_x:offset_x+scaled_size] = resized_img
        image_to_process = canvas
        prior_image_size = (fixed_canvas_size[0], fixed_canvas_size[1]) # Use canvas size for priors

        # Transform original GT coordinates to canvas coordinates
        scale_x = scaled_size / orig_w
        scale_y = scaled_size / orig_h
        x_new = gt_x * scale_x + offset_x
        y_new = gt_y * scale_y + offset_y
        w_new = gt_w * scale_x
        h_new = gt_h * scale_y
        x_min, y_min, x_max, y_max = convert_to_bbox(x_new, y_new, w_new, h_new)
        gt_box_on_output = [x_min, y_min, x_max, y_max]
        # Scale factors needed to map model output (relative to canvas) back to canvas coords
        output_scale_factors = (fixed_canvas_size[0], fixed_canvas_size[1]) # W, H

    else:
        print("Processing original image without rescaling...")
        image_to_process = original_img.copy() # Use original image
        prior_image_size = (orig_w, orig_h) # Use original size for priors
        # Convert original GT coordinates directly to bbox format
        x_min, y_min, x_max, y_max = convert_to_bbox(gt_x, gt_y, gt_w, gt_h)
        gt_box_on_output = [x_min, y_min, x_max, y_max]
        # Scale factors needed to map model output (relative to original) back to original coords
        output_scale_factors = (orig_w, orig_h) # W, H

    # --- Prepare Tensor for Inference ---
    img_h, img_w = image_to_process.shape[:2]
    rgb_mean = np.array([104, 117, 123], dtype=np.float32)
    image_tensor_input = np.float32(image_to_process) - rgb_mean
    image_tensor_input = image_tensor_input.transpose(2, 0, 1) # HWC -> CHW
    image_tensor_input = torch.from_numpy(image_tensor_input).unsqueeze(0) # Add batch dim
    image_tensor_input = image_tensor_input.to(device)

    # --- Run Inference ---
    print("Running inference...")
    loc, conf, landmarks, inference_time = inference(model, image_tensor_input, device)
    print(f"Inference time: {inference_time:.4f} seconds")

    # --- Post-processing ---
    print("Post-processing detections...")
    priorbox = PriorBox(cfg, image_size=(prior_image_size[1], prior_image_size[0])) # H, W for PriorBox
    priors = priorbox.generate_anchors().to(device)

    boxes = decode(loc.squeeze(0), priors, cfg['variance'])

    # Scale boxes to the output image dimensions (either original or canvas)
    bbox_scale_tensor = torch.tensor([output_scale_factors[0], output_scale_factors[1]] * 2, device=device) # W, H, W, H
    boxes = boxes * bbox_scale_tensor
    boxes = boxes.cpu().numpy()

    scores = conf.squeeze(0).cpu().numpy()[:, 1] # Class 1 is face score

    # Ignore low scores
    inds = np.where(scores > args.conf_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # Keep top-K before NMS
    order = scores.argsort()[::-1][:args.pre_nms_topk]
    boxes = boxes[order]
    scores = scores[order]

    # Apply NMS
    if boxes.shape[0] > 0:
        boxes_tensor = torch.from_numpy(boxes).to(device)
        scores_tensor = torch.from_numpy(scores).to(device)
        keep = tvops.nms(boxes_tensor, scores_tensor, args.nms_threshold)
        keep = keep.cpu().numpy()
        # Combine boxes and scores AFTER NMS
        detections = np.hstack((boxes[keep], scores[keep][:, np.newaxis])).astype(np.float32)
        # Keep top-K after NMS
        detections = detections[:args.post_nms_topk]
    else:
        detections = np.empty((0, 5), dtype=np.float32)

    print(f"Found {detections.shape[0]} faces.")

    # --- Visualize and Save ---
    output_image = image_to_process.copy() # Draw on the (potentially padded) image

    # Draw Ground Truth box (Red BGR)
    cv2.rectangle(output_image,
                  (gt_box_on_output[0], gt_box_on_output[1]),
                  (gt_box_on_output[2], gt_box_on_output[3]),
                  (0, 0, 255), 2)


    # Draw Detection boxes (Green BGR)
    for i in range(detections.shape[0]):
        box = detections[i, :4].astype(int)
        score = detections[i, 4]
        cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        text = f"{score:.2f}"
        cv2.putText(output_image, text, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the final image
    try:
        cv2.imwrite(args.output, output_image)
        print(f"Output image saved to: {args.output}")
    except Exception as e:
        print(f"Error saving output image: {e}")

    print("Processing finished.")