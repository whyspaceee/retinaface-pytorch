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
        default='./weights/Resnet34_Final.pth',
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='mobilenetv2',
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network'
    )
    # Detection settings
    parser.add_argument('--conf-threshold', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--pre-nms-topk', type=int, default=5000, help='Max detections before NMS')
    parser.add_argument('--nms-threshold', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('--post-nms-topk', type=int, default=750, help='Max detections after NMS')
    # Output options
    parser.add_argument('-s', '--save-image', action='store_true', help='Save detection result images')
    parser.add_argument('-v', '--vis-threshold', type=float, default=0.6, help='Visualization threshold')
    # Image input
    parser.add_argument('--image-path', type=str, default='./assets/test.jpg', help='Input image path')
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
    interW = np.maximum(0, xB - xA + 1)
    interH = np.maximum(0, yB - yA + 1)
    interArea = interW * interH

    boxAArea = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    boxesArea = (pred_boxes[:, 2] - pred_boxes[:, 0] + 1) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1)
    iou = interArea / (boxAArea + boxesArea - interArea)
    return iou

@torch.no_grad()
def batched_inference(model, images_tensor, device):
    model.eval()
    if device.type == 'cuda':
        with torch.amp.autocast("cuda"):
            loc, conf, landmarks = model(images_tensor)
    else:
        loc, conf, landmarks = model(images_tensor)
    return loc, conf, landmarks

def batched_detectRetinaface(params, model, device, images, cfg, prior_cache):
    """
    Process a batch of images (numpy array of shape [B, H, W, 3]).
    Returns a list of detection arrays (one per image).
    """
    B, H, W, _ = images.shape
    rgb_mean = np.array([104, 117, 123], dtype=np.float32)
    images_processed = images.astype(np.float32) - rgb_mean  # vectorized subtraction
    images_tensor = torch.from_numpy(images_processed).permute(0, 3, 1, 2).to(device)
    
    loc, conf, landmarks = batched_inference(model, images_tensor, device)
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
            detections = np.hstack((boxes, scores_filtered[:, None])).astype(np.float32)[keep]
            detections = detections[:params.post_nms_topk]
        else:
            detections = np.empty((0, 5), dtype=np.float32)
        detections_list.append(detections.astype(int))
    return detections_list

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
        raise ValueError("No tilt and pan values found in filename.")

def load_image_and_annotation(person_dir, groundtruth, scaled_size, fixed_canvas_size):
    """
    Loads an image and its annotation, rescales it to scaled_size, and embeds it in a fixed canvas.
    Also transforms the ground-truth coordinates and extracts tilt/pan from the image name.
    """
    mat_path = os.path.join(person_dir, groundtruth)
    base_name = os.path.splitext(groundtruth)[0]
    image_name = f"{base_name}.jpg"
    try:
        with open(mat_path, "r") as f:
            lines = f.readlines()
        # Assuming lines[3-6] contain x, y, w, h
        x = int(lines[3])
        y = int(lines[4])
        w = int(lines[5])
        h = int(lines[6])
    except Exception as e:
        print(f"Error reading {mat_path}: {e}")
        return None
    img = cv2.imread(os.path.join(person_dir, image_name), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not load image: {image_name}")
        return None
    orig_h, orig_w = img.shape[:2]
    # Resize image to scaled_size (ignoring aspect ratio)
    resized_img = cv2.resize(img, (scaled_size, scaled_size))
    # Embed resized image in a fixed canvas (e.g., 640x640)
    canvas = np.zeros((fixed_canvas_size[1], fixed_canvas_size[0], 3), dtype=img.dtype)
    offset_x = (fixed_canvas_size[0] - scaled_size) // 2
    offset_y = (fixed_canvas_size[1] - scaled_size) // 2
    canvas[offset_y:offset_y+scaled_size, offset_x:offset_x+scaled_size] = resized_img
    # Transform ground-truth bounding box coordinates
    scale_x = scaled_size / orig_w
    scale_y = scaled_size / orig_h
    x_new = int(x * scale_x + offset_x)
    y_new = int(y * scale_y + offset_y)
    w_new = int(w * scale_x)
    h_new = int(h * scale_y)
    x_min, y_min, x_max, y_max = convert_to_bbox(x_new, y_new, w_new, h_new)
    gt_box = [x_min, y_min, x_max, y_max]
    
    # Attempt to extract tilt and pan from the image name.
    try:
        tilt, pan = extract_tilt_pan(image_name)
    except Exception as e:
        print(f"Error extracting tilt-pan from {image_name}: {e}")
        tilt, pan = None, None

    return {
        "image_name": image_name,
        "image": canvas,
        "gt_box": gt_box,
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
                "scale": scale_value
            })

def convert_aflw_to_wider_detection(root_dir, output_root, args, model, device, cfg):
    """
    Converts AFLW annotations to WIDER FACE detection format and evaluates detections.
    When --rescale is enabled, images are processed in batches.
    Also aggregates evaluation results per tilt and pan and outputs a CSV file.
    """
    total_images = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_threshold = 0.1

    # This value is used when rescaling
    scaled_size = 40

    if args.rescale:
        fixed_canvas_size = (640, 640)
        tasks = []
        # Gather image tasks from the dataset
        for person in tqdm(os.listdir(root_dir), desc="Persons"):
            if person == "Front":
                continue
            person_dir = os.path.join(root_dir, person)
            for groundtruth in tqdm(os.listdir(person_dir), desc=f"Processing {person}"):
                if not groundtruth.endswith('.txt'):
                    continue
                task = load_image_and_annotation(person_dir, groundtruth, scaled_size, fixed_canvas_size)
                if task is not None:
                    tasks.append(task)
        batch_size = args.batch_size
        # Precompute the anchor boxes (prior cache) for fixed canvas size
        priorbox = PriorBox(cfg, image_size=fixed_canvas_size)
        prior_cache = priorbox.generate_anchors().to(device)
        # Process tasks in batches
        for i in tqdm(range(0, len(tasks), batch_size), desc="Batches"):
            batch_tasks = tasks[i:i+batch_size]
            images_batch = np.stack([t["image"] for t in batch_tasks], axis=0)
            detections_list = batched_detectRetinaface(args, model, device, images_batch, cfg, prior_cache)
            for j, task in enumerate(batch_tasks):
                gt_box = np.array(task["gt_box"])
                detections = detections_list[j]
                if args.save_image:
                    img_vis = task["image"].copy()
                    cv2.rectangle(img_vis, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, 0), 2)
                    for rec in detections:
                        cv2.rectangle(img_vis, (rec[0], rec[1]), (rec[2], rec[3]), (0, 255, 0), 2)
                    output_dir = os.path.join(output_root, "0--AFLW")
                    os.makedirs(output_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(output_dir, task["image_name"]), img_vis)
                total_images += 1
                # Update per tilt-pan results if available
                tilt = task.get("tilt")
                pan = task.get("pan")
                if detections.size == 0:
                    total_fn += 1
                    if tilt is not None and pan is not None:
                        eval_results[(tilt, pan)]["total_images"] += 1
                        eval_results[(tilt, pan)]["false_negatives"] += 1
                else:
                    ious = compute_iou_vectorized(gt_box, detections[:, :4])
                    if np.any(ious >= iou_threshold):
                        total_tp += 1
                        false_pos = len(ious) - 1
                        total_fp += false_pos
                        if tilt is not None and pan is not None:
                            eval_results[(tilt, pan)]["total_images"] += 1
                            eval_results[(tilt, pan)]["true_positives"] += 1
                            eval_results[(tilt, pan)]["false_positives"] += (len(ious) - 1)
                    else:
                        total_fn += 1
                        false_pos = len(ious)
                        total_fn += 1
                        if tilt is not None and pan is not None:
                            eval_results[(tilt, pan)]["total_images"] += 1
                            eval_results[(tilt, pan)]["false_negatives"] += 1
                            eval_results[(tilt, pan)]["false_positives"] += len(ious)
    else:
        # Fallback: process images one by one (original behavior)
        for person in os.listdir(root_dir):
            if person == "Front":
                continue
            person_dir = os.path.join(root_dir, person)
            for groundtruth in os.listdir(person_dir):
                if not groundtruth.endswith('.txt'):
                    continue
                mat_path = os.path.join(person_dir, groundtruth)
                base_name = os.path.splitext(groundtruth)[0]
                image_name = f"{base_name}.jpg"
                try:
                    tilt, pan = extract_tilt_pan(image_name)
                except ValueError as e:
                    print(f"Skipping {image_name}: {e}")
                    continue
                image_path = os.path.join(person_dir, image_name)
                if not os.path.exists(image_path):
                    print(f"Skipping missing image: {image_path}")
                    continue
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Could not load image: {image_path}")
                    continue
                detections = detectRetinaface(args, model, device, img, cfg)
                with open(mat_path, "r") as f:
                    lines = f.readlines()
                # Assuming lines[3-6] contain x, y, w, h
                x = int(lines[3])
                y = int(lines[4])
                w = int(lines[5])
                h = int(lines[6])
                x_min, y_min, x_max, y_max = convert_to_bbox(x, y, w, h)
                gt_box = [x_min, y_min, x_max, y_max]
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                for rec in detections:
                    cv2.rectangle(img, (rec[0], rec[1]), (rec[2], rec[3]), (0, 255, 0), 2)
                total_images += 1
                # Update overall and per tilt-pan counts
                if tilt is not None and pan is not None:
                    eval_results[(tilt, pan)]["total_images"] += 1
                if detections.size == 0:
                    total_fn += 1
                    if tilt is not None and pan is not None:
                        eval_results[(tilt, pan)]["false_negatives"] += 1
                else:
                    pred_boxes = detections[:, :4]
                    ious = [compute_iou_vectorized(np.array(gt_box), np.array([pb]))[0] for pb in pred_boxes]
                    if any(iou_val >= iou_threshold for iou_val in ious):
                        total_tp += 1
                        total_fp += len(ious) - 1      # <-- Update overall false positives here
                        if tilt is not None and pan is not None:
                            eval_results[(tilt, pan)]["true_positives"] += 1
                            eval_results[(tilt, pan)]["false_positives"] += (len(ious) - 1)
                    else:
                        total_fn += 1
                        total_fp += len(ious)       # <-- Update overall false positives here
                        if tilt is not None and pan is not None:
                            eval_results[(tilt, pan)]["false_negatives"] += 1
                            eval_results[(tilt, pan)]["false_positives"] += len(ious)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
    print("Overall Evaluation results:")
    print(f"Total images processed: {total_images}")
    print(f"True Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"confidence threshold: {args.conf_threshold}")
    print(f"scale: {scaled_size}")

    # Write the per tilt-pan results to CSV
    output_csv = os.path.join(output_root, "evaluation_results.csv")
    write_csv_results(eval_results, output_csv, args, scale_value=scaled_size)
    print(f"Per tilt-pan evaluation CSV saved to: {output_csv}")

def detectRetinaface(params, model, device, original_image, cfg):
    rgb_mean = np.array([104, 117, 123], dtype=np.float32)
    image = np.float32(original_image) - rgb_mean
    img_height, img_width, _ = image.shape
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    loc, conf, landmarks = inference(model, image, device)
    priorbox = PriorBox(cfg, image_size=(img_width, img_height))
    priors = priorbox.generate_anchors().to(device)
    boxes = decode(loc.squeeze(0), priors, cfg['variance'])
    bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
    boxes = (boxes * bbox_scale).cpu().numpy()
    scores = conf.squeeze(0).cpu().numpy()[:, 1]
    inds = scores > params.conf_threshold
    boxes = boxes[inds]
    scores_filtered = scores[inds]
    order = scores_filtered.argsort()[::-1][:params.pre_nms_topk]
    boxes = boxes[order]
    scores_filtered = scores_filtered[order]
    if boxes.shape[0] > 0:
        boxes_tensor = torch.from_numpy(boxes).to(device)
        scores_tensor = torch.from_numpy(scores_filtered).to(device)
        keep = tvops.nms(boxes_tensor, scores_tensor, params.nms_threshold)
        keep = keep.cpu().numpy()
        detections = np.hstack((boxes, scores_filtered[:, None])).astype(np.float32)[keep]
        detections = detections[:params.post_nms_topk]
    else:
        detections = np.empty((0, 5), dtype=np.float32)
    return detections.astype(int)

@torch.no_grad()
def inference(model, image, device):
    model.eval()
    if device.type == 'cuda':
        with torch.cuda.amp.autocast():
            loc, conf, landmarks = model(image)
    else:
        loc, conf, landmarks = model(image)
    return loc, conf, landmarks

if __name__ == "__main__":
    # Configure paths
    root_dir = r"data/HeadPoseImageDatabase"
    output_root = r"output"
    
    args = parse_arguments()
    cfg = get_config(args.network)
    if cfg is None:
        raise KeyError(f"Config file for {args.network} not found!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize and load the model
    model = RetinaFace(cfg=cfg)
    model.to(device)
    model.eval()
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
    
    # Optional: Convert model to TorchScript for faster inference
    if args.jit:
        dummy_input = torch.randn(1, 3, 640, 640, device=device)
        model = torch.jit.trace(model, dummy_input)
        print("Model converted to TorchScript!")
    
    # Run conversion and evaluation on the AFLW dataset
    convert_aflw_to_wider_detection(root_dir, output_root, args, model, device, cfg)
    print(f"Detection files and images (if enabled) are saved in: {output_root}")
