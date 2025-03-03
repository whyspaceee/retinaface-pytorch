import argparse
import os
import cv2
import numpy as np
import torch
import re
import torchvision.ops as tvops
from tqdm import tqdm
from config import get_config
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landmarks


def parse_arguments():
    parser = argparse.ArgumentParser(description="Optimized RetinaFace Inference")
    # Model and device options
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights/Resnet34_Final.pth',
        help='Path to the trained model weights'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='mobilenetv2',
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network architecture to use'
    )
    # Detection settings
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for filtering detections'
    )
    parser.add_argument(
        '--pre-nms-topk',
        type=int,
        default=5000,
        help='Maximum number of detections to consider before applying NMS'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.4,
        help='Non-Maximum Suppression (NMS) threshold'
    )
    parser.add_argument(
        '--post-nms-topk',
        type=int,
        default=750,
        help='Number of highest scoring detections to keep after NMS'
    )
    # Output options
    parser.add_argument(
        '-s', '--save-image',
        action='store_true',
        help='Save the detection results as images'
    )
    parser.add_argument(
        '-v', '--vis-threshold',
        type=float,
        default=0.6,
        help='Visualization threshold for displaying detections'
    )
    # Image input
    parser.add_argument(
        '--image-path',
        type=str,
        default='./assets/test.jpg',
        help='Path to the input image'
    )
    # Rescale option: rescale image to scaled_size x scaled_size and embed in a 640x640 canvas
    parser.add_argument(
        '--rescale',
        action='store_true',
        help='Rescale input image to scaled_size x scaled_size and embed it in the center of a 640x640 image'
    )
    # Optional: enable TorchScript tracing (JIT) for the model.
    parser.add_argument(
        '--jit',
        action='store_true',
        help='Convert the model to TorchScript (via tracing) for potentially faster inference'
    )
    return parser.parse_args()


def compute_iou_vectorized(gt_box, pred_boxes):
    """
    Compute IoU between one ground truth box and multiple predicted boxes.
    Boxes are expected in [x_min, y_min, x_max, y_max] format.
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
def inference(model, image, device):
    model.eval()
    # Use mixed precision if on CUDA
    if device.type == 'cuda':
        with torch.cuda.amp.autocast():
            loc, conf, landmarks = model(image)
    else:
        loc, conf, landmarks = model(image)
    # Remove batch dimension
    return loc.squeeze(0), conf.squeeze(0), landmarks.squeeze(0)


def detectRetinaface(params, model, device, original_image, cfg, prior_cache=None):
    rgb_mean = (104, 117, 123)
    resize_factor = 1

    # Preprocess image: subtract mean, transpose, and convert to tensor.
    image = np.float32(original_image)
    img_height, img_width, _ = image.shape
    image -= rgb_mean
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    # Run inference with the model.
    loc, conf, landmarks = inference(model, image, device)

    # Use cached prior boxes if available; otherwise, generate new ones.
    if prior_cache is None:
        priorbox = PriorBox(cfg, image_size=(img_height, img_width))
        priors = priorbox.generate_anchors().to(device)
    else:
        priors = prior_cache

    # Decode boxes and landmarks.
    boxes = decode(loc, priors, cfg['variance'])
    landmarks = decode_landmarks(landmarks, priors, cfg['variance'])

    bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
    boxes = (boxes * bbox_scale / resize_factor).cpu().numpy()
    landmark_scale = torch.tensor([img_width, img_height] * 5, device=device)
    landmarks = (landmarks * landmark_scale / resize_factor).cpu().numpy()

    scores = conf.cpu().numpy()[:, 1]

    # Filter by confidence threshold.
    inds = scores > params.conf_threshold
    boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

    # Sort by score and select top-k before NMS.
    order = scores.argsort()[::-1][:params.pre_nms_topk]
    boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

    # Convert to tensors for torchvision NMS.
    boxes_tensor = torch.from_numpy(boxes).to(device)
    scores_tensor = torch.from_numpy(scores).to(device)
    # Use torchvision’s NMS (fast C++/CUDA implementation).
    keep = tvops.nms(boxes_tensor, scores_tensor, params.nms_threshold)
    keep = keep.cpu().numpy()

    detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)[keep]
    landmarks = landmarks[keep]

    # Keep only the top post-NMS detections.
    detections = detections[:params.post_nms_topk]
    landmarks = landmarks[:params.post_nms_topk]

    return detections.astype(int)


def convert_to_bbox(center_x, center_y, width, height):
    x_min = center_x - (width / 2)
    y_min = center_y - (height / 2)
    x_max = center_x + (width / 2)
    y_max = center_y + (height / 2)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def extract_tilt_pan(text):
    # Extract tilt and pan from the filename.
    pattern = r'.*?([+-]\d+)([+-]\d+)(?:\.jpg)?$'
    match = re.search(pattern, text)
    if match:
        tilt = int(match.group(1))
        pan = int(match.group(2))
        return tilt, pan
    else:
        raise ValueError("No tilt and pan values found in the input.")


def convert_aflw_to_wider_detection(root_dir, output_root, args, model, device, cfg):
    """
    Converts AFLW2000-3D annotations to WIDER FACE detection format and evaluates detections.
    The RetinaFace predictions are compared against a computed ground truth from the AFLW annotations.
    """
    event_name = "0--AFLW"
    output_dir = os.path.join(output_root, event_name)
    os.makedirs(output_dir, exist_ok=True)

    # Evaluation counters.
    total_images = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_threshold = 0.1
    scaled_size = 48

    # If rescaling is enabled, we will always embed images into a 640x640 canvas.
    prior_cache = None
    fixed_canvas_size = (640, 640)
    if args.rescale:
        # Create a dummy image to compute the anchors once.
        dummy = np.zeros((fixed_canvas_size[1], fixed_canvas_size[0], 3), dtype=np.float32)
        priorbox = PriorBox(cfg, image_size=fixed_canvas_size)
        prior_cache = priorbox.generate_anchors().to(device)

    # Loop over persons and their images.
    for person in tqdm(os.listdir(root_dir)):
        if person == "Front":
            continue
        person_dir = os.path.join(root_dir, person)
        for groundtruth in tqdm(os.listdir(person_dir)):
            if not groundtruth.endswith('.txt'):
                continue

            mat_path = os.path.join(person_dir, groundtruth)
            base_name = os.path.splitext(groundtruth)[0]
            image_name = f"{base_name}.jpg"
            image_path = os.path.join(person_dir, image_name)

            try:
                tilt, pan = extract_tilt_pan(image_name)
            except ValueError as e:
                print(f"Skipping {image_name}: {e}")
                continue

            if not os.path.exists(image_path):
                print(f"Skipping missing image: {image_path}")
                continue

            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Could not load image: {image_path}")
                continue

            orig_h, orig_w = img.shape[:2]
            # Rescale image if requested.
            if args.rescale:
                resized_img = cv2.resize(img, (scaled_size, scaled_size))
                canvas = np.zeros((fixed_canvas_size[1], fixed_canvas_size[0], 3), dtype=img.dtype)
                offset_x = (fixed_canvas_size[0] - scaled_size) // 2
                offset_y = (fixed_canvas_size[1] - scaled_size) // 2
                canvas[offset_y:offset_y+scaled_size, offset_x:offset_x+scaled_size] = resized_img
                img = canvas

            # Run RetinaFace detection.
            detections = detectRetinaface(args, model, device, img, cfg, prior_cache=prior_cache)
            # (Optionally, remove or comment out debug prints in a production run)
            # print(f"Detections for {image_name}: {detections}")

            # Read the annotation file.
            with open(mat_path, "r") as file:
                lines = file.readlines()
            # Assume the file structure provides the following entries.
            # lines[0]: image name, lines[1]: label, lines[3-6]: x, y, w, h
            _ = lines[0].strip()
            _ = lines[1].strip()
            x = int(lines[3])
            y = int(lines[4])
            w = int(lines[5])
            h = int(lines[6])

            # If rescaling, transform ground-truth coordinates.
            if args.rescale:
                offset_x = (fixed_canvas_size[0] - scaled_size) // 2
                offset_y = (fixed_canvas_size[1] - scaled_size) // 2
                scale_x = scaled_size / orig_w
                scale_y = scaled_size / orig_h
                x = int(x * scale_x + offset_x)
                y = int(y * scale_y + offset_y)
                w = int(w * scale_x)
                h = int(h * scale_y)

            x_min, y_min, x_max, y_max = convert_to_bbox(x, y, w, h)
            gt_box = [x_min, y_min, x_max, y_max]
            # (Optional: draw the ground truth box in blue)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # (Optional: draw detections in green)
            for rec in detections:
                cv2.rectangle(img, (rec[0], rec[1]), (rec[2], rec[3]), (0, 255, 0), 2)

            # Evaluation: compare detections to ground truth.
            total_images += 1
            if detections.size == 0:
                total_fn += 1
            else:
                pred_boxes = detections[:, :4]
                ious = compute_iou_vectorized(np.array(gt_box), pred_boxes)
                if np.any(ious >= iou_threshold):
                    total_tp += 1
                    total_fp += (len(ious) - 1)
                else:
                    total_fn += 1
                    total_fp += len(ious)

            # (Optional: save visualizations)
            if args.save_image:
                save_path = os.path.join(output_dir, image_name)
                cv2.imwrite(save_path, img)

    # Compute evaluation metrics.
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    print("Evaluation results:")
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

if __name__ == "__main__":
    # Configure paths.
    root_dir = r"data/HeadPoseImageDatabase"
    output_root = r"output"  # Adjust output root as desired.

    args = parse_arguments()
    cfg = get_config(args.network)
    if cfg is None:
        raise KeyError(f"Config file for {args.network} not found!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model.
    model = RetinaFace(cfg=cfg)
    model.to(device)
    model.eval()

    # Load checkpoint (make sure your checkpoint is compatible).
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    # Optional: Convert model to TorchScript for faster inference.
    if args.jit:
        dummy_input = torch.randn(1, 3, 640, 640, device=device)
        model = torch.jit.trace(model, dummy_input)
        print("Model converted to TorchScript!")

    # Run conversion and evaluation on the AFLW2000 dataset.
    convert_aflw_to_wider_detection(root_dir, output_root, args, model, device, cfg)
    print(f"Detection files (and images, if enabled) are saved in: {output_root}")
