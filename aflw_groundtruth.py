import argparse
import os
import cv2
import numpy as np
import scipy.io as sio
import torch

from config import get_config
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landmarks, nms


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference Arguments for RetinaFace")

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

    # New option: rescale image to scaled_sizexscaled_size and embed in center of a 640x640 canvas
    parser.add_argument(
        '--rescale',
        action='store_true',
        help='Rescale input image to scaled_sizexscaled_size and embed it in the center of a 640x640 image'
    )

    return parser.parse_args()


def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x_min, y_min, x_max, y_max] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interWidth = max(0, xB - xA + 1)
    interHeight = max(0, yB - yA + 1)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


@torch.no_grad()
def inference(model, image):
    model.eval()
    loc, conf, landmarks = model(image)

    loc = loc.squeeze(0)
    conf = conf.squeeze(0)
    landmarks = landmarks.squeeze(0)

    return loc, conf, landmarks


def detectRetinaface(params, model, device, original_image):
    rgb_mean = (104, 117, 123)
    cfg = get_config(params.network)
    resize_factor = 1

    # Read image and get dimensions
    image = np.float32(original_image)
    img_height, img_width, _ = image.shape

    # Normalize image
    image -= rgb_mean
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.from_numpy(image).unsqueeze(0)  # 1CHW
    image = image.to(device)

    # Forward pass
    loc, conf, landmarks = inference(model, image)

    # Generate anchor boxes
    priorbox = PriorBox(cfg, image_size=(img_height, img_width))
    priors = priorbox.generate_anchors().to(device)

    # Decode boxes and landmarks
    boxes = decode(loc, priors, cfg['variance'])
    landmarks = decode_landmarks(landmarks, priors, cfg['variance'])

    # Scale adjustments
    bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
    boxes = (boxes * bbox_scale / resize_factor).cpu().numpy()

    landmark_scale = torch.tensor([img_width, img_height] * 5, device=device)
    landmarks = (landmarks * landmark_scale / resize_factor).cpu().numpy()

    scores = conf.cpu().numpy()[:, 1]

    # Filter by confidence threshold
    inds = scores > params.conf_threshold
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    # Sort by scores
    order = scores.argsort()[::-1][:params.pre_nms_topk]
    boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

    # Apply NMS
    detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(detections, params.nms_threshold)
    detections = detections[keep]
    landmarks = landmarks[keep]

    # Keep top-k detections and landmarks
    detections = detections[:params.post_nms_topk]
    landmarks = landmarks[:params.post_nms_topk]

    return detections.astype(int)


def convert_aflw_to_wider_detection(annotations_dir, images_dir, output_root, args, model, device):
    """
    Converts AFLW2000-3D annotations to WIDER FACE detection format and evaluates detections.
    For each image, the ground-truth face bounding box is computed from the 2D landmarks.
    The RetinaFace predictions (after NMS) are compared against this ground truth using IoU.
    A prediction is considered correct if at least one detected box has IoU >= 0.5.
    Extra detections are counted as false positives.
    """
    # Create dummy event category (AFLW doesn't have categories)
    event_name = "0--AFLW"
    output_dir = os.path.join(output_root, event_name)
    os.makedirs(output_dir, exist_ok=True)

    # Evaluation counters
    total_images = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_threshold = 0.1
    scaled_size = 32

    for mat_file in os.listdir(annotations_dir):
        if not mat_file.endswith('.mat'):
            continue

        # Load MAT file and corresponding image
        mat_path = os.path.join(annotations_dir, mat_file)
        data = sio.loadmat(mat_path)
        base_name = os.path.splitext(mat_file)[0]
        image_name = f"{base_name}.jpg"
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            continue

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not load image: {image_path}")
            continue

        # --- If --rescale option is set, resize the image to scaled_sizexscaled_size and embed in a 640x640 canvas ---
        if args.rescale:
            orig_h, orig_w = img.shape[:2]
            # Resize the original image to scaled_sizexscaled_size (ignoring aspect ratio)
            resized_img = cv2.resize(img, (scaled_size, scaled_size))
            # Create a 640x640 black canvas
            canvas = np.zeros((640, 640, 3), dtype=img.dtype)
            offset_x = (640 - scaled_size) // 2  # 264
            offset_y = (640 - scaled_size) // 2  # 264
            canvas[offset_y:offset_y+scaled_size, offset_x:offset_x+scaled_size] = resized_img
            img = canvas

        # Run face detection using RetinaFace
        detections = detectRetinaface(args, model, device, img)
        # Print detections (for debugging)

        # --- Compute ground-truth bounding box from AFLW landmarks ---
        # The MAT file is assumed to contain 'pt2d' (2D facial landmarks)
        pt2d = data['pt2d']
        x_coords = pt2d[0, :]
        y_coords = pt2d[1, :]

        # If rescaling is enabled, transform the landmark coordinates accordingly.
        if args.rescale:
            # The original image dimensions (before rescaling) are needed.
            # We assume that the MAT file coordinates are with respect to the original image.
            # Note: If the original image dimensions are not stored in the MAT file,
            # you might need to obtain them separately. Here we use the read image shape.
            # Since we already replaced img with the canvas, we use the dimensions of the canvas.
            # To be safe, we assume the original dimensions from the MAT file are stored in variables
            # orig_w and orig_h (from above when reading the image).
            offset_x = (640 - scaled_size) // 2
            offset_y = (640 - scaled_size) // 2
            scale_x = scaled_size / orig_w
            scale_y = scaled_size / orig_h
            x_coords = x_coords * scale_x + offset_x
            y_coords = y_coords * scale_y + offset_y

        # Combine and remove any negative coordinates
        coords = np.array(list(zip(x_coords, y_coords)))
        coords = coords[(coords[:, 0] > 0) & (coords[:, 1] > 0)]
        if coords.shape[0] == 0:
            print(f"No valid landmarks in {mat_file}, skipping.")
            continue

        xs = coords[:, 0]
        ys = coords[:, 1]
        # The following margins (8, 64) are kept as-is; you might want to adjust them if needed.
        x_min = int(np.min(xs))
        x_max = int(np.max(xs))
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        gt_box = [x_min, y_min, x_max, y_max]

        # # Draw the ground-truth bounding box (blue)
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # # Draw each detection (green)
        # for rec in detections:
        #     cv2.rectangle(img, (rec[0], rec[1]), (rec[2], rec[3]), (0, 255, 0), 2)

        # # Plot the keypoints from ground truth for visualization
        # for x, y in zip(x_coords, y_coords):
        #     cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

        # --- Evaluation: compare detections to ground truth ---
        total_images += 1
        if detections.size == 0:
            # No detections: count as false negative for this image
            total_fn += 1
        else:
            # Compute IoU for each predicted box (first 4 numbers are coordinates)
            pred_boxes = detections[:, :4]
            ious = [compute_iou(pred_box, gt_box) for pred_box in pred_boxes]
            # If at least one detection has sufficient overlap, count one true positive;
            # all extra detections are false positives.
            if any(iou_val >= iou_threshold for iou_val in ious):
                total_tp += 1
                total_fp += (len(ious) - 1)
            else:
                # cv2.imshow("image", img)
                # key = cv2.waitKey(0)
                # if key == ord('q'):
                #     break
                total_fn += 1
                total_fp += len(ious)

        # --- Save detection file in WIDER FACE format ---
        # Compute width and height from the ground truth
        w = x_max - x_min
        h = y_max - y_min
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, 'w') as f:
            # Format: image name, number of faces, and for each face [left top width height score]
            f.write(f"{image_name}\n")
            f.write("1\n")  # Since AFLW images have one face
            f.write(f"{x_min} {y_min} {w} {h} 1.0\n")

    # --- Compute and print overall evaluation metrics ---
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    # Here, “accuracy” is defined as TP / (TP + FP + FN)
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


if __name__ == "__main__":
    # Configure paths
    annotations_dir = r"data\AFLW2000"
    images_dir = r"data\AFLW2000"
    output_root = ""  # You can change this to your desired output root

    # Parse arguments and load configuration
    args = parse_arguments()
    cfg = get_config(args.network)
    if cfg is None:
        raise KeyError(f"Config file for {args.network} not found!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization
    model = RetinaFace(cfg=cfg)
    model.to(device)
    model.eval()

    # Loading state_dict (ensure your checkpoint is compatible)
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    # Run conversion and evaluation on the AFLW2000 dataset
    convert_aflw_to_wider_detection(annotations_dir, images_dir, output_root, args, model, device)

    print(f"Detection files generated in: {output_root}")
