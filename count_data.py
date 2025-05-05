import os
import time
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from config import get_config
from models import RetinaFace
from layers import PriorBox, MultiBoxLoss

from utils.dataset import WiderFaceDetection
from utils.transform import Augmentation


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Training Arguments for RetinaFace')
    parser.add_argument(
        '--train-data',
        type=str,
        default='./data/widerface/train',
        help='Path to the training dataset directory.'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='resnet34',
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network architecture to use'
    )
    parser.add_argument('--num-workers', default=8, type=int, help='Number of workers to use for data loading.')

    # Training arguments
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes in the dataset.')
    parser.add_argument('--batch-size', default=32, type=int, help='Number of samples in each batch during training.')
    parser.add_argument('--print-freq', type=int, default=10, help='Print frequency during training.')

    # Optimizer and scheduler arguments
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='Initial learning rate.')
    parser.add_argument('--lr-warmup-epochs', type=int, default=1, help='Number of warmup epochs.')
    parser.add_argument('--power', type=float, default=0.9, help='Power for learning rate policy.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum factor in SGD optimizer.')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay (L2 penalty) for the optimizer.')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD.')

    parser.add_argument(
        '--save-dir',
        default='./weights',
        type=str,
        help='Directory where trained model checkpoints will be saved.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint.ckpt from weights folder'
    )

    args = parser.parse_args()

    return args


rgb_mean = (104, 117, 123)  # bgr order


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    epoch,
    device,
    print_freq=10,
    scaler=None
) -> None:
    model.train()
    batch_loss = []
    for batch_idx, (images, targets) in enumerate(data_loader):
        start_time = time.time()
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            outputs = model(images)
            loss_loc, loss_conf, loss_land = criterion(outputs, targets)
            loss = cfg['loc_weight'] * loss_loc + loss_conf + loss_land

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Print training status
        if (batch_idx + 1) % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch: {epoch + 1}/{cfg['epochs']} | Batch: {batch_idx + 1}/{len(data_loader)} | "
                f"Loss Localization : {loss_loc.item():.4f} | Classification: {loss_conf.item():.4f} | "
                f"Landmarks: {loss_land.item():.4f} | "
                f"LR: {lr:.8f} | Time: {(time.time() - start_time):.4f} s"
            )
        batch_loss.append(loss.item())
    print(f"Average batch loss: {np.mean(batch_loss):.7f}")


def main(params):
    random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Create folder to save weights if not exists
    os.makedirs(params.save_dir, exist_ok=True)

    # --- Print training data count before and after augmentation ---
    # Create raw dataset without any augmentation transformation
    raw_dataset = WiderFaceDetection(params.train_data, transform=None)
    print(f"Number of training samples before augmentation: {len(raw_dataset)}")

    # Create dataset with augmentation applied
    dataset = WiderFaceDetection(params.train_data, Augmentation(cfg['image_size'], rgb_mean))
    print(f"Number of training samples after augmentation: {len(dataset)}")
    # --------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args.network)
    main(args)
