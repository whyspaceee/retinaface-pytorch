import os
import argparse
import numpy as np

import torch

from models import RetinaFace
from config import cfg_mnet, cfg_mnet_025, cfg_mnet_050, cfg_mnet_v2, cfg_re50, cfg_re34, cfg_re18


def parse_arguments():
    parser = argparse.ArgumentParser(description='ONNX Export')

    parser.add_argument(
        '-w', '--weights',
        default='./weights/last.pth',
        type=str,
        help='Trained state_dict file path to open'
    )
    parser.add_argument(
        '-n', '--network',
        type=str,
        default='mobilenetv1',
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network architecture to use'
    )

    args = parser.parse_args()

    return args


def get_config(network):
    configs = {
        "mobilenetv1": cfg_mnet,
        "mobilenetv1_0.25": cfg_mnet_025,
        "mobilenetv1_0.50": cfg_mnet_050,
        "mobilenetv2": cfg_mnet_v2,
        "resnet50": cfg_re50,
        "resnet34": cfg_re34,
        "resnet18": cfg_re18
    }
    return configs.get(network, None)


@torch.no_grad()
def onnx_export(params):
    torch.set_grad_enabled(False)

    cfg = get_config(params.network)
    if cfg is None:
        raise KeyError(f"Config file for {params.network} not found!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model initialization
    model = RetinaFace(cfg=cfg)
    model.to(device)

    # loading state_dict
    state_dict = torch.load(params.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    # set to evaluation mode
    model.eval()

    fname = os.path.splitext(os.path.basename(args.weights))[0]
    onnx_model = f'{fname}.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(onnx_model))

    # create a dummy input with the same size which is used during training
    x = torch.randn(1, 3, 640, 640).to(device)

    # export to onnx
    torch_out = torch.onnx.export(
        model,
        x,
        onnx_model,
        export_params=True,
        verbose=False,
        input_names=['input'],
        output_names=['output']
    )


if __name__ == '__main__':
    args = parse_arguments()
    onnx_export(args)
