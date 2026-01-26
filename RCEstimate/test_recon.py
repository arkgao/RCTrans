"""
Test script for transparent object reconstruction using RCNet.

This script loads a trained model and performs inference on test images,
generating correspondence maps and visualizations.
"""

import argparse
import os
import torch
from os import path as osp

from dataset import build_dataloader, build_dataset
from models import build_model
from utils.options import parse
from utils.misc import set_random_seed


def parse_options(root_path, is_train=False):
    """Parse options from command line and config file.

    Args:
        root_path (str): Root path of the project.
        is_train (bool): Whether in training mode. Default: False.

    Returns:
        tuple: (opt, output_dir)
            - opt (dict): Configuration dictionary.
            - output_dir (str): Output directory for results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing test images (input_*.png, mask_*.png, background_*.png)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results (.flo files and visualizations)')
    parser.add_argument('--opt', type=str, required=True,
                       help='Path to test config YAML file.')

    args = parser.parse_args()
    opt = parse(args.opt, root_path, is_train=is_train)

    # Override input_folder from command line
    opt['datasets']['val']['input_folder'] = args.input_dir

    # Single GPU settings
    opt['dist'] = False
    opt['rank'] = 0
    opt['world_size'] = 1

    # Set random seed
    seed = opt.get('manual_seed', 10)
    opt['manual_seed'] = seed
    set_random_seed(seed)

    return opt, args.output_dir


def test_recon_pipeline(root_path):
    """Main testing pipeline for reconstruction.

    Args:
        root_path (str): Root path of the project.
    """
    # Parse options
    opt, output_dir = parse_options(root_path, is_train=False)

    # Set CUDA backend
    torch.backends.cudnn.benchmark = True

    # Build validation dataset and dataloader
    dataset_opt = opt['datasets']['val']
    test_dataset = build_dataset(dataset_opt)
    test_loader = build_dataloader(
        test_dataset,
        dataset_opt
    )

    print(f'Number of test images: {len(test_dataset)}')
    print(f'Loading model from: {opt["path"]["pretrain_network"]}')

    # Build model
    model = build_model(opt)

    # Run validation
    print(f'Running network testing, results will be saved to: {output_dir}')
    model.validation_for_recon(test_loader, output_dir)
    print('Testing completed!')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_recon_pipeline(root_path)
