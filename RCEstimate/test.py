"""
Test script for evaluating RCNet on validation dataset.

This script loads a trained model and evaluates it on the validation set,
computing metrics (epe, px3, px5, px10) and saving visualizations.
"""

import torch
from os import path as osp

from dataset import build_dataloader, build_dataset
from models import build_model
from train import parse_options


def test_pipeline(root_path):
    """Main testing pipeline for validation.

    Args:
        root_path (str): Root path of the project.
    """
    # Parse options
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True

    # Build validation dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set,
            dataset_opt
        )
        print(f'Number of test images in {dataset_opt["name"]}: {len(test_set)}')
        test_loaders.append(test_loader)

    # Build model
    print(f'Loading model from: {opt["path"]["pretrain_network"]}')
    model = build_model(opt)

    # Run validation
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        print(f'Testing {test_set_name}...')
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val'].get('save_img', False)
        )

    print('Testing completed!')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_pipeline(root_path)
