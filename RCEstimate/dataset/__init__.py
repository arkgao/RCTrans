
import importlib
from torch.utils.data import DataLoader
from copy import deepcopy
from os import path as osp

from utils import scandir
from utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset', 'build_dataloader']

# Automatically scan and import dataset modules for registry
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
for module_name in dataset_filenames:
    importlib.import_module(f'dataset.{module_name}')


def build_dataset(dataset_opt):
    """Build dataset from options."""
    dataset_opt = deepcopy(dataset_opt)
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    return dataset


def build_dataloader(dataset, dataset_opt):
    """Build dataloader.
    Args:
        dataset: Dataset instance.
        dataset_opt (dict): Dataset options containing phase, batch_size, num_workers.
    """
    phase = dataset_opt['phase']

    if phase == 'train':
        batch_size = dataset_opt['batch_size']
        num_workers = dataset_opt['num_worker']
        shuffle = dataset_opt.get('use_shuffle', True)

        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=True)

    elif phase in ['val', 'test']:
        batch_size = dataset_opt.get('batch_size', 1)
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}.')

    return DataLoader(**dataloader_args)

