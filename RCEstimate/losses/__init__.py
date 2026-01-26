from copy import deepcopy
from utils import get_root_logger
from .losses import MaskP2DistanceLoss

__all__ = ['MaskP2DistanceLoss', 'build_loss']


def build_loss(opt):
    """Build MaskP2DistanceLoss."""
    opt = deepcopy(opt)
    loss = MaskP2DistanceLoss(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [MaskP2DistanceLoss] is created.')
    return loss
