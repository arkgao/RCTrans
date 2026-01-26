from copy import deepcopy
from utils import get_root_logger

__all__ = ['build_model']


def build_model(opt):
    """Build RCNet model."""
    opt = deepcopy(opt)
    from models.rcnet_model import RCNet
    model = RCNet(opt)
    logger = get_root_logger()
    logger.info(f'Model [RCNet] is created.')
    return model
