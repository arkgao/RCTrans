from copy import deepcopy
from utils import get_root_logger

__all__ = ['build_network']


def build_network(opt):
    """Build RCNet network."""
    opt = deepcopy(opt)
    from archs.rcnet_arch import RCNet
    net = RCNet(**opt)
    logger = get_root_logger()
    logger.info(f'Network [RCNet] is created.')
    return net
