import numpy as np
import torch
import torchvision.transforms.functional as TF
from typing import Union
import flow_vis

"""
    We use the same way in optical flow to store and visualize our refractive correspondence.
"""

def decode_corres(path):
    # similar with the optical flow, we use the 'flo' format to save the correspondence
    # but we store the correspondence as np.float32 instead of np.int16
    TAG_FLOAT = 202021.25
    with open(path, 'rb') as file:
        flag = np.fromfile(file, dtype=np.float32, count=1)
        if flag != TAG_FLOAT:
            raise Exception('unable to read %s, maybe broken' % path)
        size = np.fromfile(file, dtype=np.int32, count=2)
        flo = np.fromfile(file, dtype=np.float32).reshape([size[1], size[0], 2])
    return flo


def corres2color(corres:Union[np.ndarray,torch.Tensor], mask=None):
    """
        color the correspondence like the way in optical flow
        color_wheel: the left up corner is red, the right up corner is purple, the left down corner is green, the right down corner is orange.
    Args:
        corres (Union[np.ndarray,torch.Tensor]): _description_
        mask (_type_, optional): _description_. Defaults to None.
    Returns:
        _type_: _description_
    """

    if torch.is_tensor(corres):
        if corres.dim() == 3:
            corres = corres.unsqueeze(0)
        corres = corres.permute(0,2,3,1).detach().cpu().numpy().astype(np.float32)
    else:
        if corres.ndim == 3:
            corres = corres[np.newaxis,...]

    if mask is not None:
        if torch.is_tensor(mask):
            if mask.dim() == 3:
                mask = mask.unsqueeze(0)
            mask = mask.permute(0,2,3,1).detach().cpu().numpy().astype(np.float32)
        else:
            if mask.ndim == 3:
                mask = mask[np.newaxis,...]

    corres = corres*2-1     # normalize the correspondence to [-1,1]
    img_batch = []
    for cor in corres:
        img = flow_vis.flow_to_color(cor,convert_to_bgr=False)
        img_batch.append(img)
    img = np.stack(img_batch,0)
    if mask is not None:
        img = img*mask

    return img
