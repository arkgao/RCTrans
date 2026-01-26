import torch
import numpy as np
import cv2
from utils.corres_utils import decode_corres


def read_list(path):
    return np.loadtxt(path, dtype=str).tolist()

def read_img(path, tensor_flag=True):
    img = cv2.imread(path)[:,:,::-1].astype(np.float32) / 255.0
    if tensor_flag:
        tensor = torch.from_numpy(img).permute(2,0,1)
        return tensor
    else:
        return img

def read_mask(path,tensor_flag = True):
    mask = cv2.imread(path).astype(np.float32) / 255.0
    if mask.ndim == 3:
        mask = mask[:,:,0:1]
    else:
        mask = mask.unsqueeze(-1)
    if tensor_flag:
        return torch.from_numpy(mask).permute(2,0,1)
    else:
        return mask

def read_corres(path):
    corres = decode_corres(path)
    corres = corres.clip(0,1)
    corres = corres.transpose(2,0,1)
    return torch.from_numpy(corres).float()
