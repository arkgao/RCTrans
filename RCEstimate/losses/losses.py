import torch
from torch import nn as nn

_reduction_modes = ['none', 'mean', 'sum']


class MaskP2DistanceLoss(nn.Module):
    """P2 distance Loss with mask support.

    Args:
        loss_weight (float): Loss weight for P2 distance loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', **kwagrs):
        super(MaskP2DistanceLoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, mask=None, **kwargs):
        """Forward function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            mask (Tensor, optional): of shape (N, 1, H, W). Mask tensor.
        """
        eps = 1e-12
        loss = torch.sqrt((pred - target).pow(2).sum(1, keepdims=True) + eps)

        if mask is not None:
            mask = mask.float()
            loss = loss * mask
        else:
            mask = torch.ones_like(loss)

        if self.reduction == 'mean':
            return self.loss_weight * loss.sum() / (mask.sum() + eps)
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        else:
            return self.loss_weight * loss
