import torch
from torch import nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class LaplacianAleatoricUncertaintyLoss(nn.Module):
    """

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """
    def __init__(self, loss_weight=1.0, depth_focus = 1.0, variance_focus = 0.1):
        super(LaplacianAleatoricUncertaintyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.depth_focus_scale = depth_focus
        self.variance_scale = variance_focus

    def forward(self,
                input,
                target,
                log_variance):

        log_variance = log_variance.flatten()
        input = input.flatten()
        target = target.flatten()

        # Adjust the focus on depth prediction and log variance
        depth_error = self.depth_focus_scale * torch.abs(input - target)
        variance_error = log_variance

        # Calculate the modified loss
        loss = 1.4142 * torch.exp(-variance_error) * depth_error + variance_error

        return abs(loss.mean() * self.loss_weight)

        # log_variance = log_variance.flatten()
        # input = input.flatten()
        # target = target.flatten()

        # # Create a mask for non-zero targets
        # nonzero_mask = target != 0

        # # Apply the mask to the input and target
        # masked_input = input[nonzero_mask]
        # masked_target = target[nonzero_mask]
        # masked_log_variance = log_variance[nonzero_mask]

        # # Calculate the absolute error only for non-zero targets
        # abs_error = torch.abs(masked_input - masked_target)


        # loss = 1.4142 * torch.exp(-masked_log_variance) * abs_error + masked_log_variance

        # return loss.mean() * self.loss_weight


@MODELS.register_module()
class GaussianAleatoricUncertaintyLoss(nn.Module):
    """

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """
    def __init__(self, loss_weight=1.0):
        super(GaussianAleatoricUncertaintyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                input,
                target,
                log_variance):

        log_variance = log_variance.flatten()
        input = input.flatten()
        target = target.flatten()

        loss = 0.5 * torch.exp(-log_variance) * torch.abs(input - target)**2 + 0.5 * log_variance

        return loss.mean() * self.loss_weight
