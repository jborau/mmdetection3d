# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.losses import FocalLoss, SmoothL1Loss, binary_cross_entropy

from .axis_aligned_iou_loss import AxisAlignedIoULoss, axis_aligned_iou_loss
from .chamfer_distance import ChamferDistance, chamfer_distance
from .lovasz_loss import LovaszLoss
from .multibin_loss import MultiBinLoss
from .paconv_regularization_loss import PAConvRegularizationLoss
from .rotated_iou_loss import RotatedIoU3DLoss, rotated_iou_3d_loss
from .uncertain_smooth_l1_loss import UncertainL1Loss, UncertainSmoothL1Loss
from .centernet_gaussian_focal_loss import CenterNetGaussianFocalLoss
from .dim_aware_l1_loss import DimAwareL1Loss
from .uncertainty_loss import LaplacianAleatoricUncertaintyLoss, GaussianAleatoricUncertaintyLoss

__all__ = [
    'FocalLoss', 'SmoothL1Loss', 'binary_cross_entropy', 'ChamferDistance',
    'chamfer_distance', 'axis_aligned_iou_loss', 'AxisAlignedIoULoss',
    'PAConvRegularizationLoss', 'UncertainL1Loss', 'UncertainSmoothL1Loss',
    'MultiBinLoss', 'RotatedIoU3DLoss', 'rotated_iou_3d_loss', 'LovaszLoss', 
    'CenterNetGaussianFocalLoss', 'DimAwareL1Loss',
    'LaplacianAleatoricUncertaintyLoss', 'GaussianAleatoricUncertaintyLoss'
]
