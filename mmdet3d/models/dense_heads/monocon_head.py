# Copyrigth:
from typing import List, Optional, Tuple

import numpy as np
import torch
from mmdet.models.utils import (gaussian_radius, gen_gaussian_target,
                                multi_apply)
from mmdet.models.utils.gaussian_target import (get_local_maximum,
                                                get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F
from torch import nn as nn

from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.utils import (ConfigType, InstanceList, OptConfigType,
                           OptInstanceList, OptMultiConfig)
from mmengine.model import bias_init_with_prob, normal_init
from .base_mono3d_dense_head import BaseMono3DDenseHead
from mmdet3d.structures.bbox_3d import rotation_3d_in_axis
from mmdet3d.structures import points_cam2img

INF = 1e8
EPS = 1e-12
PI = np.pi

@MODELS.register_module()
class MonoConHead(BaseMono3DDenseHead):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_classes,
                 bbox3d_code_size=7,
                 num_kpt=9,
                 num_alpha_bins=12,
                 max_objs=30,
                 vector_regression_level=1,
                 pred_bbox2d=True,
                 loss_center_heatmap=None,
                 loss_wh=None,
                 loss_offset=None,
                 loss_center2kpt_offset=None,
                 loss_kpt_heatmap=None,
                 loss_kpt_heatmap_offset=None,
                 loss_dim=None,
                 loss_depth=None,
                 loss_alpha_cls=None,
                 loss_alpha_reg=None,
                 use_AN=True,
                 num_AN_affine=10,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        ## We have to do super init of BaseMono3DHead as SMOKE does
        ## ASK ChatGPT
        super().__init__(init_cfg=init_cfg)
        assert bbox3d_code_size >= 7
        # self.in_channels = in_channels
        # self.feat_channel = feat_channel
        self.num_classes = num_classes
        self.bbox3d_code_size = bbox3d_code_size
        self.num_kpt = num_kpt
        self.num_alpha_bins = num_alpha_bins
        self.max_objs = max_objs
        self.vector_regression_level = vector_regression_level
        self.pred_bbox2d = pred_bbox2d

        # self.use_AN = use_AN
        # self.num_AN_affine = num_AN_affine
        # self.norm = AttnBatchNorm2d if use_AN else nn.BatchNorm2d
        self.use_AN = False
        self.num_AN_affine = num_AN_affine
        self.norm = nn.BatchNorm2d # Different to original

        """
            INITIALIZE LAYERS VIA FUNCTION OR CONVENTIONAL MONOCON WAY
        """
        # 3d bbox regression heads
        self.heatmap_head = self._build_head(in_channels, feat_channels, num_classes)
        self.offset_head = self._build_head(in_channels, feat_channels, 2)
        self.depth_head = self._build_head(in_channels, feat_channels, 2)
        self.dim_head = self._build_head(in_channels, feat_channels, 3)
        self._build_dir_head(in_channels, feat_channels)

        # Auxiliary Context regression head
        self.kpt_heatmap_head = self._build_head(in_channels, feat_channels, num_kpt)
        self.offset_kpt_head = self._build_head(in_channels, feat_channels, num_kpt * 2)
        self.wh_head = self._build_head(in_channels, feat_channels, 2)
        self.other_heads = self._build_head(in_channels, feat_channels, 2) ## Why only one head for 2 in the paper

        """
            INITIALIZE LOSSES VIA FUNCTION OR CONVENTIONAL MONONCON WAY
        """
        # 3dbbox heads
        self.loss_center_heatmap = MODELS.build(loss_center_heatmap)
        self.loss_offset = MODELS.build(loss_offset)
        self.loss_depth = MODELS.build(loss_depth)
        self.loss_dim = MODELS.build(loss_dim)
        self.loss_alpha_cls = MODELS.build(loss_alpha_cls)
        self.loss_alpha_reg = MODELS.build(loss_alpha_reg)
        if 'Aware' in loss_dim['type']:
            self.dim_aware_in_loss = True
        else:
            self.dim_aware_in_loss = False

        # Auxiliary heads
        self.loss_kpt_heatmap = MODELS.build(loss_kpt_heatmap)
        self.loss_kpt_heatmap_offset = MODELS.build(loss_kpt_heatmap_offset)
        self.loss_wh = MODELS.build(loss_wh)
        # 2heads in one
        self.loss_other_heads = MODELS.build(loss_center2kpt_offset)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channels, feat_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            self._get_norm_layer(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1))
        return layer
    
    def _build_dir_head(self, in_channels, feat_channels):
        self.dir_feat = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            self._get_norm_layer(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.dir_cls = nn.Sequential(nn.Conv2d(feat_channels, self.num_alpha_bins, kernel_size=1))
        self.dir_reg = nn.Sequential(nn.Conv2d(feat_channels, self.num_alpha_bins, kernel_size=1))

    #TODO: Adapt to use AN
    def _get_norm_layer(self, feat_channel):
        return self.norm(feat_channel, momentum=0.03, eps=0.001) if not self.use_AN else \
            self.norm(feat_channel, self.num_AN_affine, momentum=0.03, eps=0.001)

    #Different to anchor free
    def init_weights(self):
        bias_init = bias_init_with_prob(0.1)
        self.center_heatmap_head[-1].bias.data.fill_(bias_init)  # -2.19
        self.kpt_heatmap_head[-1].bias.data.fill_(bias_init)
        
        for head in [self.offset_head, self.depth_head, self.dim_head, self.dir_feat,
                     self.dir_cls, self.dir_reg, self.kpt_heatmap_head,
                     self.offset_kpt_head, self.wh_head, self.other_heads]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
 
    def forward(self, feats):
        return multi_apply(self.forward_single, feats)
    
    # TODO: review this
    def forward_single(self, feat):
        heatmap_pred = self.heatmap_head(feat).sigmoid()
        heatmap_pred = torch.clamp(heatmap_pred, min=1e-4, max=1 - 1e-4)
        kpt_heatmap_pred = self.kpt_heatmap_head(feat).sigmoid()
        kpt_heatmap_pred = torch.clamp(kpt_heatmap_pred, min=1e-4, max=1 - 1e-4)

        offset_pred = self.offset_head(feat)
        other_heads_pred = self.other_heads(feat)

        wh_pred = self.wh_head(feat)
        offset_kpt_pred = self.offset_kpt_head(feat)
        dim_pred = self.dim_head(feat)
        depth_pred = self.depth_head(feat)
        depth_pred[:, 0, :, :] = 1. / (depth_pred[:, 0, :, :].sigmoid() + EPS) - 1

        alpha_feat = self.dir_feat(feat)
        alpha_cls_pred = self.dir_cls(alpha_feat)
        alpha_offset_pred = self.dir_reg(alpha_feat)
        return heatmap_pred, offset_pred, depth_pred, dim_pred, alpha_cls_pred, alpha_offset_pred, \
                kpt_heatmap_pred, offset_kpt_pred, wh_pred, other_heads_pred

    def loss_by_feat(self,
            heatmap_pred, 
            offset_pred, 
            depth_pred, 
            dim_pred, 
            alpha_cls_pred, 
            alpha_offset_pred,                
            kpt_heatmap_pred, 
            offset_kpt_pred, 
            wh_pred, 
            other_heads_pred,
            batch_gt_instances_3d: InstanceList,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None):
        assert len(heatmap_pred) == len(offset_pred) == len(dim_pred) \
            == len(alpha_cls_pred) == len(alpha_offset_pred) == len(kpt_heatmap_pred) \
            == len(offset_kpt_pred) == len(wh_pred) == len(other_heads_pred) == 1
            ## WHY DEPTH NOT?
        
        heatmap_pred = heatmap_pred[0]
        offset_pred = offset_pred[0]
        depth_pred = depth_pred[0]
        dim_pred = dim_pred[0]
        alpha_cls_pred = alpha_cls_pred[0]
        alpha_offset_pred = alpha_offset_pred[0]
        kpt_heatmap_pred = kpt_heatmap_pred[0]
        offset_kpt_pred = offset_kpt_pred[0]
        wh_pred = wh_pred[0]
        other_heads_pred = other_heads_pred[0]

        batch_size = heatmap_pred.shape[0]

        # GET TARGETS
        target_result = self.get_targets(batch_gt_instances_3d,
                            batch_gt_instances,
                            heatmap_pred.shape, ## HELP
                            batch_img_metas)
    
    def get_targets(self, batch_gt_instances_3d: InstanceList,
                    batch_gt_instances: InstanceList, feat_shape: Tuple[int],
                    batch_img_metas: List[dict]) -> Tuple[Tensor, int, dict]:
        
        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]
        gt_labels = [
            gt_instances.labels for gt_instances in batch_gt_instances
        ]
        gt_bboxes_3d = [
            gt_instances_3d.bboxes_3d for gt_instances_3d in batch_gt_instances_3d
        ]
        centers_2d = [
            gt_instances_3d.centers_2d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        depths = [
            gt_instances_3d.depths
            for gt_instances_3d in batch_gt_instances_3d
        ]
        gt_kpts_2d = self.create_gt_kpts_2d(gt_bboxes_3d)
        
        gt_kpts_valid_mask = [

        ]

        img_shape = batch_img_metas[0]['pad_shape']

        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)  # 1/4
        height_ratio = float(feat_h / img_h)  # 1/4

        calibs = []

        # objects as 2D center points
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])

        # 2D attributes
        wh_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])
        offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])

        # 3D attributes
        dim_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 3])
        alpha_cls_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        alpha_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        depth_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])

        # 2D-3D kpt heatmap and offset
        center2kpt_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])
        kpt_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_kpt, feat_h, feat_w])
        kpt_heatmap_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])

        # indices
        indices = gt_bboxes[-1].new_zeros([bs, self.max_objs]).type(torch.cuda.LongTensor)
        indices_kpt = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt]).type(torch.cuda.LongTensor)

        # masks
        mask_target = gt_bboxes[-1].new_zeros([bs, self.max_objs])
        mask_center2kpt_offset = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])
        mask_kpt_heatmap_offset = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])

        for batch_id in range(bs):
            img_meta = batch_img_metas[batch_id]
            cam_p2 = img_meta['cam2img']

            gt_bbox = gt_bboxes[batch_id]
            calibs.append(cam_p2)
            if len(gt_bbox) < 1:
                continue
            gt_label = gt_labels[batch_id]
            gt_bbox_3d = gt_bboxes_3d[batch_id]
            if not isinstance(gt_bbox_3d, torch.Tensor):
                gt_bbox_3d = gt_bbox_3d.tensor.to(gt_bbox.device)

            depth = depths[batch_id]

            gt_kpt_2d = gt_kpts_2d[batch_id]
            gt_kpt_valid_mask = gt_kpts_valid_mask[batch_id]

            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            gt_kpt_2d = gt_kpt_2d.reshape(-1, self.num_kpt, 2)
            gt_kpt_2d[:, :, 0] *= width_ratio
            gt_kpt_2d[:, :, 1] *= height_ratio

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio

                dim = gt_bbox_3d[j][3: 6]
                alpha = gt_bbox_3d[j][6]
                gt_kpt_2d_single = gt_kpt_2d[j]  # (9, 2)
                gt_kpt_valid_mask_single = gt_kpt_valid_mask[j]  # (9,)

                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                indices[batch_id, j] = cty_int * feat_w + ctx_int

                wh_target[batch_id, j, 0] = scale_box_w
                wh_target[batch_id, j, 1] = scale_box_h
                offset_target[batch_id, j, 0] = ctx - ctx_int
                offset_target[batch_id, j, 1] = cty - cty_int

                dim_target[batch_id, j] = dim
                depth_target[batch_id, j] = depth[j]

                alpha_cls_target[batch_id, j], alpha_offset_target[batch_id, j] = self.angle2class(alpha)

                mask_target[batch_id, j] = 1

                for k in range(self.num_kpt):
                    kpt = gt_kpt_2d_single[k]
                    kptx_int, kpty_int = kpt.int()
                    kptx, kpty = kpt
                    vis_level = gt_kpt_valid_mask_single[k]
                    if vis_level < self.vector_regression_level:
                        continue

                    center2kpt_offset_target[batch_id, j, k * 2] = kptx - ctx_int
                    center2kpt_offset_target[batch_id, j, k * 2 + 1] = kpty - cty_int
                    mask_center2kpt_offset[batch_id, j, k * 2:k * 2 + 2] = 1

                    is_kpt_inside_image = (0 <= kptx_int < feat_w) and (0 <= kpty_int < feat_h)
                    if not is_kpt_inside_image:
                        continue

                    gen_gaussian_target(kpt_heatmap_target[batch_id, k],
                                        [kptx_int, kpty_int], radius)

                    kpt_index = kpty_int * feat_w + kptx_int
                    indices_kpt[batch_id, j, k] = kpt_index

                    kpt_heatmap_offset_target[batch_id, j, k * 2] = kptx - kptx_int
                    kpt_heatmap_offset_target[batch_id, j, k * 2 + 1] = kpty - kpty_int
                    mask_kpt_heatmap_offset[batch_id, j, k * 2:k * 2 + 2] = 1

        indices_kpt = indices_kpt.reshape(bs, -1)
        mask_target = mask_target.type(torch.bool)

        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            center2kpt_offset_target=center2kpt_offset_target,
            dim_target=dim_target,
            depth_target=depth_target,
            alpha_cls_target=alpha_cls_target,
            alpha_offset_target=alpha_offset_target,
            kpt_heatmap_target=kpt_heatmap_target,
            kpt_heatmap_offset_target=kpt_heatmap_offset_target,
            indices=indices,
            indices_kpt=indices_kpt,
            mask_target=mask_target,
            mask_center2kpt_offset=mask_center2kpt_offset,
            mask_kpt_heatmap_offset=mask_kpt_heatmap_offset,
        )

        return target_result
    """Correct this to pass cam2imgs and lidar2cams"""
    def create_gt_kpts_2d(self, gt_bboxes_3d, lidar2cam, cam2img):
        gt_kpts_2d = []
        for gt_bbox_3d in gt_bboxes_3d:
            corners = get_corners(gt_bbox_3d, lidar2cam)
            gt_kpts_2d.append(points_cam2img(corners, cam2img))
        return gt_kpts_2d

    def get_pitch(transformation_matrix):
        """
        Extracts the Euler angles from a 4x4 transformation matrix.
        
        Args:
        - transformation_matrix (numpy.array): 4x4 transformation matrix

        Returns:
        - roll, pitch, yaw (floats): Euler angles in radians
        """
        
        assert transformation_matrix.shape == (4, 4), "Matrix must be 4x4"
        
        # Extract 3x3 rotation matrix from the 4x4 transformation matrix
        R = transformation_matrix[:3, :3]
        
        # Pitch
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2)) + np.pi/2
        print(pitch)
        
        return pitch

    def get_corners(bbox, lidar2cam):
        """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
        x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

        .. code-block:: none

                            front z
                                /
                                /
                (x0, y0, z1) + -----------  + (x1, y0, z1)
                            /|            / |
                            / |           /  |
            (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                            |  /      .   |  /
                            | / origin    | /
            (x0, y1, z0) + ----------- + -------> right x
                            |             (x1, y1, z0)
                            |
                            v
                    down y

        Returns:
            Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
        """
        if bbox.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=bbox.tensor.device)

        dims = bbox.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin (0.5, 1, 0.5)
        corners_norm = corners_norm - dims.new_tensor([0.5, 1, 0.5])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        pitch = get_pitch(lidar2cam)

        corners = rotation_3d_in_axis(
            corners, bbox.tensor[:, 6], axis=1)
        corners = rotation_3d_in_axis(
            corners, pitch, axis=0)
        corners += bbox.tensor[:, :3].view(-1, 1, 3)

        # Compute the geometrical center
        center = torch.mean(corners, dim=1, keepdim=True)
        
        # Concatenate the center to the corners, resulting in a shape (N, 9, 3)
        corners_with_center = torch.cat([center, corners], dim=1)
        
        return corners_with_center


















    def predict(self):
        pass