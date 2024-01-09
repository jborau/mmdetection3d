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
class ArtemisHead(BaseMono3DDenseHead):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_classes,
                 bbox3d_code_size=7,
                 num_kpt=9,
                 num_alpha_bins=12,
                 max_objs=100,
                 vector_regression_level=1,
                 pred_bbox2d=True,
                 loss_center_heatmap=None,
                 loss_offset=None,
                 loss_dim=None,
                 loss_depth=None,
                 loss_alpha_cls=None,
                 loss_alpha_reg=None,
                 use_AN=True,
                 num_AN_affine=10,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert bbox3d_code_size >= 7

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
        self.bbox_code_size = 7 # Different to original

        """
            INITIALIZE LAYERS VIA FUNCTION OR CONVENTIONAL MONOCON WAY
        """
        # 3d bbox regression heads
        self.heatmap_head = self._build_head(in_channels, feat_channels, num_classes)
        self.offset_head = self._build_head(in_channels, feat_channels, 2)
        self.depth_head = self._build_head(in_channels, feat_channels, 2)
        self.dim_head = self._build_head(in_channels, feat_channels, 3)
        self._build_dir_head(in_channels, feat_channels)

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

    def _get_norm_layer(self, feat_channel):
        return self.norm(feat_channel, momentum=0.03, eps=0.001) if not self.use_AN else \
            self.norm(feat_channel, self.num_AN_affine, momentum=0.03, eps=0.001)

    def init_weights(self):
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)  # -2.19        
        for head in [self.offset_head, self.depth_head, self.dim_head, self.dir_feat,
                     self.dir_cls, self.dir_reg]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.1) # 0.001

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)
    
    def forward_single(self, feat):
        heatmap_pred = self.heatmap_head(feat).sigmoid()
        heatmap_pred = torch.clamp(heatmap_pred, min=1e-4, max=1 - 1e-4)

        offset_pred = self.offset_head(feat)

        dim_pred = self.dim_head(feat)

        depth_pred = self.depth_head(feat)
        depth_pred[:, 0, :, :] = 1. / (depth_pred[:, 0, :, :].sigmoid() + EPS) - 1

        alpha_feat = self.dir_feat(feat)
        alpha_cls_pred = self.dir_cls(alpha_feat)
        alpha_offset_pred = self.dir_reg(alpha_feat)

        return heatmap_pred, offset_pred, depth_pred, dim_pred, alpha_cls_pred, alpha_offset_pred
    
    def loss_by_feat(self,
            heatmap_pred, 
            offset_pred, 
            depth_pred, 
            dim_pred, 
            alpha_cls_pred, 
            alpha_offset_pred,
            batch_gt_instances_3d: InstanceList,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None):
        assert len(heatmap_pred) == len(offset_pred) == len(dim_pred) \
            == len(alpha_cls_pred) == len(alpha_offset_pred) == 1
        
        heatmap_pred = heatmap_pred[0]
        offset_pred = offset_pred[0]
        depth_pred = depth_pred[0]
        dim_pred = dim_pred[0]
        alpha_cls_pred = alpha_cls_pred[0]
        alpha_offset_pred = alpha_offset_pred[0]

        batch_size = heatmap_pred.shape[0]

        # GET TARGETS
        target_result = self.get_targets(batch_gt_instances_3d,
                            batch_gt_instances,
                            heatmap_pred.shape, ## HELP
                            batch_img_metas)
        
        center_heatmap_target = target_result['center_heatmap_target']
        offset_target = target_result['offset_target']
        depth_target = target_result['depth_target']
        dim_target = target_result['dim_target']
        alpha_cls_target = target_result['alpha_cls_target']
        alpha_offset_target = target_result['alpha_offset_target']

        indices = target_result['indices']

        mask_target = target_result['mask_target']

        # 2d offset
        offset_pred = self.extract_input_from_tensor(offset_pred, indices, mask_target)
        offset_target = offset_target[mask_target]

        # depth
        depth_pred = self.extract_input_from_tensor(depth_pred, indices, mask_target)
        depth_target = depth_target[mask_target]

        # dim
        dim_pred = self.extract_input_from_tensor(dim_pred, indices, mask_target)
        dim_target = dim_target[mask_target]

        alpha_cls_pred = self.extract_input_from_tensor(alpha_cls_pred, indices, mask_target)
        alpha_cls_target = alpha_cls_target[mask_target].type(torch.long)
        alpha_cls_onehot_target = alpha_cls_target.new_zeros([len(alpha_cls_target), self.num_alpha_bins]).scatter_(
            dim=1, index=alpha_cls_target.view(-1, 1), value=1)
        
        # alpha offset
        alpha_offset_pred = self.extract_input_from_tensor(alpha_offset_pred, indices, mask_target)
        alpha_offset_pred = torch.sum(alpha_offset_pred * alpha_cls_onehot_target, 1, keepdim=True)
        alpha_offset_target = alpha_offset_target[mask_target]

        # Calculate losses
        # Heatmap loss
        loss_center_heatmap = self.loss_center_heatmap(heatmap_pred, center_heatmap_target)

        loss_offset = self.loss_offset(offset_pred, offset_target)

        # Depth loss
        depth_pred, depth_log_variance = depth_pred[:, 0:1], depth_pred[:, 1:2]

        loss_depth = self.loss_depth(depth_pred, depth_target, depth_log_variance)

        # Alpha loss
        loss_alpha_cls = self.loss_alpha_cls(alpha_cls_pred, alpha_cls_onehot_target)
        loss_alpha_reg = self.loss_alpha_reg(alpha_offset_pred, alpha_offset_target)

        if self.dim_aware_in_loss:
            loss_dim = self.loss_dim(dim_pred, dim_target, dim_pred)
        else:
            loss_dim = self.loss_dim(dim_pred, dim_target)


        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_offset=loss_offset,
            loss_dim=loss_dim,
            loss_alpha_cls=loss_alpha_cls,
            loss_alpha_reg=loss_alpha_reg,
            loss_depth=loss_depth,
        )
    
    def get_targets(self, batch_gt_instances_3d: InstanceList,
                    batch_gt_instances: InstanceList, feat_shape: Tuple[int],
                    batch_img_metas: List[dict]) -> Tuple[Tensor, int, dict]:
        
        img_shape = batch_img_metas[0]['pad_shape']

        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)  # 1/4
        height_ratio = float(feat_h / img_h)  # 1/4
        
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
        calibs = []

        # objects as 2D center points
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])

        # 2D attributes
        offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])

        # 3D attributes
        dim_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 3])
        alpha_cls_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        alpha_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        depth_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])

        # indices
        indices = gt_bboxes[-1].new_zeros([bs, self.max_objs]).type(torch.cuda.LongTensor)

        # masks
        mask_target = gt_bboxes[-1].new_zeros([bs, self.max_objs])

        # TODO: review this
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

            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            # center_x = centers_2d[batch_id][:, [0]] * width_ratio
            # center_y = centers_2d[batch_id][:, [1]] * height_ratio
    
            gt_centers = torch.cat((center_x, center_y), dim=1)


            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio

                dim = gt_bbox_3d[j][3: 6]
                alpha = gt_bbox_3d[j][6]

                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                indices[batch_id, j] = cty_int * feat_w + ctx_int

                offset_target[batch_id, j, 0] = ctx - ctx_int
                offset_target[batch_id, j, 1] = cty - cty_int

                dim_target[batch_id, j] = dim
                depth_target[batch_id, j] = depth[j]

                alpha_cls_target[batch_id, j], alpha_offset_target[batch_id, j] = self.angle2class(alpha)

                mask_target[batch_id, j] = 1

        mask_target = mask_target.type(torch.bool)

        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            offset_target=offset_target,
            dim_target=dim_target,
            depth_target=depth_target,
            alpha_cls_target=alpha_cls_target,
            alpha_offset_target=alpha_offset_target,
            indices=indices,
            mask_target=mask_target,
        )

        return target_result

    @staticmethod
    def extract_input_from_tensor(input, ind, mask):
        input = transpose_and_gather_feat(input, ind)
        return input[mask]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class and residual. '''
        angle = angle % (2 * PI)
        assert (angle >= 0 and angle <= 2 * PI)
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * PI)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, cls, residual):
        ''' Inverse function to angle2class. '''
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        angle_center = cls * angle_per_class
        angle = angle_center + residual
        return angle


    def predict_by_feat(self,
            heatmap_pred, 
            offset_pred, 
            depth_pred, 
            dim_pred, 
            alpha_cls_pred, 
            alpha_offset_pred,                
            batch_img_metas: Optional[List[dict]] = None,
            rescale: bool = None) -> InstanceList:
    
        assert len(heatmap_pred) == len(offset_pred) == len(dim_pred) \
            == len(alpha_cls_pred) == len(alpha_offset_pred) == 1
        
        cam2imgs = torch.stack([
            heatmap_pred[0].new_tensor(img_meta['cam2img'])
            for img_meta in batch_img_metas
        ])
        trans_mats = torch.stack([
            heatmap_pred[0].new_tensor(img_meta['trans_mat'])
            for img_meta in batch_img_metas
        ])
        batch_det_bboxes_3d, batch_labels, batch_scores_heatmap = self.decode_heatmap(
            heatmap_pred[0],
            offset_pred[0],
            depth_pred[0],
            dim_pred[0],
            alpha_cls_pred[0],
            alpha_offset_pred[0],
            batch_img_metas,
            cam2imgs=cam2imgs,
            trans_mats=trans_mats,
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel,
            thresh=self.test_cfg.thresh)
        

        result_list = []
        for img_id in range(len(batch_img_metas)):
            bboxes = batch_det_bboxes_3d[img_id]
            scores = batch_scores_heatmap[img_id]
            labels = batch_labels[img_id]

            keep_idx = scores > 0.35
            bboxes = bboxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]

            bboxes = batch_img_metas[img_id]['box_type_3d'](
            bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5)) # 0.5
            attrs = None

            results = InstanceData()
            results.bboxes_3d = bboxes
            results.labels_3d = labels
            results.scores_3d = scores

            if attrs is not None:
                results.attr_labels = attrs
            result_list.append(results)

        return result_list
        
    def decode_heatmap(self,
                        heatmap_pred,
                        offset_pred,
                        depth_pred,
                        dim_pred,
                        alpha_cls_pred,
                        alpha_offset_pred,
                        batch_img_metas,
                        cam2imgs,
                        trans_mats,
                        k=100,
                        kernel=3,
                        thresh=0.1):
        
        
        img_h, img_w = batch_img_metas[0]['pad_shape'][:2]
        batch, cat, height, width = heatmap_pred.shape

        center_heatmap_pred = get_local_maximum(heatmap_pred, kernel=kernel)

        *batch_dets, ys, xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        # decode 3D prediction
        dim = transpose_and_gather_feat(dim_pred, batch_index)
        alpha_cls = transpose_and_gather_feat(alpha_cls_pred, batch_index)
        alpha_offset = transpose_and_gather_feat(alpha_offset_pred, batch_index)
        depth_pred = transpose_and_gather_feat(depth_pred, batch_index)
        depth = depth_pred[:, :, 0:1]

        sigma = depth_pred[:, :, 1]
        sigma = torch.exp(-sigma)
        # batch_scores *= sigma
        # batch_scores = batch_scores[..., -1]

        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = xs + offset[..., 0]
        topk_ys = ys + offset[..., 1]


        # TODO: Center2kpt is bad
        center2kpt_offset = transpose_and_gather_feat(offset_pred, batch_index)
        center2kpt_offset = center2kpt_offset.view(batch, k, 2)

        center2kpt_offset[..., ::2] = topk_xs.view(batch, k, 1).expand(batch, k, 1)
        center2kpt_offset[..., 1::2] = topk_ys.view(batch, k, 1).expand(batch, k, 1)

        kpts = center2kpt_offset

        kpts[..., ::2] *= (img_w / width)
        kpts[..., 1::2] *= (img_h / height)

        # 1. decode alpha
        alpha = self.decode_alpha_multibin(alpha_cls, alpha_offset)  # (b, k, 1)

        # 1.5 get projected center
        center2d = center2kpt_offset  # (b, k, 2)

        # 2. recover rotY
        rot_y = self.recover_rotation(kpts, alpha, cam2imgs)  # (b, k, 3)
        rot_y = alpha

        # 2.5 recover box3d_center from center2d and depth
        center3d = torch.cat([center2d, depth], dim=-1).squeeze(0)
        # print(center3d)
        center3d = self.pts2Dto3D(center3d, cam2imgs).unsqueeze(0)
        # print(center3d)
        # print(cam2imgs)

        # 3. compose 3D box
        batch_bboxes_3d = torch.cat([center3d, dim, rot_y], dim=-1)

        return batch_bboxes_3d, batch_topk_labels, batch_scores

    def recover_rotation(self, kpts, alpha, calib):
        device = kpts.device
        calib = calib.clone().detach()

        si = torch.zeros_like(kpts[:, :, 0:1]) + calib[:, 0:1, 0:1]
        rot_y = alpha + torch.atan2(kpts[:, :, 0:1] - calib[:, 0:1, 2:3], si)

        while (rot_y > PI).any():
            rot_y[rot_y > PI] = rot_y[rot_y > PI] - 2 * PI
        while (rot_y < -PI).any():
            rot_y[rot_y < -PI] = rot_y[rot_y < -PI] + 2 * PI

        return rot_y
    
    @staticmethod
    def pts2Dto3D(points, view):
        """
        Args:
            points (torch.Tensor): points in 2D images, [N, 3], \
                3 corresponds with x, y in the image and depth.
            view (np.ndarray): camera instrinsic, [3, 3]

        Returns:
            torch.Tensor: points in 3D space. [N, 3], \
                3 corresponds with x, y, z in 3D space.
        """
        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[1] == 3

        points2D = points[:, :2]
        depths = points[:, 2].view(-1, 1)
        # # Take the reciprocal of the depth values
        # reciprocal_depths = 1.0 / depths

        # # Now, scale the depths to the appropriate range
        # # For example, if the maximum expected depth is 100 meters:
        # scaled_depths = reciprocal_depths * 40

        # depths = scaled_depths
        # print(depths)
        unnorm_points2D = torch.cat([points2D * depths, depths], dim=1)

        viewpad = torch.eye(4, dtype=points2D.dtype, device=points2D.device)

        viewpad[:view.shape[1], :view.shape[2]] = view.clone().detach()
        inv_viewpad = torch.inverse(viewpad).transpose(0, 1)

        # Do operation in homogenous coordinates.
        nbr_points = unnorm_points2D.shape[0]
        homo_points2D = torch.cat(
            [unnorm_points2D,
             points2D.new_ones((nbr_points, 1))], dim=1)
        points3D = torch.mm(homo_points2D, inv_viewpad)[:, :3]

        return points3D

    def decode_alpha_multibin(self, alpha_cls, alpha_offset):
        alpha_score, cls = alpha_cls.max(dim=-1)
        cls = cls.unsqueeze(2)
        alpha_offset = alpha_offset.gather(2, cls)
        alpha = self.class2angle(cls, alpha_offset)

        alpha[alpha > PI] = alpha[alpha > PI] - 2 * PI
        alpha[alpha < -PI] = alpha[alpha < -PI] + 2 * PI
        return alpha