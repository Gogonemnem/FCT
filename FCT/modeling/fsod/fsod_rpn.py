from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes, ImageList, Instances
@PROPOSAL_GENERATOR_REGISTRY.register()
class FsodRPN(RPN):
    @configurable
    def __init__(
            self,
            *,
            per_level_roi_poolers,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.per_level_roi_poolers = per_level_roi_poolers

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update(cls._init_per_level_poolers(cfg, input_shape))
        return ret
    
    @classmethod
    def _init_per_level_poolers(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.RPN.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        poolers = {
            f: ROIPooler(
                output_size=pooler_resolution,
                scales=(1.0 / input_shape[f].stride,),
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            ) for f in in_features
        }
        return {"per_level_roi_poolers": poolers}
    
    def forward(
        self,
        images: ImageList,
        query_features: Dict[str, torch.Tensor],
        support_features: Dict[str, torch.Tensor],
        support_boxes: List[Boxes],
        gt_instances: Optional[List[Instances]] = None,
    ):
        support_box_features = self.per_level_roi_pooling(support_features, support_boxes)
        support_box_features = {k: v.mean(dim=[0, 2, 3], keepdim=True) for k, v in support_box_features.items()}

        rpn_features = {key: F.conv2d(query_features[key], support_box_features[key].permute(1,0,2,3), groups=query_features[key].shape[1]) for key in support_box_features.keys()} # attention map for attention-style rpn
        return super().forward(images, rpn_features, gt_instances) # standard rpn

    def per_level_roi_pooling(self, features, boxes):
        box_features = {}
        for in_feature in self.in_features:
            level_features = [features[in_feature]]
            pooler = self.per_level_roi_poolers[in_feature]
            box_features[in_feature] = pooler(level_features, boxes)
        return box_features