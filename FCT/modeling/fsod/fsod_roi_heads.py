from typing import Dict, List, Optional, Tuple
import torch
from detectron2.config import configurable
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler

from .fsod_fast_rcnn import FsodFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class FsodStandardROIHeads(StandardROIHeads):
    """
    Standard ROIHeads with few-shot object detection (Fsod) functionality.
    """

    @configurable
    def __init__(
        self, 
        *,
        per_level_roi_poolers: Dict[str, ROIPooler],
        **kwargs):
        super().__init__(**kwargs)
        self.per_level_roi_poolers = per_level_roi_poolers

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        shape = ret['box_head'].output_shape
        ret["box_predictor"] = FsodFastRCNNOutputLayers(
            cfg, shape
        )
        ret.update(cls._init_per_level_poolers(cfg, input_shape))
        return ret
    
    @classmethod
    def _init_per_level_poolers(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
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

    def per_level_roi_pooling(self, features, boxes):
        box_features = {}
        for in_feature in self.in_features:
            level_features = [features[in_feature]]
            pooler = self.per_level_roi_poolers[in_feature]
            box_features[in_feature] = pooler(level_features, boxes)
        return box_features

    def forward(
        self,
        images: ImageList,
        query_features_dict: Dict[int, Dict[str, torch.Tensor]],
        support_proposals_dict: Dict[int, List[Instances]],
        support_features_dict: Dict[int, Dict[str, torch.Tensor]],
        support_boxes_dict: Dict[int, List[Boxes]],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            losses = self._forward_box(query_features_dict, support_proposals_dict, support_features_dict, support_boxes_dict, targets)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            # losses.update(self._forward_mask(features, proposals))
            # losses.update(self._forward_keypoint(features, proposals))
            # return proposals, losses
            return [], losses
        else:
            pred_instances = self._forward_box(query_features_dict, support_proposals_dict, support_features_dict, support_boxes_dict, targets)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(query_features_dict, pred_instances)
            return pred_instances, {}

    def _forward_box(self, query_features_dict, support_proposals_dict, support_features_dict, support_boxes_dict, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        cnt = 0
        full_proposals_ls = []
        full_scores_ls = []
        full_bboxes_ls = []
        num_classes = len(support_proposals_dict)

        for cls_id, proposals in support_proposals_dict.items():
            print(cls_id)
            if self.training:
                assert targets
                proposals = self.label_and_sample_proposals(proposals, targets)

            query_features = [query_features_dict[cls_id][f] for f in self.box_in_features]
            support_box_features = [support_features_dict[cls_id][f] for f in self.box_in_features]
            
            query_box_features = self.box_pooler(query_features, [x.proposal_boxes for x in proposals])
            support_box_features = self.box_pooler(support_box_features, support_boxes_dict[cls_id]).mean(0, True)

            query_box_features, support_box_features = self.box_head(query_box_features, support_box_features)
            class_logits, proposal_deltas = self.box_predictor(query_box_features, support_box_features)

            if self.training and cnt > 0:
                for item in proposals:
                    item.gt_classes = torch.full_like(item.gt_classes, 1)

            full_proposals_ls.extend(proposals)
            full_scores_ls.append(class_logits)
            full_bboxes_ls.append(proposal_deltas)

            cnt += 1
            del query_box_features
            del support_box_features
        del targets
        del support_features_dict
        del support_proposals_dict

        class_logits = torch.cat(full_scores_ls, dim=0)
        proposal_deltas = torch.cat(full_bboxes_ls, dim=0)
        proposals = [Instances.cat(full_proposals_ls)]

        predictions = class_logits, proposal_deltas
        
        if self.training:
            del query_features_dict
            losses = self.box_predictor.losses(predictions, proposals)
            
            #### TODO: Shamelessly copied from StandardROIHeads.forward, probably does not work
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            
            pred_instances, _ = self.box_predictor.inference(num_classes, predictions, proposals)
            return pred_instances