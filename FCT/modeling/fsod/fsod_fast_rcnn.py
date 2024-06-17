# Copyright (c) Facebook, Inc. and its affiliates.
import logging
# from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, cross_entropy
# from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, _log_classification_stats, fast_rcnn_inference

__all__ = ["FsodFastRCNNOutputLayers"]


logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


class FsodFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        **kwargs,
    ):
        super().__init__(input_shape, **kwargs)
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
            
        input_size = input_shape.channels
        self.input_size = input_size
        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)

        ######################## few shot #########################
        self.patch_relation = True
        self.local_correlation = True
        self.global_relation = True

        if self.patch_relation:
            self.avgpool = nn.AvgPool2d(kernel_size=3,stride=1)
            self.conv_1 = nn.Conv2d(input_size*2, int(input_size/4), 1, padding=0)
            self.conv_2 = nn.Conv2d(int(input_size/4), int(input_size/4), 3, padding=0)
            self.conv_3 = nn.Conv2d(int(input_size/4), input_size, 1, padding=0)
            
            for l in [self.conv_1, self.conv_2, self.conv_3]:
                nn.init.constant_(l.bias, 0)
                nn.init.normal_(l.weight, std=0.01)
        
        if self.local_correlation:
            self.conv_cor = nn.Conv2d(input_size, input_size, 1, padding=0)
            
            for l in [self.conv_cor]:
                nn.init.constant_(l.bias, 0)
                nn.init.normal_(l.weight, std=0.01)

        if self.global_relation:
            self.avgpool_fc = nn.AvgPool2d(7)
            self.fc_1 = nn.Linear(input_size * 2, input_size)
            self.fc_2 = nn.Linear(input_size, input_size)

            for l in [self.fc_1, self.fc_2]:
                nn.init.constant_(l.bias, 0)
                nn.init.normal_(l.weight, std=0.01)

        
        ###########################################################

    @classmethod
    def from_config(cls, cfg, input_shape):
        args = super().from_config(cfg, input_shape)
        args["input_shape"] = ShapeSpec(channels=input_shape.channels)
        return args

    def forward(self, x_query, x_support):
        support = x_support #.mean(0, True) # avg pool on res4 or avg pool here?
        # fc
        if self.global_relation:
            x_query_fc = x_query.mean(dim=(2, 3))
            support_fc = x_support.mean(dim=(2, 3)).expand_as(x_query_fc)
            x_fc = torch.cat((x_query_fc, support_fc), 1)
            x_fc = F.relu(self.fc_1(x_fc), inplace=True)
            x_fc = F.relu(self.fc_2(x_fc), inplace=True)

        # correlation
        if self.local_correlation:
            x_query_cor = self.conv_cor(x_query)
            support_cor = self.conv_cor(support)
            x_cor = F.relu(F.conv2d(x_query_cor, support_cor.permute(1,0,2,3), groups=self.input_size), inplace=True).squeeze(3).squeeze(2)

        # relation
        if self.patch_relation:
            support_relation = support.expand_as(x_query)
            x_pr = torch.cat((x_query, support_relation), 1)
            x_pr = F.relu(self.conv_1(x_pr), inplace=True) # 5x5
            x_pr = self.avgpool(x_pr)
            x_pr = F.relu(self.conv_2(x_pr), inplace=True) # 3x3
            x_pr = F.relu(self.conv_3(x_pr), inplace=True) # 3x3
            x_pr = x_pr.mean(dim=(2, 3))

        return super().forward(x_fc + x_cor + x_pr)

    def losses(self, predictions, proposals):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes, prefix='fsod_fast_rcnn')

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        num_instances = gt_classes.numel()
        fg_inds = torch.where(gt_classes == 0)[0]
        bg_inds = torch.where(gt_classes == 1)[0]

        bg_scores = scores[bg_inds, :]
        sorted_bg_inds = torch.argsort(bg_scores[:, 0], descending=True)
        real_bg_inds = bg_inds[sorted_bg_inds]

        # bg_num_0 = max(1, min(fg_inds.shape[0] * 2, int(num_instances * 0.25))) #int(num_instances * 0.5 - fg_inds.shape[0])))
        bg_num_0 = max(1, min(fg_inds.shape[0] * 2, int(128 * 0.5))) #int(num_instances * 0.5 - fg_inds.shape[0])))
        bg_num_1 = max(1, min(fg_inds.shape[0] * 1, bg_num_0))

        #### THIS IS WRONG ####, the 128 is dependent on the gt-classes and not bg_inds, also premature sorting?
        real_bg_topk_inds_0 = real_bg_inds[real_bg_inds < 128][:bg_num_0]
        real_bg_topk_inds_1 = real_bg_inds[real_bg_inds >= 128][:bg_num_1]
        # real_bg_topk_inds_0 = real_bg_inds[real_bg_inds < int(num_instances * 0.5)][:bg_num_0]
        # real_bg_topk_inds_1 = real_bg_inds[real_bg_inds >= int(num_instances * 0.5)][:bg_num_1]

        topk_inds = torch.cat([fg_inds, real_bg_topk_inds_0, real_bg_topk_inds_1], dim=0)
        scores_filtered = scores[topk_inds]
        gt_classes_filtered = gt_classes[topk_inds]
        
        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores_filtered, gt_classes_filtered)
        else:
            loss_cls = cross_entropy(scores_filtered, gt_classes_filtered, reduction="mean")

        return {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }

    def inference(self, num_classes, predictions, proposals):
        """
        Args:
            num_classes (int): number of foreground classes.
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        scores = [torch.cat([score[:, :-1].reshape(num_classes, -1).permute(1, 0), torch.zeros(score.size(0) // num_classes, 1, device=score.device)], dim=1) for score in scores]
        boxes = [box.reshape(num_classes, -1, 4).permute(1, 0, 2).reshape(-1, num_classes*4) for box in boxes]

        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )