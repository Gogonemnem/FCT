# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified on Wednesday, September 28, 2022

@author: Guangxing Han
"""
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.data.catalog import MetadataCatalog
import detectron2.data.detection_utils as utils
global c
c = 0

__all__ = ["FsodRCNN"]


@META_ARCH_REGISTRY.register()
class FsodRCNN(GeneralizedRCNN):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
            self,
            *,
            support_way: int,
            support_shot: int,
            support_dir: str,
            data_dir: str,
            dataset: str,
            **kwargs,
            ):
        super().__init__(**kwargs)

        self.logger = logging.getLogger(__name__)

        self.support_way = support_way
        self.support_shot = support_shot

        self.support_dir = support_dir
        self.data_dir = data_dir
        self.dataset = dataset

        self.evaluation_dataset = 'voc'
        self.evaluation_shot = 10
        self.keepclasses = 'all1'
        self.test_seeds = 0

    @classmethod
    def from_config(cls, cfg):
        ret = super(FsodRCNN, cls).from_config(cfg)
        ret['support_way']  = cfg.INPUT.FS.SUPPORT_WAY
        ret['support_shot'] = cfg.INPUT.FS.SUPPORT_SHOT
        ret['support_dir']  = cfg.OUTPUT_DIR
        ret['data_dir']     = cfg.DATA_DIR
        ret['dataset']      = cfg.DATASETS.TRAIN[0]
        return ret

    def init_support_features(self, evaluation_dataset, evaluation_shot, keepclasses, test_seeds):
        self.evaluation_dataset = evaluation_dataset
        self.evaluation_shot = evaluation_shot
        self.keepclasses = keepclasses
        self.test_seeds = test_seeds

        if self.evaluation_dataset == 'voc':
            self.init_model_voc()
        elif self.evaluation_dataset == 'coco':
            self.init_model_coco()

    def init_model_voc(self):
        if 1:
            if self.test_seeds == 0:
                support_path = os.path.join(self.data_dir, 'pascal_voc/voc_2007_trainval_{}_{}shot.pkl'.format(self.keepclasses, self.evaluation_shot))
            elif self.test_seeds >= 0:
                support_path = os.path.join(self.data_dir, 'pascal_voc/seed{}/voc_2007_trainval_{}_{}shot.pkl'.format(self.test_seeds, self.keepclasses, self.evaluation_shot))

            support_df = pd.read_pickle(support_path)

            min_shot = self.evaluation_shot
            max_shot = self.evaluation_shot
            self.support_dict = {'image': {}, 'box': {}}
            for cls in support_df['category_id'].unique():
                support_cls_df = support_df.loc[support_df['category_id'] == cls, :].reset_index()
                support_data_all = []
                support_box_all = []

                for index, support_img_df in support_cls_df.iterrows():
                    img_path = os.path.join(self.data_dir, 'pascal_voc', support_img_df['file_path'])
                    support_data = utils.read_image(img_path, format='BGR')
                    support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                    support_data_all.append(support_data)

                    support_box = support_img_df['support_box']
                    support_box_all.append(Boxes([support_box]).to(self.device))

                min_shot = min(min_shot, len(support_box_all))
                max_shot = max(max_shot, len(support_box_all))
                # support images
                support_images = [x.to(self.device) for x in support_data_all]
                support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
                support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)
                self.support_dict['image'][cls] = support_images
                self.support_dict['box'][cls] = support_box_all

            print("min_shot={}, max_shot={}".format(min_shot, max_shot))


    def init_model_coco(self):
        if 1:
            if self.keepclasses == 'all':
                if self.test_seeds == 0:
                    support_path = os.path.join(self.data_dir, 'coco/full_class_{}_test_shot_support_df.pkl'.format(self.evaluation_shot))##(self.data_dir, 'coco/full_class_{}_shot_support_df.pkl'.format(self.evaluation_shot))
                elif self.test_seeds > 0:
                    support_path = os.path.join(self.data_dir, 'coco/seed{}/full_class_{}_shot_support_df.pkl'.format(self.test_seeds, self.evaluation_shot))
            else:
                if self.test_seeds == 0:
                    support_path = os.path.join(self.data_dir, 'coco/{}_shot_support_df.pkl'.format(self.evaluation_shot))
                elif self.test_seeds > 0:
                    support_path = os.path.join(self.data_dir, 'coco/seed{}/{}_shot_support_df.pkl'.format(self.test_seeds, self.evaluation_shot))

            support_df = pd.read_pickle(support_path)
            if 'coco' in self.dataset:
                metadata = MetadataCatalog.get('coco_2014_train')
            else:
                metadata = MetadataCatalog.get(self.dataset)##MetadataCatalog.get('coco_2014_train')  ##HACK
            # unmap the category mapping ids for COCO
            reverse_id_mapper = lambda dataset_id: metadata.thing_dataset_id_to_contiguous_id[dataset_id]  # noqa
            support_df['category_id'] = support_df['category_id'].map(reverse_id_mapper)

            min_shot = self.evaluation_shot
            max_shot = self.evaluation_shot
            self.support_dict = {'image': {}, 'box': {}}
            for cls in support_df['category_id'].unique():
                support_cls_df = support_df.loc[support_df['category_id'] == cls, :].reset_index()
                support_data_all = []
                support_box_all = []

                for index, support_img_df in support_cls_df.iterrows():
                    img_path = os.path.join(self.data_dir, 'coco', support_img_df['file_path'])
                    support_data = utils.read_image(img_path, format='BGR')
                    support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                    support_data_all.append(support_data)

                    support_box = support_img_df['support_box']
                    support_box_all.append(Boxes([support_box]).to(self.device))

                min_shot = min(min_shot, len(support_box_all))
                max_shot = max(max_shot, len(support_box_all))
                # support images
                support_images = [x.to(self.device) for x in support_data_all]
                support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
                support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)
                self.support_dict['image'][cls] = support_images
                self.support_dict['box'][cls] = support_box_all

            print("min_shot={}, max_shot={}".format(min_shot, max_shot))

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        
        images, support_images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            for x in batched_inputs:
                x['instances'].set('gt_classes', torch.full_like(x['instances'].get('gt_classes'), 0))
            
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        B, N, C, H, W = support_images.tensor.shape
        assert N == self.support_way * self.support_shot
        
        proposal_losses = None
        detector_losses = None

        for i in range(B):
            global c
            c += 1
            support_proposals_dict = {}
            support_features_dict = {}
            query_features_dict = {}
            support_boxes_dict = {}

            # query
            query_gt_instances = [gt_instances[i]] # one query gt instances
            query_images = ImageList.from_tensors([images[i]]) # one query image
            
            for way in range(self.support_way): ### note that only 2-way contrastive learning is supported, all negative classes are set to the same
                support_subset = support_images.tensor[i, way*self.support_shot:(way+1)*self.support_shot]
                query_features, support_features = self.backbone(images.tensor[i,:].unsqueeze(0), support_subset)

                support_boxes = batched_inputs[i]['support_bboxes'][way*self.support_shot:(way+1)*self.support_shot]
                support_boxes = [Boxes(box[np.newaxis, :]).to(self.device) for box in support_boxes]

                proposals, prop_losses = self.proposal_generator(query_images, query_features, support_features, support_boxes, query_gt_instances, way>0)

                if proposal_losses is None:
                    proposal_losses = prop_losses
                else:
                    # rpn loss, reduction is sum, thus taking the sum is justified (but apparently it is still the mean...)
                    proposal_losses = {k: v + prop_losses[k] for k, v in proposal_losses.items()}

                support_proposals_dict[way] = proposals
                support_features_dict[way] = support_features
                query_features_dict[way] = query_features
                support_boxes_dict[way] = support_boxes
                from detectron2.utils.events import EventStorage
                with EventStorage() as es: # dummy storage
                    from PIL import Image
                    
                    self.visualize_training(batched_inputs, proposals)
                    # Convert numpy array to PIL Image
                    image = Image.fromarray(np.transpose(es._vis_data[0][1], (1, 2, 0)))

                    # Save the image
                    image.save(f"training-image{c}-support{way}.jpg")

            _, det_losses = self.roi_heads(query_images, query_features_dict, support_proposals_dict, support_features_dict, support_boxes_dict, query_gt_instances) # rcnn
            
            if detector_losses is None:
                detector_losses = det_losses
            else:
                # detector loss, reduction is mean, thus taking the sum is justified if divided by B(atch size)
                detector_losses = {k: v + det_losses[k] for k, v in detector_losses.items()}
        
        proposal_losses = {k: v / (B * self.support_way) for k, v in proposal_losses.items()}
        detector_losses = {k: v / B for k, v in detector_losses.items()}
            
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy(),
                labels=prop.objectness_logits[0:box_size].sigmoid().cpu().numpy(),
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    @torch.no_grad()
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        
        images = self.preprocess_image(batched_inputs)

        B, _, _, _ = images.tensor.shape
        assert B == 1 # only support 1 query image in test
        assert len(images) == 1

        for i in range(B):
            support_proposals_dict = {}
            support_features_dict = {}
            query_features_dict = {}

            # query
            query_images = ImageList.from_tensors([images[i]]) # one query image
        
            for cls_id, support_images in self.support_dict['image'].items():
                from PIL import Image
                # Convert numpy array to PIL Image
                        # Convert tensor to numpy array and ensure the correct shape
                imageeee = support_images.tensor[1].cpu().numpy()
                
                # Assuming the tensor is in (C, H, W) format, transpose to (H, W, C)
                imageeee = np.transpose(imageeee, (1, 2, 0))

                # De-normalize the image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                imageeee = std * imageeee + mean
                imageeee = np.clip(imageeee, 0, 1)
                
                # Ensure the values are in the valid range for image data
                if imageeee.max() > 1.0:
                    imageeee = (imageeee * 255).astype(np.uint8)
                else:
                    imageeee = (imageeee * 255).astype(np.uint8)
                # Save the image
                image = Image.fromarray(imageeee)
                image.save(f"inf{cls_id}.jpg")
                query_features, support_features = self.backbone(query_images.tensor, support_images.tensor)

                support_boxes = self.support_dict['box'][cls_id]
                proposals, _ = self.proposal_generator(query_images, query_features, support_features, support_boxes, None)

                support_proposals_dict[cls_id] = proposals
                support_features_dict[cls_id] = support_features
                query_features_dict[cls_id] = query_features

        results, _ = self.roi_heads(query_images, query_features_dict, support_proposals_dict, support_features_dict, self.support_dict['box']) # rcnn
        if do_postprocess:
            return FsodRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = super().preprocess_image(batched_inputs)
        if not self.training:
            return images

        support_images = [self._move_to_current_device(x["support_images"]) for x in batched_inputs]
        support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
        support_images = ImageList.from_tensors(
            support_images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images, support_images
