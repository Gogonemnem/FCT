# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified on Wednesday, September 28, 2022

@author: Guangxing Han
"""
import copy
import logging
import numpy as np
import torch
import os

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import pandas as pd
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.dataset_mapper import DatasetMapper

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithSupportCOCO"]


class DatasetMapperWithSupportCOCO(DatasetMapper):
    """
    Extends DatasetMapper to include support data for few-shot learning.
    """
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)  # Initialize the base class with the configuration
        # Additional initialization for support data
        self.support_way = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot = cfg.INPUT.FS.SUPPORT_SHOT
        self.seeds = cfg.DATASETS.SEEDS
        self.data_dir = cfg.DATA_DIR
        self.crop_gen = cfg.INPUT.CROP.ENABLED
        self.initialize_support_data(cfg)  # Initialize the support data based on the configuration

    def initialize_support_data(self, cfg):
        """
        Load support data based on the training configuration.
        """
        # Similar loading logic for support data as previously discussed
        if self.is_train:
            support_path = self.determine_support_path(cfg)
            self.support_df = pd.read_pickle(support_path)
            logging.getLogger(__name__).info(f"Loaded support data from: {support_path}")

            if 'coco' in cfg.DATASETS.TRAIN[0]:
                metadata = MetadataCatalog.get('coco_2014_train')
            else:
                metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            
            self.support_df['category_id'] = self.support_df['category_id'].map(
                lambda x: metadata.thing_dataset_id_to_contiguous_id[x]
            )

    def determine_support_path(self, cfg):
        """
        Determine the file path for the support data based on the configuration.
        """
        if cfg.INPUT.FS.FEW_SHOT:
            file_name = f"{'full_class_' if 'full' in cfg.DATASETS.TRAIN[0] else ''}{self.support_shot}_shot_support_df.pkl"
            if self.seeds:
                path = f"coco/seed{self.seeds}/{file_name}"
            else:
                path = f"coco/{file_name}"
        else:
            path = "coco/train_support_df.pkl"
        return os.path.join(self.data_dir, path)
    
    def __call__(self, dataset_dict):
        """
        Overrides the call method to customize image and annotation transformations.
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None
        
        # Apply image and annotation transforms
        annotations = dataset_dict.get("annotations")
        if self.crop_gen:
            if annotations:
                # If annotations exist, use them to guide the cropping
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(annotations)
                )
            else:
                # Apply random crop without guidance from annotations
                crop_tfm = self.crop_gen
            transforms.insert(0, crop_tfm)


        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).contiguous()  
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))          
        
        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )
        
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        support_images, support_bboxes, support_cls = self.generate_support(dataset_dict)
        dataset_dict.update({'support_images': support_images, 'support_bboxes': support_bboxes, 'support_cls': support_cls})
        
        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        return dataset_dict

    def generate_support(self, dataset_dict):
        support_way = self.support_way  # Number of different classes to include
        support_shot = self.support_shot  # Number of examples per class

        # Extract classes present in the query image
        query_classes = set([anno['category_id'] for anno in dataset_dict['annotations']])
        
        # Ensure at least one query class is included in the support set
        chosen_query_class = np.random.choice(list(query_classes))
        sampled_classes = [chosen_query_class]

        # Gather all possible classes excluding the (already chosen) query class(es)
        remaining_classes = list(set(self.support_df['category_id'].unique()) - query_classes) # set([chosen_query_class]))

        # Randomly select the rest of the classes to fill up the support set
        num_needed_classes = support_way - 1  # One less because we already included a query class
        if num_needed_classes > 0:
            if len(remaining_classes) + len(query_classes) - 1 < num_needed_classes:
                additional_classes = np.random.choice(list(query_classes) + remaining_classes, size=num_needed_classes, replace=True)
            else:
                additional_classes = np.random.choice(list(query_classes) + remaining_classes, size=num_needed_classes, replace=False)
            sampled_classes.extend(additional_classes)
            
        support_data_all = []
        support_box_all = []
        support_category_id = []

        for cls in sampled_classes:
            # Get the support examples for this class
            support_list = self.support_df[self.support_df['category_id'] == cls]
            if len(support_list) < support_shot:
                chosen_support = support_list.sample(n=support_shot, replace=True)
            else:
                chosen_support = support_list.sample(n=support_shot, replace=False)

            for _, support_item in chosen_support.iterrows():
                support_data = utils.read_image(os.path.join(self.data_dir, "coco", support_item["file_path"]), format=self.image_format)
                support_tensor = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                support_box = support_item['support_box']

                support_data_all.append(support_tensor)
                support_box_all.append(torch.tensor(support_box))
                support_category_id.append(cls)

        # Convert lists to tensors for processing
        support_data_all = torch.stack(support_data_all)
        support_box_all = torch.stack(support_box_all)
        return support_data_all, support_box_all, support_category_id




