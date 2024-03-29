#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday, April 14, 2022

@author: Guangxing Han
"""

from pycocotools.coco import COCO
import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import sys


COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Airplane "},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "Airport "},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "Baseball field "},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "Basketball court "},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "Bridge "},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "Chimney "},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "Dam "},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "Expressway service area "},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "Expressway toll station "},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "Golf course"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "Ground track field "},
    {"color": [220, 220, 0], "isthing": 1, "id": 12, "name": "Harbor "},
    {"color": [175, 116, 175], "isthing": 1, "id": 13, "name": "Overpass "},
    {"color": [250, 0, 30], "isthing": 1, "id": 14, "name": "Ship "},
    {"color": [165, 42, 42], "isthing": 1, "id": 15, "name": "Stadium "},
    {"color": [255, 77, 255], "isthing": 1, "id": 16, "name": "Storage tank "},
    {"color": [0, 226, 252], "isthing": 1, "id": 17, "name": "Tennis court "},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "Train station "},
    {"color": [0, 82, 0], "isthing": 1, "id": 19, "name": "Vehicle "},
    {"color": [120, 166, 157], "isthing": 1, "id": 20, "name": "Wind mill"},
]




split_dir = '/home/bibahaduri/DIOR/coco/datasets/cocosplit/seed1729'##'xxx/cocosplit' # please update the path in your system

for shot in [10]:##[1, 2, 3, 5, 10, 30]:
    fileids = {}
    for idx, cls_dict in enumerate(COCO_CATEGORIES):
        json_file = os.path.join(split_dir, 'full_box_{}shot_{}_trainval.json'.format(shot, cls_dict["name"]))
        print(json_file)

        coco_api = COCO(json_file)
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        fileids[idx] = list(zip(imgs, anns))

        with open(json_file,'r') as load_f:
            dataset = json.load(load_f)
            save_info = dataset['info']
            save_licenses = dataset['licenses']
            save_images = dataset['images']
            save_categories = dataset['categories']


    combined_imgs = []
    combined_anns = []
    vis_imgs = {}
    for _, fileids_ in fileids.items():
        dicts = []
        for (img_dict, anno_dict_list) in fileids_:
            if img_dict['id'] not in vis_imgs:
                combined_imgs.append(img_dict)
                vis_imgs[img_dict['id']] = True
            combined_anns.extend(anno_dict_list)

    dataset_split = {
        'info': save_info,
        'licenses': save_licenses,
        'images': combined_imgs,
        'annotations': combined_anns,
        'categories': save_categories
    }
    split_file = './new_annotations/full_class_{}_shot_instances_train2014.json'.format(shot)

    with open(split_file, 'w') as f:
        json.dump(dataset_split, f)

