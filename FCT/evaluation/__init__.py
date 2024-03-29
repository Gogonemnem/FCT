"""
Created on Wednesday, September 28, 2022

@author: Guangxing Han
"""
from .coco_evaluation import COCOEvaluator
from .pascal_voc_evaluation import PascalVOCDetectionEvaluator
from .dota_evaluation import DOTAEvaluator
from .dior_evaluation import DIOREvaluator
from .pascalcoco_evaluation import PASCALEvaluator

__all__ = [k for k in globals().keys() if not k.startswith("_")]
