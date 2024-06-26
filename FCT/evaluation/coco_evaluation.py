# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified on Wednesday, September 28, 2022

@author: Guangxing Han
"""
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from detectron2.evaluation.evaluator import DatasetEvaluator


class BaseCOCOEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        # replace fewx with d2
        self._logger = logging.getLogger(__name__)
        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.info(
                f"'{dataset_name}' is not registered by `register_coco_instances`."
                " Therefore trying to convert it to COCO format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)

            # from detectron2.utils.events import EventStorage
            # with EventStorage() as es: # dummy storage
            #     from PIL import Image
            #     self.visualize_inference(inputs, outputs)
            #     # Convert numpy array to PIL Image
            #     image = Image.fromarray(np.transpose(es._vis_data[0][1], (1, 2, 0)))

            #     # Save the image
            #     import random
            #     image.save(f"evaluator{random.random()}.jpg")
            self._predictions.append(prediction)
    
    def visualize_inference(self, batched_inputs, proposals):
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
        from detectron2.data.detection_utils import convert_image_to_rgb
        from detectron2.utils.events import get_event_storage

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            # raise Exception(input)

            import torchvision.transforms as transforms
            img = convert_image_to_rgb(transforms.Resize((512, 512))(img.unsqueeze(0)).squeeze(0).permute(1, 2, 0), "BGR")
            
            v_gt = Visualizer(img, None)

            coco = self._coco_api
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=input["image_id"]))
            # annotations = [ann['bbox'] for ann in input['annotations']]

            category_ids = [ann['category_id'] for ann in annotations]
            boxes = [ann['bbox'] for ann in annotations]
            bbox_tensor = Boxes(torch.tensor(boxes))

            bbox_tensor = BoxMode.convert(bbox_tensor.tensor, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            v_gt = v_gt.overlay_instances(boxes=Boxes(bbox_tensor), labels=category_ids)
            anno_img = v_gt.get_image()
            box_size = min(len(prop['instances'].pred_boxes), max_vis_prop)

            box_size = min(len(prop['instances'].pred_boxes), max_vis_prop)
            pred_boxes = prop['instances'].pred_boxes[0:box_size].tensor.cpu().numpy()
            pred_classes = prop['instances'].pred_classes[0:box_size].cpu().numpy()
            pred_scores = prop['instances'].scores[0:box_size].cpu().numpy()

            # Combine classes and scores into labels
            labels = [f"{cls+1}: {score:.2f}" for cls, score in zip(pred_classes, pred_scores)]


            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=pred_boxes,
                labels=labels
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            # vis_img = np.concatenate((prop_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted boxes"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = self._format_predictions_for_coco(predictions)

        if self._output_dir:
            self._logger.info(f"Saving results to {self._output_dir}")
            self._save_coco_results(coco_results, "coco_instances_results.json")

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._evaluate_and_store_results(tasks, coco_results)

    def _format_predictions_for_coco(self, predictions):
        """
        Convert predictions to COCO format and handle category ID unmapping
        """
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # Unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = reverse_id_mapping.get(result["category_id"])
                if category_id is None:
                    raise ValueError(
                        f"A prediction has category_id={result['category_id']}, "
                        "which is not available in the dataset."
                    )
                result["category_id"] = category_id
        return coco_results

    def _save_coco_results(self, coco_results, file_name):
        """
        Save results in coco format to the given file name
        """
        file_path = os.path.join(self._output_dir, file_name)
        self._logger.info(f"Saving results to {file_path}")
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

    def _evaluate_and_store_results(self, tasks, coco_results):
        """
        Evaluate coco results and store the results for each task
        """
        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        if self._output_dir:
            self._logger.info("Saving generated box proposals to file.")
            self._save_box_proposals(predictions)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._evaluate_and_log_box_proposals(predictions)

    def _save_box_proposals(self, predictions):
        """
        Save generated box proposals to the file in the output directory.
        """
        ids, boxes, objectness_logits = self._extract_box_proposal_data(predictions)
        proposal_data = {
            "boxes": boxes,
            "objectness_logits": objectness_logits,
            "ids": ids,
            "bbox_mode": BoxMode.XYXY_ABS.value,
        }
        with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
            pickle.dump(proposal_data, f)

    def _extract_box_proposal_data(self, predictions):
        """
        Extract and return box proposal data from predictions
        """
        ids, boxes, objectness_logits = [], [], []
        for prediction in predictions:
            ids.append(prediction["image_id"])
            boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
            objectness_logits.append(prediction["proposals"].objectness_logits.numpy())
        return ids, boxes, objectness_logits

    def _evaluate_and_log_box_proposals(self, predictions):
        """
        Evaluate box proposals and log the metrics
        """
        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [5, 10, 25, 50, 75, 100, 200, 500, 1000, 1500, 2000, 5000, 10000, 20000]:
            for area, suffix in areas.items():
                stats = self._evaluate_box_proposals(predictions, self._coco_api, self._metadata.get("thing_classes"), self._metadata.get("thing_dataset_id_to_contiguous_id"), area=area, limit=limit)
                key = f"AR{suffix}@{limit:d}"
                res[key] = float(stats["ar"].item() * 100)
        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res

    # inspired from Detectron:
    # https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
    # evaluate proposals for novel classes
    def _evaluate_box_proposals(self, dataset_predictions, coco_api, class_names, mapper, thresholds=None, area="all", limit=None):
        """
        Evaluate detection proposal recall metrics. This function is a much
        faster alternative to the official COCO API recall evaluation code. However,
        it produces slightly different results.
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = {
            "all": 0,
            "small": 1,
            "medium": 2,
            "large": 3,
            "96-128": 4,
            "128-256": 5,
            "256-512": 6,
            "512-inf": 7,
        }
        area_ranges = [
            [0 ** 2, 1e5 ** 2],  # all
            [0 ** 2, 32 ** 2],  # small
            [32 ** 2, 96 ** 2],  # medium
            [96 ** 2, 1e5 ** 2],  # large
            [96 ** 2, 128 ** 2],  # 96-128
            [128 ** 2, 256 ** 2],  # 128-256
            [256 ** 2, 512 ** 2],  # 256-512
            [512 ** 2, 1e5 ** 2],
        ]  # 512-inf
        assert area in areas, "Unknown area range: {}".format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = []
        num_pos = 0

        for prediction_dict in dataset_predictions:
            predictions = prediction_dict["proposals"]

            # sort predictions in descending order
            # TODO maybe remove this and make it explicit in the documentation
            inds = predictions.objectness_logits.sort(descending=True)[1]
            predictions = predictions[inds]

            ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
            anno = coco_api.loadAnns(ann_ids)

            gt_boxes = [
                BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                for obj in anno
                if obj["iscrowd"] == 0 and class_names[mapper[obj["category_id"]]] in self.CLASS_NAMES
            ]
            gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
            gt_boxes = Boxes(gt_boxes)
            gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0 and class_names[mapper[obj["category_id"]]] in self.CLASS_NAMES])

            if len(gt_boxes) == 0 or len(predictions) == 0:
                continue

            valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
            gt_boxes = gt_boxes[valid_gt_inds]

            num_pos += len(gt_boxes)

            if len(gt_boxes) == 0:
                continue

            if limit is not None and len(predictions) > limit:
                predictions = predictions[:limit]

            overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

            _gt_overlaps = torch.zeros(len(gt_boxes))
            for j in range(min(len(predictions), len(gt_boxes))):
                # find which proposal box maximally covers each gt box
                # and get the iou amount of coverage for each gt box
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)

                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ovr, gt_ind = max_overlaps.max(dim=0)
                assert gt_ovr >= 0
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert _gt_overlaps[j] == gt_ovr
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1

            # append recorded iou coverage level
            gt_overlaps.append(_gt_overlaps)
        gt_overlaps = (
            torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
        )
        gt_overlaps, _ = torch.sort(gt_overlaps)

        if thresholds is None:
            step = 0.05
            thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
        recalls = torch.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {
            "ar": ar,
            "recalls": recalls,
            "thresholds": thresholds,
            "gt_overlaps": gt_overlaps,
            "num_pos": num_pos,
        }


    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }.get(iou_type, [])

        if not coco_eval:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        results = self._compute_standard_metrics(coco_eval, metrics)
        self._log_evaluation_results(iou_type, results)

        if class_names and len(class_names) > 1:
            results = self._compute_per_category_AP(coco_eval, class_names, results, iou_type)

        return results

    def _compute_standard_metrics(self, coco_eval, metrics):
        return {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }

    def _log_evaluation_results(self, iou_type, results):
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

    def _compute_per_category_AP(self, coco_eval, class_names, results, iou_type):
        precisions = coco_eval.eval["precision"]
        assert len(class_names) == precisions.shape[2]
        results_per_category = self._calculate_per_class_AP(precisions, class_names)

        for category, is_voc in [("VOC", True), ("Non VOC", False)]:
            ap_scores = self._calculate_summary_metrics(class_names, precisions, category, is_voc)
            num_classes = sum((name in self.CLASS_NAMES) == is_voc for name in class_names)

            for i, suffix in enumerate(["", "50", "75", "s", "m", "l"]):
                metric_name = f"{category.lower()}AP{suffix}"
                results[metric_name] = ap_scores[i]

                self._logger.info(f"Evaluation results for {category} {num_classes} categories ====> {metric_name}: " + str('%.2f' % ap_scores[i]))

        self._log_per_category_results(iou_type, results_per_category)
        results.update({"AP-" + name: ap for name, ap in results_per_category})

        return results

    def _log_per_category_results(self, iou_type, results_per_category):
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

    def _calculate_per_class_AP(self, precisions, class_names):
        """
        Calculate per-class AP.

        Args:
            precisions (np.array): Precision values for each class.
            class_names (list[str]): List of class names.

        Returns:
            List[float]: List of AP scores for each class.
        """
        class_aps = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else np.nan
            class_aps.append(("{}".format(name), float(ap * 100)))
        return class_aps
    
    def _calculate_summary_metrics(self, class_names, precisions, category_prefix, is_voc):
        """
        Calculate summary AP metrics for either VOC or non-VOC categories.

        Args:
            class_names (list[str]): List of class names.
            precisions (np.array): Precision values for each class.
            category_prefix (str): Prefix for the category (VOC or Non VOC).
            is_voc (bool): Whether the category is VOC or not.

        Returns:
            List[float]: List of AP scores for all, IOU .50, IOU .75, small, medium, large areas.
        """
        aps = []  # Stores AP scores for all conditions

        # Define area range indexes for small, medium, and large
        area_range_idxs = [(0, -1),  # all areas
                           (1, 0),   # T=0 for IOU .50
                           (5, 0),   # T=5 for IOU .75
                           (1, 1),   # A=1 for small
                           (1, 2),   # A=2 for medium
                           (1, 3)]   # A=3 for large

        category_classes = [i for i, name in enumerate(class_names) if (name in self.CLASS_NAMES) == is_voc]

        for T, A in area_range_idxs:
            class_aps = []
            for idx in category_classes:
                precision = precisions[T, :, idx, A, -1]
                
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else np.nan
                
                class_aps.append(ap)

            mean_ap = np.nanmean(class_aps) * 100 if class_aps else np.nan
            aps.append(mean_ap)

        return aps

class COCOEvaluator(BaseCOCOEvaluator):
    """
    Evaluator for COCO dataset based on the COCOEvaluator for handling COCO specific class names.
    """
    CLASS_NAMES = [
        "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person",
        "potted plant", "sheep", "couch", "train", "tv",
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DIOREvaluator(BaseCOCOEvaluator):
    """
    Evaluator for DIOR dataset based on the COCOEvaluator for handling DIOR specific class names.
    """
    CLASS_NAMES = [
        "Airplane ", "Baseball field ", "Tennis court ", "Train station ",
        "Wind mill",
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DOTAEvaluator(BaseCOCOEvaluator):
    """
    Evaluator for COCO dataset based on the COCOEvaluator for handling DOTA specific class names.
    """
    CLASS_NAMES = ["storage-tank", "tennis-court", "soccer-ball-field"]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PASCALEvaluator(BaseCOCOEvaluator):
    """
    Evaluator for Pascal dataset based on the COCOEvaluator for handling Pascal specific class names.
    """
    CLASS_NAMES = ['bird', 'bus', 'cow', 'motorbike', 'sofa']
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0, "No results to evaluate."

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    if iou_type == "keypoints":
        # Use the COCO default keypoint OKS sigmas unless overrides are specified
        if kpt_oks_sigmas:
            assert hasattr(coco_eval.params, "kpt_oks_sigmas"), "pycocotools is too old!"
            coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        # COCOAPI requires every detection and every gt to have keypoints, so
        # we just take the first entry from both
        num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
        num_keypoints_gt = len(next(iter(coco_gt.anns.values()))["keypoints"]) // 3
        num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
        assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (            
            f"[COCOEvaluator] Predictions contain {num_keypoints_dt} keypoints, "
            f"but ground truth contains {num_keypoints_gt} keypoints, "
            f"and the length of kpt_oks_sigmas is {num_keypoints_oks}. "
            "These counts must match. http://cocodataset.org/#keypoints-eval."
        )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
