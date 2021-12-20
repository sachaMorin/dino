import os
import argparse
import torch
import yaml

# detectron2 utils
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor

from dino.lib.data_preparation import get_duckietown_dicts
from dino.lib.visualization import visualize_predictions
# Evaluation imports
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


def main(params):
    cfg = get_cfg()
    cfg.merge_from_file(params.config_file)
    cfg.MODEL.WEIGHTS = params.model_weights  # path to the model we just trained
    cfg.OUTPUT_DIR = params.output_dir
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = params.iou_threshold  # set a custom testing threshold
    # Duckietown classes
    class_list = ['cone', 'duckie', 'duckiebot']
    # Registering dataset
    for d in ["train", "test"]:
        DatasetCatalog.register("duckietown_" + d,
                                lambda d=d: get_duckietown_dicts(os.path.join(params.dataset_path, d)))
        MetadataCatalog.get('duckietown_' + d).set(thing_classes=class_list)
    MetadataCatalog.get('duckietown_test').set(thing_classes=class_list)
    duckietown_metadata = MetadataCatalog.get('duckietown_test')
    dataset_dicts = get_duckietown_dicts(os.path.join(params.dataset_path, 'test'))
    # Pass the validation dataset
    cfg.DATASETS.TEST = ("duckietown_test",)
    predictor = DefaultPredictor(cfg)
    visualize_predictions(dataset_dicts, predictor, duckietown_metadata, 5)
    # Evaluation
    evaluator = COCOEvaluator("duckietown_test", cfg, False, output_dir=params.output_dir)
    val_loader = build_detection_test_loader(cfg, "duckietown_test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == '__main__':
    # Receive arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='data/dt',
                        type=str, help="Path of the dataset.")
    parser.add_argument("--config_file", default='models/checkpoints/baseline/cfg.yaml',
                        type=str, help="Path to configuration file")
    parser.add_argument("--model_weights", default='models/checkpoints/baseline/model_final.pth',
                        type=str, help="Path to model weights.")
    parser.add_argument("--output_dir", default='models/checkpoints/baseline',
                        type=str, help="Path to store results.")
    parser.add_argument("--iou_threshold", default=0.9,
                        type=float, help="IoU threshold for non-maximum suppression")
    args = parser.parse_args()
    main(args)
