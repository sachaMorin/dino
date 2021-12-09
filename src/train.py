import os
import argparse
import torch
import yaml

# detectron2 utils
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog

from dino.lib.data_preparation import train_test_split, get_duckietown_dicts
from dino.lib.visualization import visualize_dataset


def main(params):
    # Create train and test split
    train_test_split(params.dataset_path, params.fraction_test)
    # Set dataset in COCO format and register
    class_list = ['cone', 'duckie', 'duckiebot']
    for d in ["train", "test"]:
        DatasetCatalog.register("duckietown_" + d,
                                lambda d=d: get_duckietown_dicts(os.path.join(params.dataset_path, d)))
        MetadataCatalog.get('duckietown_' + d).set(thing_classes=class_list)
    # Visualizing a random image
    duckietown_metadata = MetadataCatalog.get('duckietown_train')
    dataset_dicts = get_duckietown_dicts(os.path.join(params.dataset_path, 'train'))
    visualize_dataset(duckietown_metadata, dataset_dicts, 5)
    # Training Routine %%#
    cfg = get_cfg()
    cfg.merge_from_file(params.config_file)
    cfg.DATASETS.TRAIN = ("duckietown_train",)
    cfg.DATASETS.TEST = ("duckietown_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = params.model_weights
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 0.015 # Learning rate for DINO set to 0.0003
    cfg.SOLVER.MAX_ITER = 2000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)  # Number of classes
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.OUTPUT_DIR = params.output_dir
    # Save config file
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    final_config = cfg.dump()
    yaml_config = yaml.full_load(final_config)
    with open(os.path.join(params.output_dir, 'cfg.yaml'), 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    # Saving config
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    print("Start training")
    trainer.train()
    print('Saving model')
    torch.save(trainer.model, os.path.join(params.output_dir, 'model.pth'))


if __name__ == '__main__':
    # Receive arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='data/dt',
                        type=str, help="Path of the dataset.")
    parser.add_argument("--fraction_test", default=0.5,
                        type=float, help="Percentage of test set.")
    parser.add_argument("--config_file", default='models/detectron/faster_rcnn_R_50_FPN_3x.yaml',
                        type=str, help="Path to configuration file")
    parser.add_argument("--model_weights", default='models/detectron/model_final_280758.pkl',
                        type=str, help="Path to model weights.")
    parser.add_argument("--output_dir", default='models/checkpoints/baseline',
                        type=str, help="Path to store results.")
    args = parser.parse_args()
    main(args)
