import random
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import torch
from typing import List

# detectron2 utils
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo

from dino.utils.data_preparation import train_test_split, get_duckietown_dicts
from dino.utils.visualization import visualize_predictions


def visualiza_predictions(duckietown_metadata: detectron2.data.catalog.Metadata,
                          dataset_dicts: List):
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        plt.imshow(out.get_image())
        plt.show()


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
    visualize_predictions(duckietown_metadata, dataset_dicts, 5)
    # %% Training Routine %%#
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("duckietown_train",)
    cfg.DATASETS.TEST = ("duckietown_test")  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4 # 12
    cfg.SOLVER.BASE_LR = 0.015
    cfg.SOLVER.MAX_ITER = 1500 # 15000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16  # 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)  # (kitti)
    cfg.OUTPUT_DIR = os.path.join('models/Detectron/checkpoints')
    # Saving config
    torch.save({'cfg': cfg}, cfg.OUTPUT_DIR + '/' + 'test' + '_cfg.final')
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    print("Start training")
    trainer.train()
    print('Saving model')
    torch.save(trainer.model, 'models/Detectron/checkpoints/test.pth')


if __name__ == '__main__':
    # Receive arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='data/dt',
                        type=str, help="Path of the dataset.")
    parser.add_argument("--fraction_test", default=0.5,
                        type=float, help="Percentage of test set.")
    args = parser.parse_args()
    main(args)
