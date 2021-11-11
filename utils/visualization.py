import random
import cv2
import matplotlib.pyplot as plt
from typing import List

import detectron2
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
# Evaluation imports
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


def visualize_dataset(duckietown_metadata: detectron2.data.catalog.Metadata,
                      dataset_dicts: List, imgs: int = 3) -> None:
    """
    Visualize a given set of instances in the train/test set
    @param duckietown_metadata: Metadata file with duckietown classes
    @param dataset_dicts:  List with bounding boxes and classes information
    @param imgs: Number of images to display
    @return: None
    """
    for d in random.sample(dataset_dicts, imgs):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=duckietown_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        plt.imshow(out.get_image())
        plt.show()


def visualize_predictions(dataset_dicts: List, predictor: detectron2.engine.defaults.DefaultPredictor,
                          duckietown_metadata: detectron2.data.catalog.Metadata, imgs: int = 3) -> None:
    """
    Inference over a set of random images in the test set using the trained model
    @param dataset_dicts: List of elements in the dataset in COCO format
    @param predictor: Trained model
    @param duckietown_metadata: Metadata file with duckietown classes
    @param imgs: Number of images to plot
    @return: None
    """
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=duckietown_metadata,
                       scale=0.6,
                       instance_mode=ColorMode.IMAGE
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image())
        plt.show()
