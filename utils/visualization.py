import random
import cv2
import matplotlib.pyplot as plt
from typing import List

import detectron2
from detectron2.utils.visualizer import Visualizer


def visualize_predictions(duckietown_metadata: detectron2.data.catalog.Metadata,
                          dataset_dicts: List, imgs: int = 3) -> None:
    """
    Visualize a given set of instances in the train/test set
    @param duckietown_metadata: Metadata file with duckietown claases
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
