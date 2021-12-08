# Single image inference using Faster RCNN with Detectron2

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import cv2
import argparse
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def plot_image(params) -> None:
    """
    Single image inference with Faster RCNN + RCNN 50 backbone.
    @param img_path: Path to image
    @return: None
    """
    # Loading image
    im = cv2.imread(params.image_path)
    # Configuring model
    cfg = get_cfg()
    cfg.merge_from_file(params.config_file)  # Set model config
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set the testing threshold for this model
    cfg.MODEL.WEIGHTS = params.model_weights  # Set model weights
    predictor = DefaultPredictor(cfg)
    print(cfg)
    # Inference on Image
    outputs = predictor(im[..., ::-1])

    # Printing predicted instances and labels
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    # Visualizing results
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.axis('off')
    plt.imshow(out.get_image())
    plt.savefig('results/baseline11.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Receive arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default='data/dt/frames/frame_000061.png',
                        type=str, help="Path of the image to load.")
    parser.add_argument("--config_file", default='models/detectron/checkpoints/baseline/cfg.yaml',
                        type=str, help="Path to configuration file")
    parser.add_argument("--model_weights", default='models/detectron/checkpoints/baseline/model_final.pth',
                        type=str, help="Path to model weights.")
    args = parser.parse_args()
    plot_image(args)
