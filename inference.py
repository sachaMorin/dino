#!/usr/bin/env python
"""Script to run inference on a folder of images.

Save the visualizations in target_dir."""
import os
import glob
import argparse

import numpy as np
import imgviz
from PIL import Image
import cv2

from dt_segmentation import DINOSeg, DuckieSegDataset
from labelme2voc import parse_class_names
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def inference(checkpoint_path, image_dir, target_dir, labels_path):
    """Use a trained PL checkpoint to run inference on all images in image_dir."""
    mlp_dino = DINOSeg.load_from_checkpoint(checkpoint_path)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get transforms
    transforms = DuckieSegDataset("dummy").t

    # Get class names and length
    class_names, _ = parse_class_names(labels_path)

    for filename in glob.glob(os.path.join(image_dir, "*.jpg")):
        with open(filename, 'rb') as file:
            img = Image.open(file)
            x = img.convert('RGB')

        x_transformed = transforms(x)
        pred = mlp_dino.predict(x_transformed.unsqueeze(0)).reshape((60, 60))

        # Resize the original image and the predictions to 480 x 480
        img = cv2.resize(np.array(x), (480, 480))
        pred = np.kron(pred, np.ones((8, 8))).astype(int)  # Upscale the predictions back to 480x480


        # Save image
        viz = imgviz.label2rgb(
            pred,
            imgviz.rgb2gray(img),
            font_size=15,
            label_names=class_names,
            loc="rb",
        )
        f = filename.split(os.sep)[-1]
        imgviz.io.imsave(os.path.join(target_dir, f), viz)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("checkpoint_path", help="Trained PL checkpoint")
    parser.add_argument("image_dir", help="Images to run inference on")
    parser.add_argument("target_dir", help="Where to save predictions")
    parser.add_argument("--labels_path", help="Txt file with class labels.", required=False, default=os.path.join("data", "labels.txt"))
    args = parser.parse_args()

    inference(**vars(args))
