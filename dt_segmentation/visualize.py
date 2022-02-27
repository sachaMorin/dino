#!/usr/bin/env python
"""Script to run inference on a folder of images.

Save the visualizations in target_dir."""
import os
import glob
import argparse

import numpy as np
import torch
import imgviz
from PIL import Image

from src.pl_torch_modules import DINOSeg
from src.dt_utils import parse_class_names
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
torch.cuda.set_per_process_memory_fraction(1.)


def inference(checkpoint_path, image_dir, target_dir, labels_path, resolution=480):
    """Use a trained PL checkpoint to run inference on all images in image_dir."""
    mlp_dino = DINOSeg.load_from_checkpoint(checkpoint_path).to('cuda:0' if torch.cuda.is_available() else 'cpu')

    # This only affects the inference resolution. The output is still 480x480
    mlp_dino.set_resolution(resolution)

    with torch.no_grad():

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Get class names and length
        class_names, _ = parse_class_names(labels_path)

        for ext in ['jpg', 'png']:
            for filename in glob.glob(os.path.join(image_dir, f"*.{ext}")):
                with open(filename, 'rb') as file:
                    img = Image.open(file)
                    x = img.convert('RGB')

                # Ge predictions
                pred = mlp_dino.predict(x)

                # Save image
                viz = imgviz.label2rgb(
                    pred,
                    imgviz.rgb2gray(np.array(x.resize((480, 480)))),
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
    parser.add_argument("--labels_path", help="Txt file with class labels.", required=False,
                        default=os.path.join("data", "labels.txt"))
    parser.add_argument("--resolution", help="Prediction resolutions.", required=False,
                        default=480, type=int)
    args = parser.parse_args()

    inference(**vars(args))
