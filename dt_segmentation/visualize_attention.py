#!/usr/bin/env python
"""Script to compute attention of a given image.

Save the visualizations in target_dir.
dt_segmentation/visualize_attention.py results/3_mlp_frozen_1234_finetuned.ckpt data/dt_real_voc_test/JPEGImages/left0621.jpg attn
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from src.pl_torch_modules import DINOSeg
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
torch.cuda.set_per_process_memory_fraction(1.)


def vis_mask(checkpoint_path, filename, target_dir, resolution=480):
    """Use a trained PL checkpoint to compute attention mask on given image."""

    patch_size = 8

    # mlp_dino = DINOSeg(data_path='dummy', write_path='dummy', n_blocks=3)
    mlp_dino = DINOSeg.load_from_checkpoint(checkpoint_path).to('cuda:0' if torch.cuda.is_available() else 'cpu')

    # This only affects the inference resolution. The output is still 480x480
    mlp_dino.set_resolution(resolution)

    with torch.no_grad():

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        with open(filename, 'rb') as file:
            img = Image.open(file)
            x = img.convert('RGB')

        # Ge predictions
        x = mlp_dino.transforms(image=np.array(x))['image'].unsqueeze(0).to(mlp_dino.device)
        attentions = mlp_dino.dino.get_last_selfattention(x)
        nh = attentions.shape[1]  # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        attentions = attentions.reshape(nh, resolution // patch_size, resolution // patch_size)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0),
                                               scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

        torchvision.utils.save_image(torchvision.utils.make_grid(x, normalize=True, scale_each=True),
                                     os.path.join(target_dir, 'img.png'))
        for j in range(nh):
            fname = os.path.join(target_dir, "attn-head-dino" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            print(f"{fname} saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("checkpoint_path", help="Trained PL checkpoint")
    parser.add_argument("filename", help="Image to compute attention on")
    parser.add_argument("target_dir", help="Where to save attentions")
    parser.add_argument("--resolution", help="Prediction resolutions.", required=False,
                        default=480, type=int)
    args = parser.parse_args()

    vis_mask(**vars(args))
