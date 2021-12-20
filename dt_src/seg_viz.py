"""Visualize MLP results from seg_fit.py"""
import os

import torch.nn as nn

from skimage import color
import matplotlib.pyplot as plt
import cv2
import numpy as np

from seg_fit import DINOSeg
from dt_src.dt_utils import CLASS_MAP, rgb_to_c, c_to_rgb
from dt_utils import transform_img, dt_frames, RESULTS_PATH

SAVE = True

# Viz some results
data = dt_frames(path=os.path.join('..', 'data', 'dt_sim', 'test', 'images'),
                 label_path=os.path.join('..', 'data', 'dt_sim', 'test', 'labels'))

# m = load_DINOSeg('cuda:0')
m = DINOSeg.load_from_checkpoint(os.path.join(RESULTS_PATH, 'linear_frozen.pt'))
m.to('cuda')

# Display colors
col = [np.array(m[3]) / 255 for m in CLASS_MAP]


for i, img, mask in data:
    # Load image
    img_dino = transform_img(img).to(m.device)

    # Get DINO embedding for all patches
    pred = m.predict(img_dino, tensor=False).reshape((60, 60))

    # Plot
    a = 1
    fig, axes = plt.subplots(1, a, figsize=(int(5 * a), 6))
    if a == 1:
        axes = [axes]
    small_img = cv2.resize(np.array(img), (480, 480))
    big_pred = rgb_to_c(cv2.resize(np.array(mask), (480, 480)), small_img)
    big_pred = np.kron(pred, np.ones((8, 8)))  # Upscale back
    n = len(CLASS_MAP)
    big_pred[-1, -n:] = np.arange(n)
    big_mask = color.label2rgb(big_pred, small_img, colors=col)
    axes[0].imshow(big_mask)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    if SAVE:
        plt.savefig(os.path.join('..', 'results', 'video', f'image_{str(i).zfill(5)}.png'))
        plt.close()
    else:
        plt.show()
