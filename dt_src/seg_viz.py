"""Visualize MLP results from seg_run.py"""
import os

import torch.nn as nn

from skimage import color
import matplotlib.pyplot as plt
import cv2
import numpy as np

from seg_run import CLASS_MAP, c_to_rgb, load_mlp, rgb_to_c
from dt_utils import get_dino,transform_img, dt_frames

col = [m[0] for m in CLASS_MAP]

SAVE = True

# Viz some results
data = dt_frames(path=os.path.join('..', 'data', 'dt_sim', 'all', 'images'),
                 label_path=os.path.join('..', 'data', 'dt_sim', 'all', 'labels'))
model = get_dino(8)
clf = load_mlp()

for i, img, mask in data:
    # Load image
    img_dino = transform_img(img)

    # Get DINO embedding for all patches
    z_i = model(img_dino, all=True).squeeze(0)[1:]
    nn.functional.normalize(z_i, dim=1, p=2, out=z_i)
    pred = clf.predict(z_i).reshape((60, 60))
    seg = c_to_rgb(pred)
    
    # View mask
    # mask = cv2.resize(np.array(mask), (480, 480))
    # plt.imshow(mask)
    # plt.show()
    # mask = c_to_rgb(rgb_to_c(mask))
    # plt.imshow(mask)
    # plt.show()

    # Plot
    a = 1
    fig, axes = plt.subplots(1, a, figsize=(int(5 * a), 6))
    if a == 1:
        axes = [axes]
    # axes[0].set_title(f'Supervised DINO Segmentation', fontsize=20)
    small_img = cv2.resize(np.array(img), (480, 480))
    pred[pred == 1] = 0  # Hide floor and background predictions (map floor to bg label)
    big_pred = np.kron(pred.cpu().numpy(), np.ones((8, 8)))  # Upscale back
    n = len(CLASS_MAP)
    col = [np.array(m[2]) / 255 for m in CLASS_MAP]
    # Change duckie color to to distinguish it from yellow lane
    col[5] = np.array([253, 185, 200])/255
    col[6] = [0, 1, 1]
    col[7] = [0, 1, 0]

    # Change bus to magenta
    col[10] = [1, 0, 1]
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
