"""Embed top k superpixels for each image in the whole duckietown dataset.

Also runs k-means and pca on the resuling embedding.
The embeddings and the estimators are saved under ../results."""
import os

import numpy as np
import torch
from PIL import Image

from dt_utils import get_dino, transform_img, process_attentions
from att_superpixels import compute_att_superpixels

PATCH_SIZE = 8
K = 50 # Number of top attention of superpixels to consider
N_SEGMENTS = 200 # Number of segments to compute in a single image

# Script to visualize attention super pixels of a few image
# Load model
model = get_dino(PATCH_SIZE)

z = list()  # Super pixel embedding accumulator
z_img = list() # Whole image embedding accumulator
# Iterate over some frames in the duckietown dataset
for frame_no in [402, 1000]:
    # Load image
    image_path = os.path.join('..', 'data', 'dt', 'frames', f'frame_{str(frame_no).zfill(6)}.png')
    if not os.path.exists(image_path):
        continue
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    img_dino = transform_img(img)

    # Forward pass
    z_img_i = model(img_dino)
    attentions = model.get_last_selfattention(img_dino)
    attentions = process_attentions(attentions)

    # Superpixels
    super_pix = compute_att_superpixels(img, attentions, k=K, n_segments=N_SEGMENTS, plot=False)

    # TODO : this is prototype code. We redo inference on the whole image k times, which is unefficient.
    # TODO : To be improved. We only need to recompute the last MLP after masking the attention.
    z += [model(img_dino, cls_mask=p).cpu() for p in super_pix]
    z_img.append(z_img_i.cpu())

# Save super pixel embeddings
z = torch.cat(z).detach().cpu().numpy()
print(z.shape)
np.save(os.path.join('..', 'results', 'z_superpixels.npy'), z)

# Save full image embeddings
z_img = torch.cat(z_img).detach().cpu().numpy()
print(z_img.shape)
np.save(os.path.join('..', 'results', 'z_images.npy'), z_img)


