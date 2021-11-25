"""Embed superpixels for each image in the whole duckietown dataset.

Embeddings are saved in ../results."""
import os

import numpy as np

from dt_utils import get_dino, transform_img, dt_frames
from att_superpixels import compute_att_superpixels

PATCH_SIZE = 8
N_SEGMENTS = 600 # Number of segments to compute in a single image

# Script to visualize attention super pixels of a few image
# Load model
model = get_dino(PATCH_SIZE)

z = list()  # Super pixel embedding accumulator

# Iterate over some frames in the duckietown dataset
for i, img in dt_frames():
    print(f'Processing frame no {i}...')

    # Load image
    img_dino = transform_img(img)

    # Compute super pixels and use them as masks
    super_pix = compute_att_superpixels(img_dino, n_segments=N_SEGMENTS, plot=False, device=img_dino.device)

    # Masked forward pass to get an embedding per mask
    z_i = model.forward_mask(img_dino, super_pix).cpu().numpy()
    z.append(z_i)

# Save super pixel embeddings
z = np.vstack(z)
print(z.shape)
np.save(os.path.join('..', 'results', 'z_superpixels.npy'), z)

