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
i = 0
path = os.path.join('..', 'data', 'dt', 'frames')
for image_name in os.listdir(path):
    print(f'Processing frame no {i}...')
    i+= 1
    # Load image
    if not image_name.endswith('.png'):
        continue
    with open(os.path.join(path, image_name), 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    img_dino = transform_img(img)

    # Forward pass
    z_img_i, attentions = model.forward_warmup(img_dino)
    z_img.append(z_img_i.cpu())
    attentions = process_attentions(attentions)

    # Superpixels
    super_pix = compute_att_superpixels(img, attentions, k=K, n_segments=N_SEGMENTS, plot=False)

    # Get super pixel embeddings
    # forward_mask only runs the last block with the masked attentions, much faster than running the full architecture
    z += [model.forward_mask(p).cpu() for p in super_pix]

# Save super pixel embeddings
z = torch.cat(z).detach().cpu().numpy()
print(z.shape)
np.save(os.path.join('..', 'results', 'z_superpixels.npy'), z)

# Save full image embeddings
z_img = torch.cat(z_img).detach().cpu().numpy()
print(z_img.shape)
np.save(os.path.join('..', 'results', 'z_images.npy'), z_img)


