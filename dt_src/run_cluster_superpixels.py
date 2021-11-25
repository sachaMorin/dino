"""This script runs segmentation on the duckietown object detection dataset.

Can be used to visualize segmentation result or save segmentation for the whole dataset.
The dataset should first be embedded with run_embed_all.py."""
import os

import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from dt_utils import get_dino, transform_img, dt_frames
from att_superpixels import compute_att_superpixels

SAVE_ALL = False  # Save png or plot results
VIEW_PCA = True
REFIT = False # Refit K-Means and PCA
N_SEGMENTS = 600
N_CLUSTERS = 30  # Number of clusters for K-Means

norm = 'standard_scale'  # Image wise normalization, 'standard_scale', 'sum' or None

# Color scheme for standard normalized and 30 clusters
# To do potentially reassign clusters
class_list = [
    'bg',  # 0 (unlabeled) class. The rest are clusters.
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
    'bg',
]
color_d = dict(
    duckie='gold',
    bot='blue',
    wheel='k',
    human='lightgray',
    w_lane='white',
    y_lane='yellow',
    stop='red',
    road='gray',
    side='gray',
    bg='gray',
    # Test colors for exploration
    m='magenta',
    l='lime',
    c='cyan',
    s='salmon',
)
cmap = ListedColormap([color_d[i] for i in class_list])

# Load embeddings
z = np.load(os.path.join('..', 'results', 'z_superpixels.npy'))
# z_img = np.load(os.path.join('..', 'results', 'z_images.npy'))

# Standardize superpixels from the same image
mean = list()  # Memorize normalization
std = list()
for i in range(int(z.shape[0]//N_SEGMENTS)):
    K = N_SEGMENTS
    if norm == 'standard_scale':
        m = z[i*K:(i+1)*K].mean(axis=0)
        s = z[i*K:(i+1)*K].std(axis=0)
        z[i*K:(i+1)*K] -= m
        z[i*K:(i+1)*K] /= s
        mean.append(m)
        std.append(s)
    elif norm == 'sum':
        z /= z.sum(axis=1, keepdims=True)
    else:
        pass

# KMeans and PCA on DINO space
if REFIT:
    print('Fitting K-Means...')
    m = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    m.fit(z)
    print('Fitting DR algo...')
    dr = PCA(n_components=2, random_state=42)
    dr.fit(z)

    joblib.dump(m, os.path.join('..', 'results', 'kmeans.joblib.pkl'))
    joblib.dump(dr, os.path.join('..', 'results', 'dr.pkl'))

# Load prefit estimators
m = joblib.load(os.path.join('..', 'results', 'kmeans.joblib.pkl'))
clr = m.predict(z) + 1  # Add 1 to keep the zero token as no label

# Visualize results
# Add a dummy point to shift the color scheme (to match the image we'll display where 0 is already background)
if VIEW_PCA:
    dr = joblib.load(os.path.join('..', 'results', 'dr.pkl'))
    clr = np.concatenate((clr, [-1]))
    z_pca = dr.transform(z)
    z_pca = np.vstack((z_pca, z_pca.mean(axis=0)))
    plt.scatter(*z_pca.T, c=clr, cmap=cmap)
    plt.show()


# Visualize cluster assignments of some frames
model = get_dino(8)

# If SAVE_ALL, save visualization for all images. Else visualize a few images with plt.show().
images_idx = None if SAVE_ALL else [817, 838, 386, 1769, 268, 1572, 374, 49, 1396, 97, 1319, 923, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1100]


for i, img in dt_frames(images_idx):
    # Load image
    img_dino = transform_img(img)

    # Compute super pixels and use them as masks
    super_pix = compute_att_superpixels(img_dino, n_segments=N_SEGMENTS, plot=False, device=img_dino.device)

    # Masked forward pass to get an embedding per mask
    z_i = model.forward_mask(img_dino, super_pix).cpu().numpy()

    if norm == 'standard_scale':
        z_i -= z_i[-K:].mean(axis=0, keepdims=True)
        z_i /= z_i[-K:].std(axis=0, keepdims=True)
    elif norm == 'sum':
        z_i/=z_i.sum(axis=1, keepdims=True)
    else:
        pass
    c = m.predict(z_i) + 1

    # Get image with cluster assignment
    super_pix *= torch.Tensor(c).to(super_pix.device).reshape((-1, 1, 1))
    img_clr = super_pix.sum(axis=0).cpu().detach().numpy()
    img_clr[-1, -N_CLUSTERS-1:] = np.arange(N_CLUSTERS+1)   # This is a hack to force the same color scheme on all images

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].set_title('Image', fontsize=20)
    axes[0].imshow(img.resize((480, 480)))
    axes[1].set_title(f'Unsupervised DINO Segmentation', fontsize=20)
    axes[1].imshow(img_clr, cmap=cmap)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()

    if SAVE_ALL :
        plt.savefig(os.path.join('..', 'results', 'video', f'image_{str(i).zfill(5)}.png'))
    else:
        plt.show()
