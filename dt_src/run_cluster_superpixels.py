import os

import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap import UMAP

from dt_utils import get_dino, transform_img, process_attentions, dt_frames
from att_superpixels import compute_att_superpixels

REFIT = False  # Refit K-Means and PCA
K = 50
N_SEGMENTS = 200
N_CLUSTERS = 30

norm = 'mean'

# This is for K = 50 (standard normalization) and 12 clusters
cmap = ListedColormap([
    'gray',
    'yellow',
    'gray',
    'red',
    'gray',
    'orange',
    'pink',
    'white',
    'red',
    'red',
    'gray',
    'gray',
    'orange',
])

# Standard normalized and 18 clusters
cmap = ListedColormap([
    'gray',
    'gray',
    'white',
    'gray',
    'gray',  # Duckiebot but messy
    'gray',
    'gray',
    'orange',
    'red',
    'orange',
    'gray',  # Grass
    'red',
    'white',
    'black',
    'gray',
    'gray',  # Close road
    'red',
    'yellow',
    'pink',
])

# Standard normalized and 30 clusters
cmap = ListedColormap([
    'gray',
    'gray',  # Signs?
    'gray',
    'red',  # Red tape?
    'gray',  # Close road
    'blue',
    'blue',
    'white',
    'blue',
    'k',
    'gray',
    'gray',
    'gray',
    'gold',
    'gray',  # Messy duckiebot
    'white',
    'gray',
    'blue',  # Front lights
    'gray',
    'gray',
    'gray', #Grass
    'white',
    'k', # Wheels?
    'gold',
    'lightgray', #oomans
    'lightgray',
    'white',
    'gray',
    'blue',  # Front Dash
    'yellow',
    'yellow',
])

# Load embeddings
z = np.load(os.path.join('..', 'results', 'z_superpixels.npy'))
z_img = np.load(os.path.join('..', 'results', 'z_images.npy'))

# Standardize superpixels from the same image
mean = list()  # Memorize normalization
std = list()
for i, z_i in enumerate(z_img):
    if norm == 'mean':
        m = z[i*K:(i+1)*K].mean(axis=0)
        s = z[i*K:(i+1)*K].std(axis=0)
        z[i*K:(i+1)*K] -= m
        z[i*K:(i+1)*K] /= s
        mean.append(m)
        std.append(s)
    else:
        z /= z.sum(axis=1, keepdims=True)


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
dr = joblib.load(os.path.join('..', 'results', 'dr.pkl'))
clr = m.predict(z)
z_pca = dr.transform(z)

# Visualize results
# Add a dummy point to shift the color scheme (to match the image we'll display where 0 is already background)
clr = np.concatenate((clr, [-1]))
z_pca = np.vstack((z_pca, z_pca.mean(axis=0)))
plt.scatter(*z_pca.T, c=clr + 1, cmap=cmap)
plt.show()


# Visualize cluster assignments of some frames
model = get_dino(8)

for i, img in dt_frames([817, 838, 386, 1769, 268, 1572, 374, 49, 1396, 97, 1319, 923]):
# for i, img in dt_frames(list(range(1, 1950, 75))):
# for i, img in dt_frames(path=os.path.join('..', 'data', 'custom')):
# for i, img in dt_frames(subset=[18, 48, 817, 838, 1769, 1572]):
    # Load image
    img_dino = transform_img(img)

    # Forward pass
    _, attentions = model.forward_warmup(img_dino)
    attentions = process_attentions(attentions)

    # Superpixels
    super_pix = compute_att_superpixels(img, attentions, k=200, n_segments=N_SEGMENTS, plot=False)
    z_i = torch.cat([model.forward_mask(p).cpu() for p in super_pix]).detach().cpu().numpy()
    if norm == 'mean':
        z_i -= z_i[-K:].mean(axis=0, keepdims=True)
        z_i /= z_i[-K:].std(axis=0, keepdims=True)
    else:
        z_i/=z_i.sum(axis=1, keepdims=True)
    c = m.predict(z_i)

    # Get image with cluster assignment
    super_pix *= torch.Tensor(c).to(super_pix.device).reshape((-1, 1, 1)) + 1
    img_clr = super_pix.sum(axis=0).cpu().detach().numpy()
    img_clr[-1, -N_CLUSTERS-1:] = np.arange(N_CLUSTERS+1)   # This is a hack to force the same color scheme on all images

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].set_title('Image', fontsize=20)
    axes[0].imshow(img.resize((480, 480)))
    # axes[1].set_title('Avg. Attention')
    # axes[1].imshow(attentions.mean(axis=0).cpu().detach().numpy(), cmap='inferno')
    axes[1].set_title(f'Unsupervised DINO Segmentation', fontsize=20)
    axes[1].imshow(img_clr, cmap=cmap)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    # plt.savefig(os.path.join('..', 'results', 'video', f'image_{str(i).zfill(5)}.png'))
    plt.show()
