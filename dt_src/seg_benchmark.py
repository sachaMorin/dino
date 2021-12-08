"""Quick script to benchmark prediction time."""
import os

import torch.cuda

from seg_run import load_mlp
from dt_utils import get_dino,transform_img, dt_frames
import time

import torch.nn as nn

N = 2000

data = dt_frames(path=os.path.join('..', 'data', 'dt_sim', 'all', 'images'))
model = get_dino(8, device='cuda:0')
clf = load_mlp(device='cuda:0')
print(torch.cuda.max_memory_allocated('cuda:0')/1e9)


# Load just one image
for i, img in data:
    the_image = img
    break

start = time.time()

# Put only required layers in GPU
# for i, blk in enumerate(model.blocks):
#     model.blocks[i] = blk.to('cuda:0')
#     if i == 3:
#         break
torch.cuda.empty_cache()

# Predict
for _ in range(N):
    # Load image
    img_dino = transform_img(the_image, device='cuda:0')

    # Get DINO embedding for all patches
    z_i = model(img_dino, all=True).squeeze(0)[1:]
    nn.functional.normalize(z_i, dim=1, p=2, out=z_i)
    pred = clf.predict(z_i).reshape((60, 60)).cpu()

t = time.time() - start
print(f"Segmented {N} images in {t:.2f} seconds, i.e., {N/t:.3f} Hz")
