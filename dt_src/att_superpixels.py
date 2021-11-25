"Unsupervised object detection experiment using DINO attention layers."
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as pth_transforms
from skimage.segmentation import slic
from dt_utils import get_dino, transform_img, process_attentions

import matplotlib.pyplot as plt

PATCH_SIZE = 8


def compute_att_superpixels(image, n_segments=200, patch_size=8, attentions=None, k=None, plot=False, device='cpu'):
    """Computes top k attention superpixels.

    Compute superpixels over the base image (downsized to patch size) and returns the masks of the top
    k superpixels in terms of average transformer attention. Attention can also be omitted to simply return
    all masks.

    Args:
        image(torch.Tensor):
        n_segments(int): Number of superpixels to consider.
        patch_size(int): Patch size of the transformer architecture.
        k(int): Number of superpixels to keep (in terms of average attention). Ignored if Attention is none.
        plot(bool): Plot image, average attention and selected superpixels.
        attentions(torch.Tensor): Attentions of the CLS token.
        device(str): Torch device.

    Returns:
        torch.Tensor: Tensor of size (k, image_height, image_width) where each channel is a superpixel mask.


    """
    # Downsize image to patch dimensions
    transform = pth_transforms.Compose([
        pth_transforms.Resize((480//patch_size, 480//patch_size)),
        # pth_transforms.ToTensor(),
    ])
    small_img = transform(image)

    # Rearange images to RGB convention
    small_img = small_img.squeeze(0).permute((1, 2, 0))

    # Run SLIC on small image
    seg_img = slic(small_img.cpu().detach().numpy(), n_segments=n_segments, enforce_connectivity=True)
    seg_img = torch.from_numpy(seg_img).to(device)

    masks = torch.zeros((n_segments, seg_img.shape[0], seg_img.shape[1])).to(device)
    for i in range(n_segments):
        masks[i] = seg_img == i

    # Find top k superpixels with most attention
    if attentions is not None:
        # Note : I stopped using this block. Needs to be retested if used.
        att_sum = attentions.sum(axis=0)

        mask_sum = masks.sum(axis=-1).sum(axis=-1)
        mask_sum[mask_sum == 0] = 1 # Avoid division by 0
        sums = (masks * att_sum).sum(axis=-1).sum(axis=-1)/mask_sum
        # sums = (masks * att_sum).max(axis=-1).values.max(axis=-1).values

        order = sums.argsort()

        # Keep only masks of top super pixels with top attention
        result = masks[order[-k:]]
    else:
        # Keep all masks
        result = masks

    if plot and attentions is not None:
        # Get a 480x480 copy of the original image for visualizing purposes
        transform = pth_transforms.Compose([
            pth_transforms.Resize((480, 480)),
            pth_transforms.ToTensor(),
        ])
        img_og = transform(image).squeeze(0).permute((1, 2, 0))

        # Find top regions
        top_regions = torch.zeros(seg_img.shape).to(image.device)

        for i, o in enumerate(order[-k:]):
            top_regions += i * (seg_img == o)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title('Image')
        axes[0].imshow(img_og.cpu().detach().numpy())
        axes[1].set_title('Avg. Attention')
        axes[1].imshow(attentions.mean(axis=0).cpu().detach().numpy(), cmap='inferno')
        axes[2].set_title(f'Top {k} Superpixels')
        axes[2].imshow(top_regions.cpu().detach().numpy(), cmap='inferno')
        for a in axes:
            a.set_xticks([])
            a.set_yticks([])
        plt.tight_layout()
        plt.show()

    return result


if __name__ == '__main__':
    # Script to visualize attention super pixels of a few image
    # Load model
    model = get_dino(PATCH_SIZE)

    # Iterate over some frames in the duckietown dataset
    for frame_no in [32, 402, 800, 1000, 1200, 1600]:
        # Load image
        image_path = os.path.join('..', 'data', 'dt', 'frames', f'frame_{str(frame_no).zfill(6)}.png')
        if not os.path.exists(image_path):
            continue
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img_dino = transform_img(img)

        # Forward pass
        attentions = model.get_last_selfattention(img_dino)
        attentions = process_attentions(attentions)

        # Top 10 superpixels
        super_pix = compute_att_superpixels(img, attentions, k=70, n_segments=200, plot=True)

