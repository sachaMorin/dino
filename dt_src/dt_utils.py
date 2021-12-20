"Utils."
import os
import sys
import inspect

import cv2
import numpy as np

import torch

# Add parent dir to scope
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import vision_transformer as vits
from torchvision import transforms as pth_transforms
from PIL import Image

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATA_PATH = os.path.join('..', 'data', 'dt_sim')
RESULTS_PATH = os.path.join('..', 'results')

# Create class name/id/RGB associations
# First element is class name
# Second is class id
# Third element is the RGB color from simulator rendering. (Not perfect accurate for some class, see rgb_to_c_
# Fourth element is the displayed color for that class sin seg_viz.py
class_map = [
    ("floor", 0, "000000", "000000"),
    ("bg", 1, "ff00ff", "000000"),
    ("y_lane", 2, "ffff00", "ffff00"),
    ("w_lane", 3, "ffffff", "000099"),
    ("red_tape", 4, "fe0000", "fe0000"),
    ("duckiebot", 5, "ad0000", "ad0000"),
    ("sign", 6, "4a4342", "ffc0cb"),
    ("duckie", 7, "cfa923", "00ff00"),
    ("duckie_pass", 8, "846c16", "00ffff"),  # Passenger duckies are different objects!
    ("cone", 9, "ffa600", "ffa600"),
    ("house", 10, "279621", "279621"),
    ("bus", 11, "ebd334", "ff00ff"),
    ("truck", 12, "961fad", "df4f4f"),
    ("barrier", 13, "000099", "964b00"),
]


def to_rgb(hex):
    return [int(hex[i:i + 2], 16) for i in (0, 2, 4)]


# Convert Hex to RGB
CLASS_MAP = [(m[0], m[1], to_rgb(m[2]), to_rgb(m[3])) for m in class_map]


def get_dino(patch_size=8, device=DEVICE):
    """Load vit_small model and send to device."""
    # From the original DINO code
    # Build model
    url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    model = vits.__dict__['vit_small'](patch_size=patch_size, num_classes=0)
    model.to(device)
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    model.load_state_dict(state_dict, strict=True)

    return model


def transform_img(img, patch_size=8, grayscale=False, device=DEVICE):
    """Preprocess an image (PIL or array-like) for DINO compatibility."""
    # From the original DINO code
    # Transform image
    t = [pth_transforms.Grayscale(num_output_channels=3)] if grayscale else []
    t += [
        pth_transforms.Resize((480, 480)),
        pth_transforms.ToTensor(),
    ]
    if not grayscale:
        # Normalize with Image net values
        t += [pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    transform = pth_transforms.Compose(t)
    img = transform(img).to(device)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)
    # img[:, :, :120, :] = 0

    return img


def process_attentions(attentions, threshold=None, patch_size=8):
    """Extract CLS attentions and normalize them to keep only threshold density."""
    # From the original DINO code
    nh = attentions.shape[1]  # number of head
    w_featmap = 480 // patch_size
    h_featmap = 480 // patch_size

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # Thresholding
    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        attentions = th_attn.reshape(nh, w_featmap, h_featmap).float()

    # Reshape and convert to numpy array. Put channels (heads) at the end.
    attentions = attentions.reshape(nh, w_featmap, h_featmap)

    return attentions


def dt_frames(subset=None, max=None, path=os.path.join('..', 'data', 'dt', 'frames'), label_path=None):
    """Generator to iterate over the dt object detection frames."""
    files = [f for f in os.listdir(path) if f.endswith('.png') or f.endswith('.jpg')]
    files.sort()
    j = 0
    for i, f in enumerate(files):
        if subset is not None and i not in subset:
            continue
        with open(os.path.join(path, f), 'rb') as file:
            img = Image.open(file)
            img = img.convert('RGB')
        j += 1
        if label_path is None:
            yield i, img
        else:
            with open(os.path.join(label_path, f), 'rb') as file:
                mask = Image.open(file)
                mask = mask.convert('RGB')
            yield i, img, mask
        if max is not None and j == max:
            break


def rgb_to_c(mask_img, raw_img):
    """Map RGB pixels to class to create an (approximate) segmentation mask.
    Segmentation colors from the simulator are not perfect (e.g., yellow lanes often have an offset)
    so we still use HSV filters over the raw image for some classes.

    Args:
        mask_img(PIL): Simulation rendering of the image (approximately discrete colors).
        raw_img(PIL): Image of interest.

    Returns:
        (np.ndarray) : (image height, image width) array with class assignments.

    """
    mask_img = np.array(mask_img)
    raw_img = np.array(raw_img)
    raw_hsv = cv2.cvtColor(raw_img, cv2.COLOR_RGB2HSV)

    result = np.zeros(mask_img.shape[:-1], dtype='int')
    for m in CLASS_MAP[1:]:
        if m[0] == 'duckiebot':
            # Also map wheel and camera to duckiebot class
            mask = (mask_img == m[2]) + (mask_img == [30, 12, 5])
            # Add backplate from the raw image
            mask += raw_img == [0, 0, 0]  # Pure black pixels
            mask = mask.all(axis=-1)

            # Get rest of the plate
            # Won't capture all backplates, but filter needs to be conservative to not capture white lanes/floor
            # Will cover some signs, but since we process signs after, the class in result should be mostly correct.
            lower_rgb = np.array([88, 88, 88])
            higher_rgb = np.array([95, 95, 95])
            mask += cv2.inRange(raw_img, lower_rgb, higher_rgb) == 255
        elif m[0] == 'y_lane':
            # Yellow lanes are trickier so we use HSV filter
            lower_hsv = np.array([25, 60, 150])
            higher_hsv = np.array([30, 255, 255])
            mask = cv2.inRange(raw_hsv, lower_hsv, higher_hsv) == 255
            result[mask] = m[1]
        elif m[0] == 'red_tape':
            # Red tape is also tricky
            lower_hsv = np.array([175, 120, 0])
            higher_hsv = np.array([180, 255, 255])
            mask = cv2.inRange(raw_hsv, lower_hsv, higher_hsv) == 255
        elif m[0] == 'sign':
            # 3 types of pixel for signs
            mask = (mask_img == m[2]) + (mask_img == [52, 53, 8]) + (mask_img == [76, 71, 71])
            mask = mask.all(axis=-1)
        elif m[0] == 'w_lane':
            # Include some grey pixels in white lanes
            lower_hsv = np.array([0, 0, 145])
            higher_hsv = np.array([180, 40, 255])
            mask = cv2.inRange(raw_hsv, lower_hsv, higher_hsv) == 255
        else:
            mask = (mask_img == m[2]).all(axis=-1)
        result[mask] = m[1]

    return result


def c_to_rgb(seg_mask):
    """Map segmentation mask back to RGB space.

    Args:
        seg_mask(np.ndarray): Segmentation mask.

    Returns:
        (np.ndarray) : (image height, image width, 3) RGB array.

    """
    result = list()
    for row in seg_mask:
        result.append(list())
        for p in row:
            result[-1].append(CLASS_MAP[p][2])
    return np.array(result)


def prepare_seg_dataset():
    """Compute segmentation masks for train, val and test splits and save them in the data/torch folder.
    Save segmentation mask of original size and a 60x60 version to match dino "patch resolution".
    """
    for split in ['train', 'val', 'test']:
        # Get dataset
        data = dt_frames(path=os.path.join('..', 'data', 'dt_sim', split, 'images'),
                         label_path=os.path.join('..', 'data', 'dt_sim', split, 'labels'))

        for i, img, sim_mask in data:
            print(f'Processing image no {i}')

            # Compute labels
            # We need to use the original image as well since the lane labels
            # in the mask are NOT reliable
            seg_mask = rgb_to_c(sim_mask, img)  # Discretize

            # Resize labels to match size of transformer patch space
            seg_mask_small = cv2.resize(seg_mask, (60, 60), interpolation=cv2.INTER_NEAREST)

            # Save image and labels for future torch training
            img.save(os.path.join('..', 'data', 'torch', split, f'{i}_x.png'))
            cv2.imwrite(os.path.join('..', 'data', 'torch', split, f'{i}_y.png'), seg_mask)
            torch.save(torch.from_numpy(seg_mask_small).reshape((-1,)),
                       f=os.path.join('..', 'data', 'torch', split, f'{i}_y.pt'))
