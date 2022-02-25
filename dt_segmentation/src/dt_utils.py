"Utils."
from __future__ import print_function

import os
import torch

from torchvision import transforms as pth_transforms
from PIL import Image

from .vision_transformer import vit_small

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATA_PATH = os.path.join('../..', 'data', 'dt_sim')
RESULTS_PATH = os.path.join('../..', 'results')


def get_dino(patch_size=8, device=DEVICE):
    """Load vit_small model and send to device."""
    # From the original DINO code
    # Build model
    url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    model =vit_small(patch_size=patch_size, num_classes=0)
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


def dt_frames(subset=None, max=None, path=os.path.join('../..', 'data', 'dt', 'frames'), label_path=None):
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


def parse_class_names(path):
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(path).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    return class_names, class_name_to_id