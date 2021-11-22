"Utils."
import os

import torch
import vision_transformer as vits
from torchvision import transforms as pth_transforms
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_dino(patch_size=8):
    """Load vit_small model and send to device."""
    # From the original DINO code
    # Build model
    url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    model = vits.__dict__['vit_small'](patch_size=patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    model.load_state_dict(state_dict, strict=True)

    return model


def transform_img(img, patch_size=8):
    """Preprocess an image (PIL or array-like) for DINO compatibility."""
    # From the original DINO code
    # Transform image
    transform = pth_transforms.Compose([
        # pth_transforms.GaussianBlur(kernel_size=31, sigma=(50)),
        pth_transforms.Resize((480, 480)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
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


def dt_frames(subset=None, max=None, path=os.path.join('..', 'data', 'dt', 'frames')):
    """Generator to iterate over the dt object detection frames."""
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png') or f.endswith('.jpg')]
    files.sort()
    j = 0
    for i, f in enumerate(files):
        if subset is not None and i not in subset:
            continue
        with open(f, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        j+=1
        yield i, img
        if max is not None and j == max:
            break
