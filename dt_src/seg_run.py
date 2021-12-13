"""Script to embed the simulation dataset and get segmentation predictions from an MLP, a linear classifier and
a knn classifier.

Train data should be organized in the folder data/dt_sim/train/images and data/dt_sim/train/labels.
Val data should be organized in the folder data/dt_sim/val/images and data/dt_sim/val/labels.

All results are saved to the results folder.
"""
import torch
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import balanced_accuracy_score

from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from eval_knn import knn_classifier

from dt_utils import get_dino, transform_img, dt_frames

DATA_PATH = os.path.join('..', 'data', 'dt_sim')
RESULTS_PATH = os.path.join('..', 'results')

# Create class name/id/RGB associations
# RGB color should reflect the color of the class in simulator
class_map = [
    ("floor", 0, "000000"),
    ("bg", 1, "ff00ff"),
    ("y_lane", 2, "ffff00"),
    ("w_lane", 3, "ffffff"),
    ("red_tape", 4, "fe0000"),
    ("sign", 5, "4a4342"),
    ("duckie", 6, "cfa923"),
    ("duckie_pass", 7, "846c16"),  # Passenger duckies are different objects!
    ("cone", 8, "ffa600"),
    ("house", 9, "279621"),
    ("bus", 10, "ebd334"),
    ("truck", 11, "961fad"),
    ("barrier", 12, "000099"),
    ("duckiebot", 13, "ad0000"),
]

# Convert hexa to rgb
CLASS_MAP = [(m[0], m[1], [int(m[2][i:i + 2], 16) for i in (0, 2, 4)]) for m in class_map]


def rgb_to_c(img):
    """Map RGB pixels to class to create an (approximate) segmentation mask.
    Segmentation colors from the simulator are not perfect (e.g., yellow lanes are actually represented by different
    colors) so we still need HSV filters for some class.

    Args:
        img(PIL): Image of interest.

    Returns:
        (np.ndarray) : (image height, image width) array with class assignments.

    """
    img = np.array(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    raw = np.zeros(img.shape[:-1], dtype='int')
    for m in CLASS_MAP[1:]:
        if m[0] == 'duckiebot':
            # Also map wheel and camera to duckiebot class
            mask = (img == m[2]) + (img == [30, 12, 5])
            mask = mask.all(axis=-1)
        elif m[0] == 'y_lane':
            # Yellow lanes are trickier so we use HSV filter
            lower_hsv = np.array([27.5, 0, 70])
            higher_hsv = np.array([32.5, 255, 255])
            mask = cv2.inRange(hsv, lower_hsv, higher_hsv) == 255
            raw[mask] = m[1]
        elif m[0] == 'red_tape':
            # Red tape is also tricky
            lower_hsv = np.array([0, 200, 200])
            higher_hsv = np.array([5, 255, 255])
            mask = cv2.inRange(hsv, lower_hsv, higher_hsv) == 255
        elif m[0] == 'sign':
            # 3 types of pixel for signs
            mask = (img == m[2]) + (img == [52, 53, 8]) + (img == [76, 71, 71])
            mask = mask.all(axis=-1)
        elif m[0] == 'w_lane':
            # Include some grey pixels in white lanes
            lower_hsv = np.array([0, 0, 125])
            higher_hsv = np.array([0, 0, 255])
            mask = cv2.inRange(hsv, lower_hsv, higher_hsv) == 255
        else:
            mask = (img == m[2]).all(axis=-1)
        raw[mask] = m[1]

    return raw


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


def prepare_seg_dataset(grayscale=False):
    """This will embed the segmentation dataset with DINO and prepare labels for consumption by a downstream classifier.
    Processed data will be saved in the results folder under the names z_train.pt, y_train.pt, z_val.pt and y_val.pt.

    """
    # Get model
    model = get_dino(8)

    for split in ['train', 'val']:
        # Get dataset
        data = dt_frames(path=os.path.join('..', 'data', 'dt_sim', split, 'images'),
                         label_path=os.path.join('..', 'data', 'dt_sim', split, 'labels'))

        results = dict(z=list(), y=list())

        # Load DINO
        for i, img, mask in data:
            print(f'Processing image no {i}')
            # Load image
            img_dino = transform_img(img, grayscale=grayscale)

            # Get DINO embedding for all patches
            z_i = model(img_dino, all=True).squeeze(0)[1:].cpu()

            # Standard scale
            # z_i -= z_i.mean(axis=0, keepdims=True)
            # z_i /= z_i.std(axis=0, keepdims=True)
            nn.functional.normalize(z_i, dim=1, p=2, out=z_i)

            # Compute labels
            seg_mask = rgb_to_c(mask)  # Discretize

            # Resize labels to match size of transformer patch space
            seg_mask = cv2.resize(seg_mask, (60, 60), interpolation=cv2.INTER_NEAREST)
            seg_mask = np.array(seg_mask).reshape((-1,))

            # Keep only 400 background and floor patches per image to keep the dataset size manageable
            bg = np.argwhere(seg_mask < 2).reshape((-1,))
            keep = np.random.choice(bg, size=400, replace=False)
            not_bg = seg_mask >= 2
            z_i = torch.vstack((z_i[not_bg], z_i[keep]))
            y_i = torch.from_numpy(np.concatenate((seg_mask[not_bg], seg_mask[keep])))

            results['z'].append(z_i)
            results['y'].append(y_i)

        # Save tensors
        path = os.path.join('..', 'data', 'dt_sim')
        torch.save(torch.cat(results['z']), f=os.path.join(path, f'z_{split}.pt'))
        torch.save(torch.cat(results['y']), f=os.path.join(path, f'y_{split}.pt'))


class TensorDataset(Dataset):
    """Handy wrapper class for input tensors x and label tensor y."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LitSegMLP(pl.LightningModule):
    """MLP for patch segmentation."""

    def __init__(self, mlp=True):
        super().__init__()
        if mlp:
            self.layer_1 = nn.Linear(384, 200)
            self.layer_2 = nn.Linear(200, 100)
            self.layer_3 = nn.Linear(100, len(CLASS_MAP))
        else:
            # Linear regressor
            self.layer_1 = nn.Linear(384, len(CLASS_MAP))

    def forward(self, x):
        # This is for the MLP config, subclass this class for the regressor
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def predict(self, x, tensor=True):
        """Return nd array of class predictions for input.
        Set tensor=True to return a tensor instead."""
        prob = self(x)
        if tensor:
            return torch.argmax(prob, dim=1)
        else:
            return torch.argmax(prob, dim=1).cpu().numpy()

    def predict_dl(self, data_loader=None):
        """Same as predict, but on a torch data loader."""
        if data_loader is None:
            data_loader = self.val_dataloader()

        result = torch.cat([self.predict(b.to(self.device)) for b, _ in data_loader]).cpu().numpy()

        return result

    def validation_step(self, batch, batch_idx):
        """Compute validation prediction."""
        x, y = batch
        pred = self.predict(x, tensor=True)
        return pred, y

    def validation_epoch_end(self, outputs):
        """Report validation accuracy over the whole validation split."""
        pred, gt = zip(*outputs)
        pred = torch.cat(pred).cpu().numpy()
        gt = torch.cat(gt).cpu().numpy()

        acc = balanced_accuracy_score(gt, pred)
        print(f'\nValidation accuracy : {acc:.5f}\n')
        self.log('val_acc', acc)

    def train_dataloader(self):
        path = os.path.join('..', 'data', 'dt_sim')
        x, y = torch.load(os.path.join(path, 'z_train.pt')), torch.load(os.path.join(path, 'y_train.pt')).long()
        return DataLoader(TensorDataset(x, y), batch_size=256, shuffle=True)

    def val_dataloader(self):
        path = os.path.join('..', 'data', 'dt_sim')
        x, y = torch.load(os.path.join(path, 'z_val.pt')), torch.load(os.path.join(path, 'y_val.pt')).long()
        return DataLoader(TensorDataset(x, y), batch_size=256, shuffle=False)


class LitSegReg(LitSegMLP):
    """Same as above, but only one linear layer."""

    def __init__(self):
        super().__init__(mlp=False)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.log_softmax(x, dim=1)
        return x


def train_nn(epochs=2):
    """Train neural network for max epochs and save checkpoint. Return model."""
    model = LitSegMLP()
    trainer = Trainer(gpus=1,
                      max_epochs=epochs,
                      val_check_interval=.5,
                      callbacks=[EarlyStopping(monitor="val_acc", mode='max')])
    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(RESULTS_PATH, 'seg_NN.pt'))


def train_reg(epochs=2):
    """Train linear regressor for max epochs and save checkpoint. Return model."""
    model = LitSegReg()
    trainer = Trainer(gpus=1,
                      max_epochs=epochs,
                      val_check_interval=.5,
                      callbacks=[EarlyStopping(monitor="val_acc", mode='max')])
    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(RESULTS_PATH, 'seg_reg.pt'))


def load_mlp(device='cuda:0'):
    return LitSegMLP().load_from_checkpoint(os.path.join(RESULTS_PATH, 'seg_NN.pt')).to(device)


def load_reg(device='cuda:0'):
    return LitSegReg().load_from_checkpoint(os.path.join(RESULTS_PATH, 'seg_reg.pt')).to(device)


if __name__ == '__main__':
    # Note : you may need to run this step by step (prep data then run or clear GPU memory, then train)
    np.random.seed(42)

    # Make sure to run this before training classifiers
    prepare_seg_dataset(grayscale=False)

    # Fit MLP and linear regressor to train set
    # Note : they monitor the validation set for early stopping
    train_nn(100)
    train_reg(100)

    # KNN prediction
    x_train, y_train = torch.load(os.path.join(DATA_PATH, 'z_train.pt')).to('cuda'), torch.load(
        os.path.join(DATA_PATH, 'y_train.pt')).to('cuda')
    x_val, y_val = torch.load(os.path.join(DATA_PATH, 'z_val.pt')).to('cuda'), torch.load(os.path.join(DATA_PATH, 'y_val.pt')).to(
        'cuda')

    # KNN prediction
    _, _, pred_knn = knn_classifier(x_train, y_train, x_val, y_val, 20, 0.07, len(CLASS_MAP))

    # Linear prediction
    reg = load_reg()
    pred_reg = reg.predict_dl()  # Computed on validation set by default

    # MLP prediction
    nn = load_mlp()
    pred_nn = nn.predict_dl()  # Computed on validation set by default

    # Save predictions in a data frame
    # Keep raw predictions to compute desired metrics afterwards
    results = pd.DataFrame.from_dict(dict(ground_truth=y_val.cpu().numpy(),
                                          pred_nn=pred_nn,
                                          pred_knn=pred_knn,
                                          pred_reg=pred_reg))
    results.to_pickle(os.path.join(RESULTS_PATH, 'val_pred.pkl'))
