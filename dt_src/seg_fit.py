"""Script to fit classifiers on the duckietown simulation segmentation dataset.

Train data should be organized in the folder data/torch/train.
Val data should be organized in the folder data/torch/val.
Test data should be organized in the folder data/torch/test.

All results are saved in the results folder.
"""
import torch
import os
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from PIL import Image

from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as pth_transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dt_src.dt_utils import DATA_PATH, RESULTS_PATH, CLASS_MAP, prepare_seg_dataset, get_dino
from eval_knn import knn_classifier


class DuckieSegDataset(Dataset):
    """Wrapper to get the image/label dataset of duckietown."""

    def __init__(self, split, grayscale=False):
        self.path = os.path.join('..', 'data', 'torch', split)
        files = [f for f in os.listdir(self.path) if f.endswith('x.png')]
        self.len = len(files)
        t = [pth_transforms.Grayscale()] if grayscale else []
        t += [
            pth_transforms.Resize((480, 480)),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
        self.t = pth_transforms.Compose(t)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        f = os.path.join(self.path, f'{idx}_x.png')
        with open(f, 'rb') as file:
            img = Image.open(file)
            x = img.convert('RGB')
        y = torch.load(os.path.join(self.path, f'{idx}_y.pt'))
        return self.t(x), y


class MLP(torch.nn.Module):
    """MLP for patch segmentation."""

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(384, 200)
        self.layer_2 = nn.Linear(200, 100)
        self.layer_3 = nn.Linear(100, len(CLASS_MAP))

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x


class Linear(torch.nn.Module):
    """Linear classifier for patch segmentation."""

    def __init__(self, mlp=True):
        super().__init__()
        # Linear regressor
        self.layer_1 = nn.Linear(384, len(CLASS_MAP))

    def forward(self, x):
        x = self.layer_1(x)
        x = F.log_softmax(x, dim=1)
        return x


class DINOSeg(pl.LightningModule):
    """DINO + Segmentation Head"""

    def __init__(self, n_blocks, head='linear', batch_size=1, lr=1e-6, optimizer=AdamW, freeze_backbone=True, max_epochs=200, grayscale=False):
        super().__init__()
        self.n_blocks = n_blocks
        self.head = head
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.freeze_backbone = freeze_backbone
        self.max_epochs = max_epochs
        self.grayscale = grayscale

        # First load dino to "cpu"
        dino = get_dino(8, device='cpu')

        # Only keep n_blocks
        dino.blocks = dino.blocks[:n_blocks]
        self.dino = dino

        # Load segmentation head
        if head == 'linear':
            self.clf = Linear()
        elif head == 'mlp':
            self.clf = MLP()

        # Save hyperparameters to checkpoint
        self.save_hyperparameters()

    def forward(self, x):
        # DINO + Patch segmentation head
        x = self.dino(x)[:, 1:]

        # Put all patches on same axis
        x = x.reshape((-1, x.shape[-1]))
        x = self.clf(x)
        return x

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        y = y.reshape((-1,)).long()
        loss = F.nll_loss(probs, y)
        return loss

    def predict(self, x, tensor=True):
        """Return nd array of class predictions for input.
        Set tensor=True to return a tensor instead."""
        prob = self(x)
        if tensor:
            return torch.argmax(prob, dim=-1)
        else:
            return torch.argmax(prob, dim=-1).cpu().numpy()

    def predict_dl(self, data_loader):
        """Same as predict, but on a torch data loader."""
        device = self.device
        self.to('cuda')
        self.eval()
        with torch.no_grad():
            result = torch.cat([self.predict(b.to(self.device)).cpu() for b, _ in data_loader]).numpy()
        self.to(device)

        return result

    def validation_step(self, batch, batch_idx):
        """Compute validation prediction."""
        x, y = batch
        pred = self.predict(x, tensor=True)
        return pred, y.squeeze(0)

    def validation_epoch_end(self, outputs):
        """Report validation accuracy over the whole validation split."""
        pred, gt = zip(*outputs)
        pred = torch.cat(pred).cpu().numpy().flatten()
        gt = torch.cat(gt).cpu().numpy().flatten()

        acc = balanced_accuracy_score(gt, pred)
        print(f'\nValidation accuracy : {acc:.5f}\n')
        self.log('val_acc', acc)

    def train_dataloader(self):
        return DataLoader(DuckieSegDataset(split='train', grayscale=self.grayscale), batch_size=self.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self):
        return DataLoader(DuckieSegDataset(split='val', grayscale=self.grayscale), batch_size=self.batch_size, shuffle=False, num_workers=12)

    def test_dataloader(self):
        return DataLoader(DuckieSegDataset(split='test', grayscale=self.grayscale), batch_size=self.batch_size, shuffle=False, num_workers=12)

    def fit(self, file_name=None):
        if self.freeze_backbone:
            self.freeze_bb()
        else:
            self.unfreeze_bb()

        trainer = Trainer(gpus=1,
                          max_epochs=self.max_epochs,
                          val_check_interval=.5,
                          callbacks=[EarlyStopping(monitor="val_acc", mode='max', patience=20)])
        trainer.fit(self)

        if file_name is None:
            file_name = self.head + ('_frozen' if self.freeze_backbone else '') + '.pt'
        trainer.save_checkpoint(os.path.join(RESULTS_PATH, file_name))

    def freeze_bb(self):
        for p in self.dino.parameters():
            p.requires_grad = False

    def unfreeze_bb(self):
        for p in self.dino.parameters():
            p.requires_grad = True


if __name__ == '__main__':
    BLOCKS = 1  # Number of transformer blocks to use in the backbone
    GRAYSCALE = False

    # Make sure to run this once before training classifiers
    prepare_seg_dataset()

    # Linear Head
    lin_frozen = DINOSeg(head='linear', freeze_backbone=True, optimizer=Adam, lr=1e-3, batch_size=6,
                         n_blocks=BLOCKS, max_epochs=1, grayscale=GRAYSCALE)
    lin_frozen.fit()
    pred_lin_frozen = lin_frozen.predict_dl(lin_frozen.test_dataloader())

    # # MLP Head
    mlp_frozen = DINOSeg(head='mlp', freeze_backbone=True, optimizer=Adam, lr=1e-3, batch_size=6,
                         n_blocks=BLOCKS, max_epochs=1, grayscale=GRAYSCALE)
    mlp_frozen.fit()
    pred_mlp_frozen = mlp_frozen.predict_dl(mlp_frozen.test_dataloader())

    # MLP Head + Fine tune backbone
    mlp_dino = DINOSeg.load_from_checkpoint(os.path.join(RESULTS_PATH, 'linear_frozen.pt'))
    mlp_dino.freeze_backbone = False
    mlp_dino.optimizer = AdamW
    mlp_dino.batch_size = 1
    mlp_dino.lr = 1e-6
    mlp_dino.fit()
    pred_mlp_dino = mlp_dino.predict_dl(mlp_dino.test_dataloader())

    # Get ground truth
    gt = torch.cat([y_i.flatten() for _, y_i in mlp_dino.test_dataloader()]).cpu().numpy()


    # Save results
    results = pd.DataFrame.from_dict(dict(ground_truth=gt,
                                          pred_dino=pred_mlp_dino,
                                          pred_nn=pred_mlp_frozen,
                                          pred_reg=pred_lin_frozen))
    results.to_pickle(os.path.join(RESULTS_PATH, 'test_pred.pkl'))
