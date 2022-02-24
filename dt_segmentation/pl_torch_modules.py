"""Script to fit classifiers on the duckietown simulation segmentation dataset.

Train data should be organized in the folder data/torch/train.
Val data should be organized in the folder data/torch/val.
Test data should be organized in the folder data/torch/test.

All results are saved in the results folder.
"""
import numpy as np
import torch
import os
import glob
from sklearn.metrics import balanced_accuracy_score, jaccard_score, f1_score
from PIL import Image
import cv2

from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as pth_transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from .dt_utils import RESULTS_PATH, get_dino


class DuckieSegDataset(Dataset):
    """Wrapper to get the image/label dataset of duckietown."""

    def __init__(self, path, grayscale=False):
        self.path = path
        self.files = glob.glob(os.path.join(path, 'JPEGImages', "*.jpg"))
        self.len = len(self.files)
        t = [pth_transforms.Grayscale(num_output_channels=3)] if grayscale else []
        t += [
            pth_transforms.Resize((480, 480)),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
        self.t = pth_transforms.Compose(t)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Get image
        f = self.files[idx]
        with open(f, 'rb') as file:
            img = Image.open(file)
            x = img.convert('RGB')

        # Get labels
        file_name = self.files[idx].split(os.sep)[-1][:-4]
        f = os.path.join(self.path, 'SegmentationClass', file_name + '.npy')
        y = np.load(f)

        # Resize to the shape of the DINO token output and flatten
        y = cv2.resize(y, (60, 60), interpolation=cv2.INTER_NEAREST).flatten()

        return self.t(x), torch.from_numpy(y)


class MLP(torch.nn.Module):
    """MLP for patch segmentation."""

    def __init__(self, n_classes):
        super().__init__()
        self.layer_1 = nn.Linear(384, 200)
        self.layer_2 = nn.Linear(200, 100)
        self.layer_3 = nn.Linear(100, n_classes)

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

    def __init__(self, n_classes):
        super().__init__()
        # Linear regressor
        self.layer_1 = nn.Linear(384, n_classes)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.log_softmax(x, dim=1)
        return x


class DINOSeg(pl.LightningModule):
    """DINO + Segmentation Head"""

    def __init__(self, n_blocks, train_path, val_path, test_path, write_path, class_names=None, head='linear',
                 batch_size=1, lr=1e-6, optimizer=AdamW, freeze_backbone=True, max_epochs=200, patience=10,
                 grayscale=False, n_classes=7, comet_logger=None):
        super().__init__()
        self.n_blocks = n_blocks
        self.head = head
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.freeze_backbone = freeze_backbone
        self.max_epochs = max_epochs
        self.patience = patience
        self.grayscale = grayscale
        self.n_classes = n_classes
        self.comet_logger = comet_logger
        self.class_names = class_names

        # First load dino to "cpu"
        dino = get_dino(8, device='cpu')

        # Only keep n_blocks
        dino.blocks = dino.blocks[:n_blocks]
        self.dino = dino

        # Load segmentation head
        if head == 'linear':
            self.clf = Linear(self.n_classes)
        elif head == 'mlp':
            self.clf = MLP(self.n_classes)

        # Save hyperparameters to checkpoint
        self.save_hyperparameters()

        # Paths to data
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.write_path = write_path

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

        pred = probs.argmax(dim=-1).detach().cpu()
        return {"loss": loss, "pred": pred, "gt": y.squeeze(0), "probs": probs.detach().cpu()}

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
        # Put model on GPU
        device = self.device
        self.to('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Predict data loader
        self.eval()
        with torch.no_grad():
            result = torch.cat([self.predict(b.to(self.device)).cpu() for b, _ in data_loader]).numpy()

        # Return model to original device
        self.to(device)

        return result

    def validation_step(self, batch, batch_idx):
        """Compute validation prediction."""
        x, y = batch
        probs = self(x).detach().cpu()
        pred = probs.argmax(dim=-1)
        return {"pred": pred, "gt": y.squeeze(0), "probs": probs}

    def validation_epoch_end(self, outputs, prefix='val'):
        """Report various metrics over the whole validation split."""
        pred = torch.cat([x["pred"] for x in outputs]).cpu().numpy().flatten()
        gt = torch.cat([x["gt"] for x in outputs]).cpu().numpy().flatten()
        probs = torch.cat([x["probs"] for x in outputs]).cpu().numpy()

        # Log metrics
        acc = balanced_accuracy_score(gt, pred)
        f1 = f1_score(gt, pred, average='macro')
        iou = jaccard_score(gt, pred, average='macro')
        self.log(prefix + '_acc', acc, prog_bar=True)
        self.log(prefix + '_iou', iou, prog_bar=True)
        self.log(prefix + '_F1', f1, prog_bar=True)

        # Log confusion matrix
        # We don't log the confusion matrix over the train set to save time
        if self.comet_logger is not None and prefix != 'train':
            desired = np.zeros(probs.shape)
            desired[np.arange(desired.shape[0]), gt] = 1
            self.comet_logger.experiment.log_confusion_matrix(desired, probs, title=prefix, labels=self.class_names,
                                                              file_name=f"{prefix}_epoch_{self.current_epoch}.json")

        return {prefix + '_acc': acc, prefix + '_iou': iou, prefix + '_F1': f1}

    def test_step(self, batch, batch_idx):
        """Compute test prediction."""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """Report various metrics over the whole test split."""
        metrics = self.validation_epoch_end(outputs, prefix='test')
        return metrics

    def training_epoch_end(self, outputs):
        """Report various metrics over the whole train split."""
        metrics = self.validation_epoch_end(outputs, prefix='train')

    def train_dataloader(self):
        return DataLoader(DuckieSegDataset(self.train_path), batch_size=self.batch_size,
                          shuffle=True, num_workers=12)

    def val_dataloader(self):
        return DataLoader(DuckieSegDataset(self.val_path), batch_size=self.batch_size,
                          shuffle=False, num_workers=12)

    def test_dataloader(self):
        return DataLoader(DuckieSegDataset(self.test_path), batch_size=self.batch_size,
                          shuffle=False, num_workers=12)

    def fit(self, ck_file_name=None):
        if self.freeze_backbone:
            self.freeze_bb()
        else:
            self.unfreeze_bb()

        if ck_file_name is None:
            # Create checkpoint file name
            ck_file_name = str(self.n_blocks) + '_' + self.head \
                           + ('_frozen' if self.freeze_backbone else '_finetuned') \
                           + ('_grayscale' if self.grayscale else '')

        # PL callbacks
        callbacks = [
            ModelCheckpoint(
                monitor='val_acc',
                mode='max',
                dirpath=self.write_path,
                filename=ck_file_name,
                auto_insert_metric_name=False),
            EarlyStopping(
                monitor="val_acc",
                mode='max',
                patience=self.patience)
        ]

        trainer = Trainer(gpus=1,
                          max_epochs=self.max_epochs,
                          check_val_every_n_epoch=1,
                          callbacks=callbacks,
                          logger=self.comet_logger)
        trainer.fit(self)

        # Also test!
        trainer.test(self)

        # If we have a logger, log the checkpoint
        if self.comet_logger is not None:
            ck_path = os.path.join(self.write_path, ck_file_name + '.ckpt')
            self.comet_logger.experiment.log_asset(ck_path)

    def freeze_bb(self):
        for p in self.dino.parameters():
            p.requires_grad = False

    def unfreeze_bb(self):
        for p in self.dino.parameters():
            p.requires_grad = True