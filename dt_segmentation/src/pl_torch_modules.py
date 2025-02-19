"""Code to fit classifiers on the duckietown simulation segmentation dataset.

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

from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as pth_transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from .dt_utils import get_dino, get_dino_cnn


def get_transforms(resolution=480):
    """Return basic transforms (no augmentations). Use this for testing and inference."""
    t = [
        A.Resize(resolution, resolution),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    return A.Compose(t)


def get_augmented_transforms():
    """Transforms with augmentations."""
    t = [
        A.Resize(480, 480),
        A.RandomResizedCrop(480, 480, scale=(.25, 1), ratio=(.9, 1.1), p=.75),
        A.ShiftScaleRotate(shift_limit=0.4, scale_limit=0.1, rotate_limit=15, p=0.25),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.5, p=.5),
        A.GaussianBlur(blur_limit=(3, 41), p=.25),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    return A.Compose(t)


class DuckieSegDataset(Dataset):
    """Wrapper to get the image/label dataset of duckietown."""

    def __init__(self, path, augmented=False, resolution=480):
        self.path = path
        self.files = glob.glob(os.path.join(path, 'JPEGImages', "*.jpg"))
        self.len = len(self.files)
        self.augmented = augmented
        self.resolution = resolution
        mask_size = self.resolution //8
        self.mask_resize = pth_transforms.Resize(size=(mask_size, mask_size), interpolation=pth_transforms.InterpolationMode.NEAREST)
        if augmented:
            self.t = get_augmented_transforms()
        else:
            self.t = get_transforms(self.resolution)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Get image
        f = self.files[idx]
        with open(f, 'rb') as file:
            img = Image.open(file)
            x = np.array(img.convert('RGB'))

        # Get labels
        file_name = self.files[idx].split(os.sep)[-1][:-4]
        f = os.path.join(self.path, 'SegmentationClass', file_name + '.npy')
        y = np.load(f)

        # Resize to the shape of the DINO token output and flatten
        transformed = self.t(image=x, mask=y)

        image, mask = transformed['image'], transformed['mask']
        #
        # if self.augmented:
        #     plt.imshow(image.permute((1, 2, 0)))
        #     plt.show()
        #     exit()


        # Resize the mask to match the Vision transformer number of tokens
        mask = self.mask_resize(mask.unsqueeze(0)).flatten()

        return image, mask


class MLP(torch.nn.Module):
    """MLP for patch segmentation."""

    def __init__(self, n_classes, input_dim):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, 200)
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

    def __init__(self, data_path, write_path, class_names=None, head='linear', n_blocks=1,
                 batch_size=1, lr=1e-6, optimizer=AdamW, freeze_backbone=True, max_epochs=200, patience=10,
                 grayscale=False, n_classes=7, pretrain_on_sim=False, comet_logger=None, augmented=True, random_init=False,
                 backbone='vit'):
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
        self.pretrain_on_sim = pretrain_on_sim
        self.augmented = augmented
        self.random_init = random_init
        self.backbone = backbone
        self.mlp_input_dim = None

        # Vanilla transforms. Handy for inference
        # You can change resolution with the set_resolution method
        self.resolution = 480
        self.transforms = get_transforms(self.resolution)

        # First load dino to "cpu"
        if self.backbone == 'vit':
            dino = get_dino(8, device='cpu')

            # Only keep n_blocks
            dino.blocks = dino.blocks[:n_blocks]
            self.dino = dino
            self.mlp_input_dim = 384

            # Random init for the backbone
            if random_init:
                self.dino.apply(self.dino._init_weights)
        elif self.backbone == 'cnn1':
            # Shallow CNN. Rough benchmark for ViT 1-block
            dino = get_dino_cnn(device='cpu')

            # Only keep the first layers
            dino = nn.Sequential(dino.conv1, dino.bn1, dino.relu, dino.maxpool, dino.layer1, dino.layer2)

            # Remove the last activation
            dino[-1][-1].relu = nn.Identity()

            self.dino = dino
            self.mlp_input_dim = 512
        elif self.backbone == 'cnn2':
            # Deeper CNN. Rough benchmark for ViT 3-block
            dino = get_dino_cnn(device='cpu')
            dino = torch.nn.Sequential(dino.conv1, dino.bn1, dino.relu, dino.maxpool, dino.layer1, dino.layer2,
                                       dino.layer3[0], dino.layer3[1])

            # Remove the last activation
            dino[-1].relu = nn.Identity()

            self.dino = dino

            # On 480x480 data, this backbone will output a (1024, 30, 30) feature map. We add an upconv block
            # to upsample back to (512, 60, 60), then a conv layer to (384, 60, 60)
            self.upconv = nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=2, output_padding=1)
            self.relu = nn.ReLU(inplace=True)

            # Additional conv layer without activation (should not affect the feature map size)
            self.conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.mlp_input_dim = 512



        # Load segmentation head
        if head == 'linear':
            self.clf = Linear(self.n_classes)
        elif head == 'mlp':
            self.clf = MLP(self.n_classes, input_dim=self.mlp_input_dim)

        # Save hyperparameters to checkpoint
        self.save_hyperparameters()

        # Paths to real train data
        self.train_path = os.path.join(data_path, 'dt_real_voc_train')
        self.val_path = os.path.join(data_path, 'dt_real_voc_val')
        self.test_path = os.path.join(data_path, 'dt_real_voc_test')

        # Paths to sim data for finetuning
        self.train_path_sim = os.path.join(data_path, 'dt_sim_voc_train')
        self.val_path_sim = os.path.join(data_path, 'dt_sim_voc_val')
        self.test_path_sim = os.path.join(data_path, 'dt_sim_voc_test')

        self.write_path = write_path

    def forward(self, x):
        # DINO + Patch segmentation head
        if self.backbone == 'vit':
            # Remove CLS token
            x = self.dino(x)[:, 1:]
        else:
            x = self.dino(x)
            if self.backbone == 'cnn2':
                x = self.upconv(x)
                x = self.relu(x)
                x = self.conv(x)
            x = x.permute((0, 2, 3, 1))

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

    def set_resolution(self, resolution=480):
        if resolution % 8 != 0:
            raise ValueError('Resolution should be a multiple of 8.')
        self.transforms = get_transforms(resolution)
        self.resolution = resolution

    def predict(self, x):
        """Run inference on a single image.

        Parameters
        ----------
        x : PIL.Image
            Image to process.

        Returns
        -------
        predictions : ndarray
            480x480 segmentation of the image.

        """
        with torch.no_grad():
            x = self.transforms(image=np.array(x))['image']
            prob = self(x.unsqueeze(0).to(self.device))

            output_size = self.resolution // 8
            low_res = torch.argmax(prob, dim=-1).cpu().numpy().reshape((output_size, output_size))

            high_res_patch_size = 480 // output_size
            pred = np.kron(low_res, np.ones((high_res_patch_size, high_res_patch_size), dtype=int))

            return pred

    def validation_step(self, batch, batch_idx):
        """Compute validation prediction."""
        x, y = batch
        probs = self(x).detach().cpu()
        pred = probs.argmax(dim=-1).detach().cpu()
        y = y.reshape((-1,)).long()
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

    def train_dataloader(self, sim=False):
        path = self.train_path_sim if sim else self.train_path
        data = DuckieSegDataset(path, augmented=self.augmented)

        # We use a sampler to make sure we have 1000 images per epoch, regardless of the dataset we are using
        sampler = torch.utils.data.WeightedRandomSampler(torch.ones((len(data),)), num_samples=1000, replacement=True)

        return DataLoader(data, batch_size=self.batch_size, num_workers=24, sampler=sampler)

    def val_dataloader(self, sim=False):
        path = self.val_path_sim if sim else self.val_path
        return DataLoader(DuckieSegDataset(path, augmented=False), batch_size=self.batch_size,
                          shuffle=False, num_workers=3)

    def test_dataloader(self):
        return DataLoader(DuckieSegDataset(self.test_path, augmented=False), batch_size=self.batch_size,
                          shuffle=False, num_workers=3)

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
            # EarlyStopping(
            #     monitor="val_acc",
            #     mode='max',
            #     patience=self.patience)
        ]

        if self.pretrain_on_sim:
            print('Pretraining on simulation data...')
            # Pretrain on sim, but don't log
            data_sim_train = self.train_dataloader(sim=True)
            data_sim_val = self.val_dataloader(sim=False)
            trainer = Trainer(gpus=1,
                              max_epochs=self.max_epochs,
                              check_val_every_n_epoch=1,
                              callbacks=callbacks,
                              logger=None)
            trainer.fit(self, train_dataloader=data_sim_train, val_dataloaders=data_sim_val)

        # Main training
        # PL callbacks
        callbacks = [
            ModelCheckpoint(
                monitor='val_acc',
                mode='max',
                dirpath=self.write_path,
                filename=ck_file_name,
                auto_insert_metric_name=False),
            # EarlyStopping(
            #     monitor="val_acc",
            #     mode='max',
            #     patience=self.patience)
        ]
        trainer = Trainer(gpus=1,
                          max_epochs=self.max_epochs,
                          check_val_every_n_epoch=1,
                          callbacks=callbacks,
                          logger=self.comet_logger)
        trainer.fit(self)

        # Also test!
        trainer.test(self)

        # Best checkpoint path
        self.best_ck = callbacks[0].best_model_path

        # If we have a logger, log the checkpoint
        if self.comet_logger is not None:
            self.comet_logger.experiment.log_asset(self.best_ck)

    def freeze_bb(self):
        for p in self.dino.parameters():
            p.requires_grad = False

    def unfreeze_bb(self):
        for p in self.dino.parameters():
            p.requires_grad = True
