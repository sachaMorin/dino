#!/usr/bin/env python
"""Minimal script to fit DINO backbone for Duckietown segmentation."""
import comet_ml
import os
import argparse

import numpy as np
import torch
from torch.optim import Adam, AdamW
from pytorch_lightning.loggers import CometLogger
from src.dt_utils import parse_class_names

from src.pl_torch_modules import DINOSeg
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def run_experiment(data_path, write_path, batch_size, epochs, learning_rate, n_blocks, finetune, unfreeze=False, random_init=False,
                   augmentations=False, pretrain_on_sim=False, ck_file_name=None, comet_tag=None, random_state=42, patience=10, backbone='vit', optimizer='adam'):
    """Fit coarse segmentation model on Duckietown data. We use DINO as the backbone and output a prediction for
    every 8x8 token in the image.

    Parameters
    ----------
    data_path: str,
        Path to source the data.
    write_path: str,
        Where to save predictions and model checkpoints.
    batch_size: int,
        Batch size for training.
    epochs : int,
        Max epochs for training.
    learning_rate : float,
        Learning rate.
    optimizer : str in {"adam", "adamw"},
        Optimizer to use.
    patience : int,
        Patience for early stopping. Currently ignored.
    backbone : str in {"vit", "cnn"},
        Use DINO ViT or CNN (ResNet50) pretrained weights for the backbone.
    n_blocks : n_blocks,
        Number of DINO blocks to use in the ViT backbone. Should range from 1 to 12
    finetune : bool,
        Initial training is done on the frozen backbone. If this is set to true, we then unfreeze the
        backbone and refit the data.
    freeze : bool,
        Train with a frozen backbone.
    pretrain_on_sim : bool,
        Pretrain on the larger simulation dataset before training on real data.
    random_init : bool,
        Use random initialization instead of DINO pretrained weights.
    augmentations : bool,
        Train on augmentations.
    random_state : int,
        Random seed.
    ck_file_name : str, default:None
        Name of the checkpoint and prediction file names.
    comet_tag : str, default:None
        If a comet tag is provided we log the experiments to comet with the provided tag.
    """
    # Seed experiments
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Initialize comet logger if requested
    if comet_tag is not None:
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name="duck"
        )
        comet_logger.experiment.add_tag(comet_tag)
        comet_logger.experiment.log_parameter("random_state", random_state)
    else:
        comet_logger = None

    # Get class names and length
    class_names, _ = parse_class_names(os.path.join(data_path, 'labels.txt'))

    # Optimizer
    if optimizer == 'adam':
        optimizer = Adam
    elif optimizer == 'adamw':
        optimizer = AdamW

    # MLP Head
    dino_seg = DINOSeg(head='mlp', data_path=data_path, pretrain_on_sim=pretrain_on_sim,
                         write_path=write_path, n_classes=len(class_names), class_names=class_names,
                         freeze_backbone=not unfreeze, optimizer=optimizer, lr=learning_rate, batch_size=batch_size,
                         n_blocks=n_blocks, max_epochs=epochs, patience=patience, comet_logger=comet_logger,
                         augmented=augmentations, random_init=random_init, backbone=backbone)

    if ck_file_name is None:
        # Generate a checkpoint file name
        ck_file_name = str(n_blocks) + '_' + f'{backbone}_mlp_' + str(random_state)

    dino_seg.fit(ck_file_name)

    # Fine tune
    # This is logged as a separate comet experiment
    if finetune:
        # Initialize comet logger if requested
        if comet_tag is not None:
            comet_logger = CometLogger(
                api_key=os.environ.get("COMET_API_KEY"),
                project_name="duck"
            )
            comet_logger.experiment.add_tag(comet_tag)
            comet_logger.experiment.log_parameter("is_finetuned", True)
        else:
            comet_logger = None

        print("\n Finetuning the previous model...")
        # MLP Head + Fine tune backbone
        # Only finetune with fewer than 5 blocks, might otherwise run out of GPU RAM
        dino_seg = DINOSeg.load_from_checkpoint(dino_seg.best_ck)
        dino_seg.freeze_backbone = False
        dino_seg.optimizer = optimizer
        # dino_seg.lr /= 100  # Lower the learning rate for fine-tuning

        # Add new comet logger
        dino_seg.comet_logger = comet_logger
        ck_file_name = ck_file_name + '_finetuned'
        dino_seg.fit(ck_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_path", '-d', help="Data folder", required=False, default="data")
    parser.add_argument("--write_path", '-w', help="Where to write results", required=False, default="results")
    parser.add_argument("--batch_size", '-b', help="Batch size. Number of 480p images. 1 image = 3,600 image patches.", required=False, default=1, type=int)
    parser.add_argument("--epochs", '-e', help="Max number of training epochs", required=False, default=200, type=int)
    parser.add_argument("--learning_rate", '-lr', help="Learning rate", required=False, default=1e-3, type=float)
    parser.add_argument("--optimizer", '-op', help="Optimizer", required=False, default="adam", type=str)
    parser.add_argument("--patience", '-p', help="Patience for early stopping (Not implemented).", required=False, default=200, type=int)
    parser.add_argument("--backbone", '-ba', help="Use ViT or Resnet50 backbone.", required=False, default="vit", type=str)
    parser.add_argument("--n_blocks", help="Number of DINO blocks to use", required=False, default=1, type=int)
    parser.add_argument("--pretrain_on_sim", help="Pretrain on simulation data.", required=False, action='store_true')
    parser.add_argument("--finetune",
                        help="Finetune DINO backbone after an initial training phase with a frozen backbone",
                        required=False, action='store_true')
    parser.add_argument("--unfreeze",
                        help="Unfreeze DINO backbone during training. If you want to first train with a frozen backbone and then unfrezze, use the --finetune flag.",
                        required=False, action='store_true')
    parser.add_argument("--random_init",
                        help="Reinitialize the weights instead of using pretrained DINO weidghts.",
                        required=False, action='store_true')
    parser.add_argument("--augmentations",
                        help="Augment data during training.",
                        required=False, action='store_true')
    parser.add_argument("--comet_tag",
                        help=" If a comet tag is provided we log the experiments to comet with the provided tag.",
                        required=False, default=None, type=str)
    parser.add_argument("--random_state", help="Random seed", required=False, default=42, type=int)
    args = parser.parse_args()

    run_experiment(**vars(args))
