#!/usr/bin/env python
"""Minimal script to fit DINO backbone for Duckietown segmentation."""
import comet_ml
import os
import pandas as pd
import argparse

import numpy as np
import torch
from torch.optim import Adam, AdamW, SGD
from pytorch_lightning.loggers import CometLogger
from labelme2voc import parse_class_names

from dt_segmentation import DINOSeg
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def run_experiment(data_path, write_path, batch_size, epochs, learning_rate, patience, n_blocks, finetune, seed,
                   ck_file_name=None, comet_tag=None):
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
    patience : int,
        Patience for early stoppping.
    n_blocks : n_blocks,
        Number of DINO blocks to use in the backbone. Should range from 1 to 12
    finetune : bool,
        Initial training is done on the frozen backbone. If this is set to true, we then unfreeze the
        backbone and refit the data.
    seed : int,
        Random seed.
    ck_file_name : str, default:None
        Name of the checkpoint and prediction file names.
    comet_tag : str, default:None
        If a comet tag is provided we log the experiments to comet with the provided tag.
    """
    # Seed experiments
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Fit DINO backbone for duckie segmentation
    train_path = os.path.join(data_path, 'dt_real_voc_train')
    val_path = os.path.join(data_path, 'dt_real_voc_val')
    test_path = os.path.join(data_path, 'dt_real_voc_test')

    # Initialize comet logger if requested
    if comet_tag is not None:
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name="duck"
        )
        comet_logger.experiment.add_tag(comet_tag)
    else:
        comet_logger = None

    # Get class names and length
    class_names, _ = parse_class_names(os.path.join(data_path, 'labels.txt'))

    # Number of transformer blocks to use in the backbone
    # MLP Head
    mlp_frozen = DINOSeg(head='mlp', train_path=train_path, val_path=val_path, test_path=test_path,
                         write_path=write_path, n_classes=len(class_names), class_names=class_names,
                         freeze_backbone=True, optimizer=Adam, lr=learning_rate, batch_size=batch_size,
                         n_blocks=n_blocks, max_epochs=epochs, patience=patience, comet_logger=comet_logger)

    if ck_file_name is None:
        # Generate a checkpoint file name
        ck_file_name = str(n_blocks) + '_' + 'mlp_frozen_' + str(seed)

    mlp_frozen.fit(ck_file_name)
    pred_mlp_frozen = mlp_frozen.predict_dl(mlp_frozen.test_dataloader())

    # Get ground truth
    gt = torch.cat([y_i.flatten() for _, y_i in mlp_frozen.test_dataloader()]).cpu().numpy()

    # Save results
    # We save ground truth and predictions so we can recompute metrics later on
    results = pd.DataFrame.from_dict(dict(ground_truth=gt,
                                          pred_mlp_frozen=pred_mlp_frozen))

    # Fine tune, we don't do this for now
    if finetune:
        # Initialize comet logger if requested
        if comet_tag is not None:
            comet_logger = CometLogger(
                api_key=os.environ.get("COMET_API_KEY"),
                project_name="duck"
            )
            comet_logger.experiment.add_tag(comet_tag)
        else:
            comet_logger = None

        print("\n Finetuning the previous model...")
        # MLP Head + Fine tune backbone
        # Only finetune with fewer than 5 blocks, might otherwise run out of GPU RAM
        mlp_dino = DINOSeg.load_from_checkpoint(os.path.join(write_path, ck_file_name + '.ckpt'))
        mlp_dino.freeze_backbone = False
        mlp_dino.optimizer = AdamW
        mlp_dino.lr /= 2  # Lower the learning rate for fine-tuning

        # Add new comet logger
        mlp_dino.comet_logger = comet_logger
        ck_file_name = ck_file_name + '_finetuned'
        mlp_dino.fit(ck_file_name)
        results["pred_mlp_finetuned"] = mlp_dino.predict_dl(mlp_dino.test_dataloader())

    file_name = ck_file_name + '.pkl'
    results.to_pickle(os.path.join(write_path, file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_path", '-d', help="Data folder", required=False, default="data")
    parser.add_argument("--write_path", '-w', help="Where to write results", required=False, default="results")
    parser.add_argument("--batch_size", '-b', help="Batch size", required=False, default=2, type=int)
    parser.add_argument("--epochs", '-e', help="Max number of training epochs", required=False, default=200, type=int)
    parser.add_argument("--learning_rate", '-lr', help="Learning rate", required=False, default=1e-3, type=float)
    parser.add_argument("--patience", '-p', help="Patience for early stopping", required=False, default=200, type=int)
    parser.add_argument("--n_blocks", help="Number of DINO blocks to use", required=False, default=1, type=int)
    parser.add_argument("--finetune",
                        help="Finetune DINO backbone after an initial training phase with a frozen backbone",
                        required=False, default=False, type=bool)
    parser.add_argument("--comet_tag",
                        help=" If a comet tag is provided we log the experiments to comet with the provided tag.",
                        required=False, default=None, type=str)
    parser.add_argument("--seed", help="random_seed", required=False, default=42, type=int)
    args = parser.parse_args()

    run_experiment(**vars(args))
