#!/usr/bin/env python
"""Minimal script to fit DINO backbone for Duckietown segmentation."""
import os
import pandas as pd

import torch
from torch.optim import Adam, AdamW, SGD

from dt_segmentation import DINOSeg
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_PATH = os.path.join('', 'results')
train_path = os.path.join('', 'data', 'dt_real_voc_train')
val_path = os.path.join('', 'data', 'dt_real_voc_val')
test_path = os.path.join('', 'data', 'dt_real_voc_test')
write_path = os.path.join('', 'models')

# Main Segmentation experiment for our report
MAX_EPOCHS = 100

# Number of transformer blocks to use in the backbone
# MLP Head
mlp_frozen = DINOSeg(head='mlp', train_path=train_path, val_path=val_path, test_path=test_path,
                     write_path=write_path,
                     freeze_backbone=True, optimizer=Adam, lr=1e-3, batch_size=1,
                     n_blocks=2, max_epochs=MAX_EPOCHS)
mlp_frozen.fit()
pred_mlp_frozen = mlp_frozen.predict_dl(mlp_frozen.test_dataloader())

# Get ground truth
gt = torch.cat([y_i.flatten() for _, y_i in mlp_frozen.test_dataloader()]).cpu().numpy()

# Fine tune, we don't do this for now
# if blocks < 5:
#     # MLP Head + Fine tune backbone
#     # Only finetune with fewer than 5 blocks, might otherwise run out of GPU RAM
#     # Start from frozen linear checkpoint
#     ck_file_name = str(blocks) + '_' + 'mlp_frozen' + ('_grayscale' if grayscale else '') + '.ckpt'
#     mlp_dino = DINOSeg.load_from_checkpoint(os.path.join(RESULTS_PATH, ck_file_name))
#     mlp_dino.freeze_backbone = False
#     mlp_dino.optimizer = AdamW
#     mlp_dino.batch_size = 1
#     mlp_dino.lr = 1e-6
#     mlp_dino.fit()
#     pred_mlp_dino = mlp_dino.predict_dl(mlp_dino.test_dataloader())
# else:
#     # Dummy predictions for compatibility with the rest of the pipeline
#     mask = torch.randperm(pred_mlp_frozen.shape[0])
#     pred_mlp_dino = pred_mlp_frozen[mask]

# Save results
results = pd.DataFrame.from_dict(dict(ground_truth=gt,
                                      pred_nn=pred_mlp_frozen))
file_name = 'test_pred_' + str(1) + '.pkl'
results.to_pickle(os.path.join(RESULTS_PATH, file_name))
