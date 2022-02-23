"""Process output of pl_torch_modules.py to show performance metrics.

Deprecated do not use."""
import os
import pandas as pd
from dt_segmentation.dt_utils import CLASS_MAP
from sklearn.metrics import jaccard_score, balanced_accuracy_score, confusion_matrix
import numpy as np
pd.options.display.float_format = '{:,.3f}'.format

results = pd.read_pickle(os.path.join('..', 'results', 'test_pred_4_grayscale.pkl'))

iou = dict( reg=list(), nn=list(), dino=list())

gt = results['ground_truth']

for key, l in iou.items():
    pred = results[f'pred_{key}']

    iou_m = list()

    print('\n\nConfusion matrix for ' + key)
    names = [t[0] for t in CLASS_MAP]
    mx = pd.DataFrame(confusion_matrix(gt, pred), columns=names, index=names)
    mx = mx.div(mx.sum(axis=1), axis=0) * 100

    print(mx.to_string())
    # Average class IOU
    for i in range(len(CLASS_MAP)):
        pred_i = pred == i
        gt_i = gt == i
        jacc = jaccard_score(pred_i, gt_i)
        iou_m.append(jacc)
    l.append(np.mean(iou_m))
    l.append(balanced_accuracy_score(gt, pred))

metrics = pd.DataFrame.from_dict(iou)
print('\n\n Avg. class IoU and Accuracy')
metrics.index = ['IoU', 'Acc.']
print(metrics.transpose().to_string())
