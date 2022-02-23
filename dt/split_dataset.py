#!/usr/bin/env python
"""Split the outputs of labelme2voc or sim2voc to train/test splits."""
import shutil
import glob
import os
import os.path as osp
import numpy as np
import argparse

np.random.seed(42)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input voc style dataset")
    parser.add_argument("--n_test", help="Number of images in test set", default=20, type=int,
                        required=False)
    parser.add_argument("--n_val", help="Number of images in val set", default=10, type=int,
                        required=False)
    args = parser.parse_args()

    for split in ["_train", "_test", "_val"]:
        os.makedirs(args.input_dir + split)
        for dir in ["JPEGImages", "SegmentationClass", "SegmentationClassPNG", "SegmentationClassVisualization"]:
            os.makedirs(osp.join(args.input_dir + split, dir))
            shutil.copy(osp.join(args.input_dir, 'class_names.txt'),
                        osp.join(args.input_dir + split, 'class_names.txt'))

    files = glob.glob(osp.join(args.input_dir, "JPEGImages", "*.jpg"))
    np.random.shuffle(files)

    for i, filename in enumerate(files):
        if i < args.n_test:
            split = '_test'
        elif i < args.n_test + args.n_val:
            split = '_val'
        else:
            split = '_train'

        file = filename.split(os.sep)[-1][:-4]  # This will be the jpg, we need to change the extenson
        for dir, ext in [("JPEGImages", 'jpg'), ("SegmentationClass", 'npy'), ("SegmentationClassPNG", 'png'),
                         ("SegmentationClassVisualization", 'jpg')]:
            file_i = file + '.' + ext
            source = osp.join(args.input_dir, dir, file_i)
            target = osp.join(args.input_dir + split, dir, file_i)
            shutil.copy(source, target)


if __name__ == "__main__":
    main()
