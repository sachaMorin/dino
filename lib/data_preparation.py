import os
import shutil
import numpy as np
import json
from collections import defaultdict
from typing import List
import cv2

from detectron2.structures import BoxMode


def train_test_split(dataset_path: str, test_size: float = 0.5):
    annotation_file = os.path.join(dataset_path, 'annotations')
    frames_path = os.path.join(dataset_path, 'frames')
    # Opening dataset
    with open(os.path.join(annotation_file, 'final_anns.json')) as f:
        data = json.load(f)
    # %%% Creating train and test folders %%%#
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    check_folder_train = os.path.isdir(train_dir)
    check_folder_test = os.path.isdir(test_dir)
    if not (check_folder_train and check_folder_test):
        os.makedirs(os.path.join(train_dir, 'annotations'))
        os.makedirs(os.path.join(train_dir, 'frames'))
        os.makedirs(os.path.join(test_dir, 'annotations'))
        os.makedirs(os.path.join(test_dir, 'frames'))
        print("Creating Train and Test folders...")
    else:
        print("Folders already created")
        return None
    num_elements = len(data.keys())
    number_samples = int(num_elements * test_size)
    data_keys = list(data.keys())
    np.random.shuffle(data_keys)
    test_indexes = data_keys[number_samples:-1]
    train_indexes = data_keys[:number_samples]
    print('Creating train set...')
    train_annotations = save_data(train_indexes, data, frames_path, os.path.join(train_dir, 'frames'))
    with open(os.path.join(train_dir, 'annotations', 'anns.json'), 'w') as f:
        json.dump(train_annotations, f)
    print('Creating test set...')
    test_annotations = save_data(test_indexes, data, frames_path, os.path.join(test_dir, 'frames'))
    with open(os.path.join(test_dir, 'annotations', 'anns.json'), 'w') as f:
        json.dump(test_annotations, f)


def save_data(indexes: List, data: dict, src_dir: str, trg_dir: str) -> dict:
    """
    Iterate over the given set of indexes and create train/test set based
    on the whole original dataset.

    @param indexes: indexes of the train/test set
    @param data: Dictionary with the ground truth data
    @param src_dir: Source path of images
    @param trg_dir: Path to store images
    @return: Dictionary with ground truth bounding boxes
    """
    data_dict = defaultdict(list)
    for image_id in indexes:
        img_path = os.path.join(src_dir, image_id)
        data_dict[image_id] = data[image_id]
        shutil.copy(img_path, trg_dir)
    return data_dict


def get_duckietown_dicts(dataset_path: str) -> List:
    """
    Arrange the annotations file as a list in COCO format.
    @param dataset_path: Path to the train/test set
    @return: List of elements in the dataset in COCO format
    """
    # Defining paths and variables
    annotation_file = os.path.join(dataset_path, 'annotations')
    frame_path = os.path.join(dataset_path, 'frames')
    # Loading dataset
    with open(os.path.join(annotation_file, 'anns.json')) as f:
        data = json.load(f)

    dataset_dicts = []
    for idx, image_id in enumerate(data.keys()):
        image_name = os.path.join(frame_path, image_id)
        record = {}
        height, width = cv2.imread(image_name).shape[:2]

        record["file_name"] = image_name
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for annotation in data[image_id]:
            obj_ann = {
                "bbox": [annotation['bbox'][0], annotation['bbox'][1],
                         annotation['bbox'][0] + annotation['bbox'][2],
                         annotation['bbox'][1] + annotation['bbox'][3]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": annotation['cat_id'] - 1,
                "iscrowd": 0
            }
            objs.append(obj_ann)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
