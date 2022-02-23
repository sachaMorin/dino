#!/usr/bin/env python
"""Process duckietown simulation data to a VOC-style segmentation dataset following the same format as labelme2voc.py.
Adapted the labelme repo. See https://github.com/wkentaro/labelme/blob/main/examples/semantic_segmentation/labelme2voc.py.

This script assumes the raw sim data follows the following format :

input_dir
    images
        0.png
        1.png
        ...
    labels
        0.png
        1.png
        ...

Where the labels are the object renderings of the simulator. Those are imperfect and we process them accordingly
in this script."""

from __future__ import print_function

import shutil
import argparse
import glob
import os
import os.path as osp
import sys
from PIL import Image
import cv2

import imgviz
import numpy as np

import labelme

# Create class name/id/RGB associations
# First element is class name
# Second is class id
# Third element is the RGB color from simulator rendering. (Not perfect accurate for some class, see rgb_to_c_
# Fourth element is the displayed color for that class sin seg_viz.py
# The simulator likely has more class than the real data, so we will only take the intersection of class_map and
# the provided labels file
class_map = [
    ("_background_", 0, "000000", "000000"),
    ("yellow-lane", 1, "ffff00", "ffff00"),
    ("white-lane", 2, "ffffff", "df4f4f"),
    ("duckiebot", 3, "ad0000", "ad0000"),
    ("sign", 4, "4a4342", "00ff00"),
    ("duck", 5, "cfa923", "00ffff"),
    ("red-tape", 6, "fe0000", "fe0000"),
    ("cone", 7, "ffa600", "ffa600"),
    ("house", 8, "279621", "279621"),
    ("bus", 9, "ebd334", "ff00ff"),
    ("truck", 10, "961fad", "000099"),
    ("barrier", 11, "000099", "964b00"),
    # ("hand", 12, "000099", "964b00"),  # Not in simulator, we can safely ignore
]

# Convert hex codes to RGB values
def to_rgb(hex):
    return [int(hex[i:i + 2], 16) for i in (0, 2, 4)]


# Convert Hex to RGB
CLASS_MAP = [(m[0], m[1], to_rgb(m[2]), to_rgb(m[3])) for m in class_map]


def rgb_to_c(mask_img, raw_img, current_classes):
    """Map RGB pixels to class to create an (approximate) segmentation mask.
    Segmentation colors from the simulator are not perfect (e.g., yellow lanes often have an offset)
    so we still use HSV filters over the raw image for some classes.

    Args:
        mask_img(PIL): Simulation rendering of the image (approximately discrete colors).
        raw_img(PIL): Image of interest.

    Returns:
        (np.ndarray) : (image height, image width) array with class assignments.

    """
    mask_img = np.array(mask_img)
    raw_img = np.array(raw_img)
    raw_hsv = cv2.cvtColor(raw_img, cv2.COLOR_RGB2HSV)

    result = np.zeros(mask_img.shape[:-1], dtype='int')
    for m in CLASS_MAP[1:]:
        if m[0] in current_classes:
            if m[0] == 'duckiebot':
                # Also map wheel and camera to duckiebot class
                mask = (mask_img == m[2]) + (mask_img == [30, 12, 5])
                # Add backplate from the raw image
                mask += raw_img == [0, 0, 0]  # Pure black pixels
                mask = mask.all(axis=-1)

                # Get rest of the plate
                # Won't capture all backplates, but filter needs to be conservative to not capture white lanes/floor
                # Will cover some signs, but since we process signs after, the class in result should be mostly correct.
                # lower_rgb = np.array([88, 88, 88])
                # higher_rgb = np.array([95, 95, 95])
                # mask += cv2.inRange(raw_img, lower_rgb, higher_rgb) == 255
            elif m[0] == 'yellow-lane':
                # Yellow lanes are trickier so we use HSV filter
                lower_hsv = np.array([25, 60, 150])
                higher_hsv = np.array([30, 255, 255])
                mask = cv2.inRange(raw_hsv, lower_hsv, higher_hsv) == 255
                result[mask] = m[1]
            elif m[0] == 'red-tape':
                # Red tape is also tricky
                lower_hsv = np.array([175, 120, 0])
                higher_hsv = np.array([180, 255, 255])
                mask = cv2.inRange(raw_hsv, lower_hsv, higher_hsv) == 255
            elif m[0] == 'sign':
                # 3 types of pixel for signs
                mask = (mask_img == m[2]) + (mask_img == [52, 53, 8]) + (mask_img == [76, 71, 71])
                mask = mask.all(axis=-1)
            elif m[0] == 'white-lane':
                # Include some grey pixels in white lanes
                lower_hsv = np.array([0, 0, 145])
                higher_hsv = np.array([180, 40, 255])
                mask = cv2.inRange(raw_hsv, lower_hsv, higher_hsv) == 255
            elif m[0] == 'duck':
                # Duckie passengers have a different color. Add them as well.
                mask = (mask_img == m[2]) + (mask_img == [132, 108, 22])
                mask = mask.all(axis=-1)
            else:
                mask = (mask_img == m[2]).all(axis=-1)

            # We assign the class position in current classes
            # This ensures compatibility with the real data, which follows this convention
            result[mask] = current_classes.index(m[0])

    # Then we take the other classes and make sure they are mapped to 0
    # Important to do this after processing the positive classes. Some white HSV filters we use may cover buses
    # for example, and we correct that here by forcing the bus class to 0
    for m in CLASS_MAP[1:]:
        if m[0] not in current_classes:
            mask = mask_img == m[2]
            mask = mask.all(axis=-1)
            result[mask] = 0

    return result


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input smi data")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClassPNG"))
    if not args.noviz:
        os.makedirs(
            osp.join(args.output_dir, "SegmentationClassVisualization")
        )
    print("Creating dataset:", args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    for filename in glob.glob(osp.join(args.input_dir, 'images', "*.png")):
        print("Generating dataset from:", filename)


        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        out_lbl_file = osp.join(
            args.output_dir, "SegmentationClass", base + ".npy"
        )
        out_png_file = osp.join(
            args.output_dir, "SegmentationClassPNG", base + ".png"
        )
        if not args.noviz:
            out_viz_file = osp.join(
                args.output_dir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )

        # Save image as jpeg
        img = Image.open(filename)
        rgb_im = img.convert('RGB')
        rgb_im.save(out_img_file)

        # Get labels
        sim_mask = Image.open(osp.join(args.input_dir, 'labels', filename.split(os.sep)[-1]))
        sim_mask = sim_mask.convert('RGB')

        lbl = rgb_to_c(sim_mask, rgb_im, class_names)

        labelme.utils.lblsave(out_png_file, lbl)

        np.save(out_lbl_file, lbl)

        if not args.noviz:
            viz = imgviz.label2rgb(
                lbl,
                imgviz.rgb2gray(np.array(img)),
                font_size=15,
                label_names=class_names,
                loc="rb",
            )
            imgviz.io.imsave(out_viz_file, viz)


if __name__ == "__main__":
    main()