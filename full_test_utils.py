import os
import numpy as np

from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

import re
from full_utils import *
from PIL import Image, ImageOps
# from full_plotting import *

import pandas as pd
import csv


from sam_model import get_bounding_boxes, SAM



def crop_and_resize(img):
    # Crop the image into left and right halves
    mid = img.width // 2
    left_half = img.crop((0, 0, mid, img.height))
    # Adjust the start of the right half if the width is not divisible by 2
    right_half_start = mid if img.width % 2 == 0 else mid + 1
    right_half = img.crop((right_half_start, 0, img.width, img.height))
    # Resize each half to 256x256
    left_resized = left_half.resize((256, 256))
    right_resized = right_half.resize((256, 256))
    return left_resized, right_resized


def transform_img_split(resize, totensor, normalize):
    options = []

    if resize:
        options.append(transforms.Lambda(crop_and_resize))

    if totensor:
        # Adjust to handle a pair of images (left and right halves)
        options.append(transforms.Lambda(lambda imgs: (transforms.ToTensor()(imgs[0]), transforms.ToTensor()(imgs[1]))))
        
    if normalize:
        # Normalize each image in the pair
        options.append(transforms.Lambda(lambda imgs: (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[0]), 
                                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgs[1]))))
        
    transform = transforms.Compose(options)
    return transform



def make_dataset(dir, gt=False):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    f = dir.split('/')[-1].split('_')[-1]
    print (dir, len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]))
    for root, dirs, files in os.walk(dir):
        for file in files:
            if 'checkpoint' not in file:
                path = os.path.join(dir, file)
                images.append(path)
    return images


def align_image_and_label_paths(image_paths, label_paths):
    def extract_identifier(path):
        # Extracts the filename without extension for comparison
        return os.path.splitext(os.path.basename(path))[0]

    # Sort both lists based on the extracted identifier
    image_paths_sorted = sorted(image_paths, key=extract_identifier)
    label_paths_sorted = sorted(label_paths, key=extract_identifier)
    print(image_paths_sorted)
    print('\n')
    print(label_paths_sorted)
    return image_paths_sorted, label_paths_sorted



def write_dice_scores_to_csv(storage, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row, assuming all dictionaries have the same keys
        if storage:
            headers = ['Image Name'] + [f'Dice Score Class {cls}' for cls in range(len(storage[0]) - 1)]
            writer.writerow(headers)
            
            # Write each set of Dice scores to the CSV file
            for record in storage:
                row = [record["image_name"]] + [record[f'Dice Score Class {cls}'] for cls in range(len(record) - 1)]
                writer.writerow(row)


    

def visualize_and_dice(predictions, targets, image_names, storage, dataset=None):
    """Visualize and save predicted and ground truth segmentation maps."""
    for idx in range(len(predictions)):
        pred_image = predictions[idx]
        gt_image = targets[idx]

        gt_image = np.squeeze(gt_image)
        gt_image = gt_image * 255

        dice_scores = dice_coefficient(pred_image, gt_image)

        # Store the dice scores along with the image name
        storage.append({
            "image_name": image_names[idx],
            **dice_scores
        })
        
    return storage


def dice_coefficient(pred, target, num_classes=6):
    """Compute the Dice score, handling edge cases where classes may be missing."""
    
    temp_pred = np.copy(pred)
    temp_target = np.copy(target)
    
    dice_scores = {}
    for cls in range(num_classes):  # Iterate over each class
        pred_cls = (temp_pred == cls)
        target_cls = (temp_target == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        if union == 0:
            # If both pred and target don't have this class, it's a perfect match.
            dice_score = 1.0
        else:
            dice_score = 2 * intersection / (union + 1e-6)
        
        dice_scores[f'Dice Score Class {cls}'] = dice_score
        
    return dice_scores

