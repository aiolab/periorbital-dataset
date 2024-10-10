import numpy as np
from segment_anything import sam_model_registry, SamPredictor
# from tester import SAM_CHECKPOINT_PATH, SAM_ENCODER_VERSION, DEVICE  
import os
import torch
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn

import pandas as pd




def get_bounding_boxes(csv_path, image_name, dataset):
    df = pd.read_csv(csv_path)
    
    if dataset == 'ted':
        boxes = df.loc[df['2023_crop_named'] == image_name, ['L_iris_box', 'R_iris_box']]

    else:
        boxes = df.loc[df['new_crop_name'] == image_name, ['L_iris_box', 'R_iris_box']]
    
    if not boxes.empty:
        # Convert the string representation of boxes to numpy arrays
        l_iris_box = np.fromstring(boxes.iloc[0]['L_iris_box'].strip('[]'), sep=' ').astype(int)
        r_iris_box = np.fromstring(boxes.iloc[0]['R_iris_box'].strip('[]'), sep=' ').astype(int)
        return l_iris_box, r_iris_box
    else:
        # Handle case where no match is found
        print(f"No bounding box found for {image_name}")
        return None, None


class SAM:
    def __init__(self):
        self.predictor = self._initialize_predictor()

    def _initialize_predictor(self):
        SAM_CHECKPOINT_PATH = os.path.join('..','..', 'SAM_WEIGHTS', 'sam_vit_h_4b8939.pth')
        SAM_ENCODER_VERSION = "vit_h"

        DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
        return SamPredictor(sam)

    def set_image(self, image: np.ndarray):
        self.predictor.set_image(image)

    def segment_no_jitter(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        result_array = np.array(result_masks) 
        return {'right_iris': result_array[0],
                'left_iris':  result_array[1]}
        
    def get_embeddings(self, image):
        self.set_image(image)
        image_features = self.predictor.get_image_embedding()
        return image_features
        
    # helper functions to show mask and box for SAM display 
    @staticmethod
    def show_mask(mask, plt, color):
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.imshow(mask_image)
        
    @staticmethod
    #display the bounding box on the plot
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

 
