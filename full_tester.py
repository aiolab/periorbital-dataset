
import os
import time
import torch
import datetime
import numpy as np
import pickle
from scipy.ndimage import label

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import PIL

from full_test_utils import *

import re
from full_unet import unet
from full_utils import *
from PIL import Image, ImageOps
# from plotting import *

import pandas as pd
import csv

# from maskExtraction import EyeFeatureExtractor
# from measureAnatomy import EyeMetrics
# from distance_plot import Plotter
# from sam_model import get_bounding_boxes, SAM
from torchvision.models.segmentation import deeplabv3_resnet101

# SAM_CHECKPOINT_PATH = os.path.join('..','..', 'SAM_WEIGHTS', 'sam_vit_h_4b8939.pth')
# SAM_ENCODER_VERSION = "vit_h"


def transformer(dynamic_resize_and_pad, totensor, normalize, centercrop, imsize, is_mask=False):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if dynamic_resize_and_pad:
        options.append(ResizeAndPad(output_size=(imsize, imsize)))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(options)


class ResizeAndPad:
    def __init__(self, output_size=(512, 512), fill=0, padding_mode='constant'):
        self.output_size = output_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # Calculate new height maintaining aspect ratio
        original_width, original_height = img.size
        new_height = int(original_height * (self.output_size[0] / original_width))
        img = img.resize((self.output_size[0], new_height), Image.NEAREST)

        # Calculate padding
        padding_top = (self.output_size[1] - new_height) // 2
        padding_bottom = self.output_size[1] - new_height - padding_top

        # Apply padding
        img = ImageOps.expand(img, (0, padding_top, 0, padding_bottom), fill=self.fill)
        
        return img
    

class Tester(object):
    def __init__(self, config, device):
        # exact model and loss
        # self.model = config.model

        # Model hyper-parameters
        self.imsize = config.imsize
        self.parallel = config.parallel

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.lr = config.lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.img_path = config.img_path
        
        self.label_path = config.label_path 
        
        # self.log_path = config.log_path
        
        self.model_save_path = config.model_save_path
        # self.sample_path = config.sample_path
        
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        # self.dataset = config.dataset
        self.device = device
        
        # Path
        # self.log_path = os.path.join(config.log_path, self.version)
        # self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        
        self.dlv3 = config.dlv3
        
        # if self.dataset != 'ted_long':
        #     self.test_label_path = config.test_label_path
        #     self.test_label_path_gt = config.test_label_path_gt
        #     self.test_color_label_path = config.test_color_label_path
            
        self.test_image_path = config.test_image_path
        self.test_label_path = config.test_label_path

        # self.get_sam_iris = config.get_sam_iris

        # Test size and model
        self.test_size = config.test_size
        self.model_name = config.model_name
        
        # self.train_limit = config.train_limit
        
        self.csv_path = config.csv_path
        
        # self.pickle_in = config.pickle_in
        # self.split_face = config.split_face

        self.build_model()

    # def test(self):
    #     # sam = SAM()
    #     # plotter = Plotter()

    #     # if not self.pickle_in:
    #     # if self.split_face:
    #     transform = transform_img_split(resize=True, totensor=True, normalize=True)
    #         # transform_plotting = transform_img_split(resize=True, totensor=False, normalize=False)
    #     transform_gt = transformer(dynamic_resize_and_pad=True, totensor=True, normalize=False,centercrop=False, imsize=512)
    #     transform_gt_plot = transformer(dynamic_resize_and_pad=True, totensor=False, normalize=False,centercrop=False, imsize=512)

    #     # transform_gt = transformer(dynamic_resize_and_pad=True, totensor=True, normalize=False,centercrop=False, imsize=256)
            
    #     # if self.dataset != 'celeb':
    #     #     custom = True
    #     # else:
    #     #     custom = False
            
    #     # if custom, return list of paths, otherwise return list of paths for celeb that are numbers.jpg/png
    #     test_paths_imgs = make_dataset(self.test_image_path)
        
    #     # if self.dataset != 'ted_long':
    #     test_paths_labels = make_dataset(self.test_label_path, gt=True)
    #     print(f'length of test path is {len(test_paths_imgs)} and gt is {len(test_paths_labels)}')
    
    #     #align the paths so the indices match if custom dataset
    #     # if custom and self.dataset != 'ted_long':
    #     test_paths_imgs_forward, test_paths_labels_forward = align_image_and_label_paths(test_paths_imgs, test_paths_labels)
            
    #     # make_folder(self.test_label_path, '')
        
    #     # make_folder(self.test_color_label_path, '') 
        
    #     # load model
    #     self.G.load_state_dict(torch.load(os.path.join('models', self.version, self.model_name)))
    #     self.G.eval() 
        
    #     batch_num = int(self.test_size / self.batch_size)
    #     storage = []
    #     corrected_masks = []
    #     # iris_masks = []
    #     # images_plotting = []
    #     names = []
    #     gt_for_storage = []

    #     # start prediction process
    #     for i in range(batch_num):
            
    #         imgs = []
    #         gt_labels = []
    #         l_imgs = []
    #         r_imgs = []
    #         original_sizes = []
            
    #         for j in range(self.batch_size):
    #             current_idx = i * self.batch_size + j
    #             if current_idx < len(test_paths_imgs_forward):
    #                 path = test_paths_imgs_forward[current_idx]
    #                 name = path.split('/')[-1][:-4]
    #                 names.append(name)
    #                 print(name)

    #                 original_sizes.append(Image.open(path).size)
    #                 l_img, r_img = transform(Image.open(path))
    #                 l_imgs.append(l_img)
    #                 r_imgs.append(r_img)
    #                 # images_plotting.append(transform_gt(Image.open(path)))
                    
    #                 gt_path = test_paths_labels_forward[current_idx]
    #                 gt_img = Image.open(gt_path)
    #                 gt_label = transform_gt(gt_img)
    #                 print(type(transform_gt(gt_img)))
    #                 gt_labels.append(transform_gt(gt_img).numpy())
    #                 gt_for_storage.append(np.array(gt_label))

    #             else:
    #                 break 
                
                
    #         if len(imgs) != 0 or len(l_imgs)!=0:
    #             print('PREDICTING IMAGES NOW ')
    #             labels_predict_plain = predict_split_face(l_imgs,r_imgs, self.imsize, transform_gt_plot, self.device, original_sizes, self.G, dlv3=self.dlv3)                        

    #             # visualize and get the dice scores for plotting later
    #             visualize_and_dice(labels_predict_plain, np.array(gt_labels), self.batch_size, storage, dataset = self.version)

    #     # print(storage)
    #     if self.dlv3:
    #         csv_file_path = self.csv_path
           
    #     write_dice_scores_to_csv(storage, csv_file_path)

    def test(self):
        transform = transform_img_split(resize=True, totensor=True, normalize=True)
        transform_gt = transformer(dynamic_resize_and_pad=True, totensor=True, normalize=False, centercrop=False, imsize=512)
        transform_gt_plot = transformer(dynamic_resize_and_pad=True, totensor=False, normalize=False, centercrop=False, imsize=512)

        test_paths_imgs = make_dataset(self.test_image_path)
        test_paths_labels = make_dataset(self.test_label_path, gt=True)
        print(f'length of test path is {len(test_paths_imgs)} and gt is {len(test_paths_labels)}')
        
        test_paths_imgs_forward, test_paths_labels_forward = align_image_and_label_paths(test_paths_imgs, test_paths_labels)

        self.G.load_state_dict(torch.load(os.path.join('models', self.version, self.model_name)))
        self.G.eval() 

        batch_num = int(self.test_size / self.batch_size)
        storage = []
        names = []

        for i in range(batch_num):
            imgs = []
            gt_labels = []
            l_imgs = []
            r_imgs = []
            original_sizes = []
            
            for j in range(self.batch_size):
                current_idx = i * self.batch_size + j
                if current_idx < len(test_paths_imgs_forward):
                    path = test_paths_imgs_forward[current_idx]
                    name = path.split('/')[-1][:-4]
                    names.append(name)
                    print(name)

                    original_sizes.append(Image.open(path).size)
                    l_img, r_img = transform(Image.open(path))
                    l_imgs.append(l_img)
                    r_imgs.append(r_img)
                    
                    gt_path = test_paths_labels_forward[current_idx]
                    gt_img = Image.open(gt_path)
                    gt_label = transform_gt(gt_img)
                    gt_labels.append(transform_gt(gt_img).numpy())

                else:
                    break 
                
            if len(l_imgs) != 0:
                print('PREDICTING IMAGES NOW ')
                labels_predict_plain = predict_split_face(l_imgs, r_imgs, self.imsize, transform_gt_plot, self.device, original_sizes, self.G, dlv3=self.dlv3)                        

                # Visualize and get the dice scores for plotting later
                visualize_and_dice(labels_predict_plain, np.array(gt_labels), names, storage, dataset=self.version)
                names = []  # Reset names for the next batch

        # Save the results to a CSV file
        if self.dlv3:
            csv_file_path = self.csv_path

        write_dice_scores_to_csv(storage, csv_file_path)



    def build_model(self):
        if not self.dlv3:
            self.G = unet().to(self.device)
        # elif self.dlv3:
        #     self.G = deeplabv3_resnet101(num_classes=3).to(self.device)
        elif self.dlv3:
            self.G = deeplabv3_resnet101(pretrained=True)
            self.G.classifier = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 6, kernel_size=(1, 1), stride=(1, 1))
            )
            self.G.to(self.device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)


        
 
def overlay_mask_on_image(image, mask, alpha=0.5):
    color_map = {
        0: (255, 0, 0),     # Class 0: Red
        1: (0, 255, 0),     # Class 1: Green
        2: (0, 0, 255),     # Class 2: Blue
        3: (255, 255, 0),   # Class 3: Yellow
        4: (255, 0, 255),   # Class 4: Magenta
        5: (0, 255, 255),   # Class 5: Cyan
    }
    overlay = Image.new("RGB", image.size)
    
    for class_index, color in color_map.items():
        class_mask = (mask == class_index).astype(np.uint8) * 255
        class_overlay = Image.new("RGB", image.size, color)
        overlay = Image.composite(class_overlay, overlay, Image.fromarray(class_mask))

    return Image.blend(image, overlay, alpha)



def predict_split_face(l_imgs,r_imgs, imsize, transform_plotting, device, original_sizes, G, dlv3=False):
    l_imgs = torch.stack(l_imgs) 
    r_imgs = torch.stack(r_imgs) 
    
    print(l_imgs.size())
    
    l_imgs = l_imgs.to(device)
    r_imgs = r_imgs.to(device)
    

    if dlv3:
        l_labels_predict = G(l_imgs)['out'] 
        r_labels_predict = G(r_imgs)['out']  
        
    # else:
    #     l_labels_predict = G(l_imgs)
    #     r_labels_predict = G(r_imgs)
    
    l_labels_predict_plain = generate_label_plain(l_labels_predict, imsize)
    r_labels_predict_plain = generate_label_plain(r_labels_predict, imsize)
    
    labels_predict_plain = []

    for idx, (left_pred, right_pred) in enumerate(zip(l_labels_predict_plain, r_labels_predict_plain)):
        original_width, original_height = original_sizes[idx]
        mid = original_width // 2
        
        # Calculate dimensions for left and right halves based on the original sizes
        left_width = mid  # Since mid is the midpoint
        right_width = original_width - mid  # Width from midpoint to right edge

        # Resize predictions to match these dimensions
        left_pred_resized = cv2.resize(left_pred, (left_width, original_height), interpolation=cv2.INTER_NEAREST)
        right_pred_resized = cv2.resize(right_pred, (right_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Create a new empty array for the stitched prediction
        stitched = np.zeros((original_height, original_width), dtype=np.uint8)

        # Place the resized predictions onto the stitched canvas
        stitched[:, :mid] = left_pred_resized
        stitched[:, mid:] = right_pred_resized
    
        
        # Resize stitched prediction to 512x512
        resized_stitched = transform_plotting(Image.fromarray(stitched))
        labels_predict_plain.append(np.array(resized_stitched))
        
        # plt.imshow(np.array(resized_stitched))
        # plt.savefig('test.jpg')
        
        
        # # Load the original image for overlay
        # original_image = Image.open(os.path.join(self.test_image_path, f'{names[idx]}.jpg'))

        # # Overlay the mask on the original image
        # overlaid_image = overlay_mask_on_image(original_image, np.array(resized_stitched), color_map)

        # # Save the overlaid image
        # save_path = os.path.join(save_dir, f'overlaid_{idx}.png')
        # overlaid_image.save(save_path)
        

        
        
    print('converting labels')
    labels_predict_plain = np.array(labels_predict_plain)
    print(len(labels_predict_plain))
    
    return labels_predict_plain








# def save_to_csv(dataset, names, measurements, landmarks, gt=False, dlv3=False):
#     # Save measurements to CSV
#     if gt:
#         title = f'{dataset}_dlv3_{dlv3}_measurements_GROUND_TRUTH.csv'
#         with open(title, 'w', newline='') as file:
#             writer = csv.writer(file)
            
#             # Write header row
#             header = ['Name'] + list(measurements[0].keys()) + ['right_iris_diameter', 'left_iris_diameter']
#             writer.writerow(header)
            
#             # Write data rows
#             for name, measurement, landmark in zip(names, measurements, landmarks):
#                 row = [name] + list(measurement.values()) + [landmark['right_iris_diameter'], landmark['left_iris_diameter']]
#                 writer.writerow(row)
#     else:
#         with open(f'{dataset}_dlv3_{dlv3}_measurements.csv', 'w', newline='') as file:
#             writer = csv.writer(file)
            
#             # Write header row
#             header = ['Name'] + list(measurements[0].keys()) + ['right_iris_diameter', 'left_iris_diameter']
#             writer.writerow(header)
            
#             # Write data rows
#             for name, measurement, landmark in zip(names, measurements, landmarks):
#                 row = [name] + list(measurement.values()) + [landmark['right_iris_diameter'], landmark['left_iris_diameter']]
#                 writer.writerow(row)




        # plot_dice_boxplots(storage, self.version)
        
        
        # plot_dice_histograms(storage, title)
        

            
        # pred_measurements = []
        # pred_landmarks = []
        # gt_measurements = []
        # gt_landmarks = []

        # bad_indices_pred = []
        # bad_indices_gt = []
        # print(len(features_list))
        # # measurments for predictions
        # print('ANALYZING AI PREDICTIONS NOW')

        # for idx, features in enumerate(features_list):
        #     try:
        #         _, features_array = features  
                
        #         extractor = EyeFeatureExtractor(features_array, images_plotting[idx],idx)
        #         landmarks = extractor.extract_features()
        #         pred_landmarks.append(landmarks)

        #         # Create an instance of EyeMetrics with the landmarks
        #         eye_metrics = EyeMetrics(landmarks, features_array) 
        #         measurements = eye_metrics.run()
        #         pred_measurements.append(measurements)
                
        #         # store marked up images to see if this is working
        #         plotter.create_plots(images_plotting[idx], features_array, landmarks, names[idx], measurements, self.dlv3)
        #     except (ValueError, KeyError):
        #         bad_indices_pred.append(idx)

        # if self.train_limit == None:
        #     save_to_csv(self.dataset, names, pred_measurements, pred_landmarks, dlv3=self.dlv3)
                

        # #measurements for gt 
        # if self.dataset != 'ted_long':
        #     print('ANALYZING GT NOW')
        #     for idx, features in enumerate(gt_features_list):
        #         try:
        #             _, features_array_gt = features           

        #             extractor_gt = EyeFeatureExtractor(features_array_gt, images_plotting[idx],idx, gt=True)
        #             landmarks_gt = extractor_gt.extract_features()
        #             gt_landmarks.append(landmarks_gt)

        #             # Create an instance of EyeMetrics with the landmarks
        #             eye_metrics_gt = EyeMetrics(landmarks_gt, features_array_gt) 
        #             measurements_gt = eye_metrics_gt.run()
        #             gt_measurements.append(measurements_gt)
        #             plotter.create_plots(images_plotting[idx], features_array_gt, landmarks_gt, names[idx], measurements_gt, gt=True)

        #         except (ValueError, KeyError):
        #             bad_indices_gt.append(idx)
                    
        #     if self.train_limit == None:
        #         save_to_csv(self.dataset, names, gt_measurements, gt_landmarks, gt=True)

                    
        #     print(f'PRINTING BAD INDICES PRED:{bad_indices_pred} and GT: {bad_indices_gt}. REMOVING PRED FROM GT ONLY')
        #     # all_bad_indices = set(bad_indices_pred) | set(bad_indices_gt)

        #     # # Remove bad indices from the names list
        #     # names = [name for i, name in enumerate(names) if i not in all_bad_indices]

        #     # # Remove bad indices from gt_measurements and gt_landmarks if they are in bad_indices_pred
        #     # gt_measurements = [m for i, m in enumerate(gt_measurements) if i not in bad_indices_pred]
        #     # gt_landmarks = [l for i, l in enumerate(gt_landmarks) if i not in bad_indices_pred]

        #     # # Remove bad indices from pred_measurements and pred_landmarks if they are in bad_indices_gt
        #     # pred_measurements = [m for i, m in enumerate(pred_measurements) if i not in bad_indices_gt]
        #     # pred_landmarks = [l for i, l in enumerate(pred_landmarks) if i not in bad_indices_gt]
            
        #     title = f'{self.dataset}_{self.train_limit}_dlv3_{self.dlv3}test'
        #     mae_df = calculate_mae_for_all_images(names, gt_measurements, gt_landmarks, pred_measurements, pred_landmarks)        
        #     mae_df.to_csv(f'{title}_mae.csv')



    # def build_model(self):
    #     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     print(f"Using device: {self.device}")
    #     self.G = unet().to(self.device)
    #     if self.parallel:
    #         self.G = nn.DataParallel(self.G)
