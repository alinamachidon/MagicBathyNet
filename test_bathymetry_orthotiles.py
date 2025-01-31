# Standard Library
import os
import sys
import time
import random
import itertools

# Numerical Libraries
import numpy as np
import numpy.ma as ma

import scipy
from scipy import ndimage

# Image Processing Libraries
from skimage import io
from skimage.transform import resize, rotate

# Matplotlib
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.metrics import confusion_matrix, precision_score, recall_score, mean_squared_error

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler
import torch.nn.init

# Raster and Remote Sensing Libraries
import rasterio
import gdal

# Deep Learning Model
from bathymetry.magicbathy_unet import *
# Dataset
from dataset_bathymetry import dataset
from custom_loss import CustomLoss
from utils import *

# Utility Libraries
from glob import glob
from IPython.display import clear_output
from torchvision.transforms import RandomCrop, Resize
from tqdm import tqdm_notebook as tqdm
from skimage import exposure


location = "agia_napa"
#location = "puck_lagoon"
img_type = "aerial"

DATA_FOLDER =  location+"/img/"+img_type+"/img_{}.tif"
LABEL_FOLDER =  location+"/depth/"+img_type+"/depth_{}.tif"
ERODED_FOLDER = location+"depth/"+img_type+"/depth_{}.tif"

# Parameters

#set the dataset modality here
#dataset = "S2"
#dataset = "SPOT6"
dataset = "UAV"

norm_param_depth = {'agia_napa':-30.443, 'puck_lagoon':-11}

if dataset == "UAV":
    norm_param = np.load(f'{location}/norm_param_aerial.npy')
    norm_param_depth = norm_param_depth[location]   #-30.443 FOR AGIA NAPA, -11 FOR PUCK LAGOON
    WINDOW_SIZE = (720, 720)
    STRIDE = 16
    BATCH_SIZE = 1
    train_images = ['409', '418', '350', '399', '361', '430', '380', '359', '371', '377', '379', '360', '368', '419', '389', '420', '401', '408', '352', '388', '362', '421', '412', '351', '349', '390', '400', '378']
    test_images = ['411', '387', '410', '398', '370', '369', '397']
    
elif dataset == "SPOT6":
    norm_param = np.load('agia_napa/norm_param_spot6_an.npy')
    norm_param_depth = -30.443   #-30.443 FOR AGIA NAPA, -11 FOR PUCK LAGOON
    WINDOW_SIZE = (30, 30)
    STRIDE = 2
    BATCH_SIZE = 1
    train_images = ['409', '418', '350', '399', '361', '430', '380', '359', '371', '377', '379', '360', '368', '419', '389', '420', '401', '408', '352', '388', '362', '421', '412', '351', '349', '390', '400', '378']
    test_images = ['411', '387', '410', '398', '370', '369', '397']
    
elif dataset == "S2":
    norm_param = np.load('agia_napa/norm_param_s2_an.npy')
    norm_param_depth = -30.443   #-30.443 FOR AGIA NAPA, -11 FOR PUCK LAGOON
    WINDOW_SIZE = (18, 18)
    STRIDE = 2
    BATCH_SIZE = 1
    train_images = ['409', '418', '350', '399', '361', '430', '380', '359', '371', '377', '379', '360', '368', '419', '389', '420', '401', '408', '352', '388', '362', '421', '412', '351', '349', '390', '400', '378']
    test_images = ['411', '387', '410', '398', '370', '369', '397']
    
net = UNet_bathy(3, 1)
base_lr = 0.0001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.init()
net = net.to(device)

params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        params += [{'params':[value],'lr': base_lr}]
    else:
        params += [{'params':[value],'lr': base_lr}] 
        
optimizer = optim.Adam(net.parameters(), lr=base_lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.1)

crop_size = 256
pad_size = 32  # Define pad size here


def read_water_tiles_only():
    water_tiles_file = ("water_tiles_new.txt")
    # Print the absolute path to check if it's correct
    print("File path:", os.path.abspath(water_tiles_file))

    #water_tiles_file = "./././drone_processing/water_tiles.txt"
    # Read file names from water_tiles.txt
    try:
        with open(water_tiles_file, "r") as f:
            water_tile_names = [line.strip() for line in f.readlines()]

        if not water_tile_names:
            raise ValueError("No water tiles found in the file.")
        # Load images using the filenames from water_tiles.txt

        base_path = os.path.expanduser("~/drone_processing/data/chips/")
        
        # for tile_name in water_tile_names:
        #     print(os.path.abspath(os.path.join("../../../../drone_processing/data/chips/",tile_name)))
        #     print(os.path.abspath(os.path.join(base_path, tile_name)))

        test_images = (
            (np.asarray(io.imread(os.path.abspath(os.path.join(base_path, tile_name))), dtype="float32") - norm_param[0]) /
            (norm_param[1] - norm_param[0])
            for tile_name in water_tile_names
        )
        print(f" Successfully loaded {len(water_tile_names)} water tile images.")
        return test_images, water_tile_names
    
    except FileNotFoundError:
        print(f" Error: File '{water_tiles_file}' not found!")
    except ValueError as ve:
        print(f" {ve}")
    except Exception as e:
        print(f" Unexpected error: {e}")


def test_on_orthophoto(net):
    # Use the network on the test set
    #test_images = ((np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') - norm_param[0]) / (norm_param[1] - norm_param[0]) for id in test_ids)
    test_images, water_tiles_names =  read_water_tiles_only()
   
    # Switch the network to inference mode
    net.eval()
        
    ratio = crop_size / WINDOW_SIZE[0]

    for filename, img in zip(water_tiles_names, test_images):
        img = scipy.ndimage.zoom(img, (ratio, ratio, 1), order=1)
        
        # Pad the image, ground truth, and eroded ground truth with reflection
        img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
       
        # Convert image to tensor
        img_tensor = np.copy(img).transpose((2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = torch.from_numpy(img_tensor).cuda()

        # Do the inference on the whole image
        with torch.no_grad():
            outs = net(img_tensor.float())
            pred = outs.data.cpu().numpy().squeeze()

         # Remove padding from prediction
        pred = pred[pad_size:-pad_size, pad_size:-pad_size]
        img = img[pad_size:-pad_size, pad_size:-pad_size]
        
        # Display the result
        clear_output()
        fig = plt.figure()
        # fig.add_subplot(1, 2, 1) 
        # plt.imshow(np.asarray(255 * img, dtype='uint8'))
        # fig.add_subplot(1, 2, 2) 
        # plt.imshow(pred)  
        # fig.savefig(f"soca/uav_{location}/test_image_result_{filename[:-4]}.png")
        # plt.close()

        # Plot the first image (side 1)
        ax1 = fig.add_subplot(1, 2, 1)  # Layout: 1 row, 2 columns, 1st subplot
        cax1 = ax1.imshow(np.asarray(255 * img, dtype='uint8'))  # Convert to uint8 if needed
        ax1.set_title("Original Image")
        ax2 = fig.add_subplot(1, 2, 2)  
        cax2 = ax2.imshow(pred)
        ax2.set_title("Predicted Depth")
        # Add a colorbar for the second subplot (depth prediction)
        #fig.colorbar(cax2, ax=ax2, orientation='vertical', label='Depth Values')
        # Add a single colorbar (linked to the first image's plot)
        fig.colorbar(cax2, ax=[ax1, ax2], orientation='vertical', label='Depth Values', fraction=0.02, pad=0.04)

        fig.savefig(f"soca/uav_{location}/test_image_result_{filename[:-4]}.png")
        plt.close()
       
        

if dataset=="S2":
    net.load_state_dict(torch.load('./bathymetry_s2_an'))
elif dataset=="UAV":
    #net.load_state_dict(torch.load('./bathymetry_aerial_pl'))
    net.load_state_dict(torch.load('./bathymetry_aerial_an'))
else:
    net.load_state_dict(torch.load('./bathymetry_spot6_an'))

test_on_orthophoto(net)

