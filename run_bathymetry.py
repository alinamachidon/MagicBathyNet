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
img_type = "s2"

DATA_FOLDER =  location+"/img/"+img_type+"/img_{}.tif"
LABEL_FOLDER =  location+"/depth/"+img_type+"/depth_{}.tif"
ERODED_FOLDER = location+"depth/"+img_type+"/depth_{}.tif"

# Parameters

#set the dataset modality here
dataset = "S2"
#dataset = "SPOT6"
#dataset = "UAV"

if dataset == "UAV":
    norm_param = np.load('agia_napa/norm_param_aerial.npy')
    norm_param_depth = -30.443   #-30.443 FOR AGIA NAPA, -11 FOR PUCK LAGOON
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

CACHE = True # Store the dataset in-memory

# We load one tile from the dataset and we display it
img = io.imread('agia_napa/img/s2/img_409.tif')
fig = plt.figure()
fig.add_subplot(121)
norm_img = (img - norm_param[0]) / (norm_param[1] - norm_param[0]) 
plt.imshow(norm_img)

# We load the ground truth
gt = io.imread('agia_napa/depth/aerial/depth_409.tif')
fig.add_subplot(122)
plt.imshow(gt/norm_param_depth)
plt.savefig("409_aerial_bathymetry.png")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.init()
net = net.to(device)

train_ids = train_images
test_ids = test_images

print("Tiles for training : ", train_ids)
print("Tiles for testing : ", test_ids)

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
epoch_folder = '/.../'


def train(net, optimizer, epochs, scheduler=None, save_epoch = 15):
    global epoch_folder
    global data_folder
    losses = np.zeros(10000000)
    mean_losses = np.zeros(100000000)
    mean_rmse_plot = np.zeros(1000000)
    mean_mse_plot = np.zeros(1000000)
    rmse_plot = np.zeros(1000000)
    mse_plot = np.zeros(1000000)
    epoch_folder = epoch_folder
    criterion = CustomLoss()
    iter_ = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.to(device)), Variable(target.to(device))
            optimizer.zero_grad()

            size=(256, 256)
            
            # Resizing data_p and label_p
            data = F.interpolate(data, size=size, mode='nearest')
            target = F.interpolate(target.unsqueeze(0), size=size, mode='nearest')
            
            #target = target.unsqueeze(0) #needed for aerial
            
            data_size = data.size()[2:]  # Get the original data size

            if data_size[0] > crop_size and data_size[1] > crop_size:
                    # Use RandomCrop transformation for data and target
                data_transform = RandomCrop(size=crop_size)
                target_transform = RandomCrop(size=crop_size)
    
                    # Apply RandomCrop transformation to data and target
                data = data_transform(data)
                target = target_transform(target)
                
            # Generate mask for non-annotated pixels in depth data
            target_mask = (target.cpu().numpy() != 0).astype(np.float32)  
            target_mask = torch.from_numpy(target_mask)  
            target_mask = target_mask.reshape(crop_size, crop_size)
            target_mask = target_mask.to(device)  
            
            data_mask = (data.cpu().numpy() != 0).astype(np.float32)  
            data_mask = np.mean(data_mask, axis=1)
            data_mask = torch.from_numpy(data_mask) 
            #data_mask = data_mask.reshape(crop_size, crop_size)
            data_mask = data_mask.to(device) 
            
            # Combine the masks
            combined_mask = target_mask * data_mask
            combined_mask = (combined_mask >= 0.5).float()
            # Check if combined_mask is 0
            if torch.sum(combined_mask) == 0:
            # Use another pair of data and target
                continue

            data = torch.clamp(data, min=0, max=1)
            output = net(data.float())

            loss = criterion(output, target, combined_mask)
            loss.backward()
            optimizer.step()
            losses[iter_] = loss.item() ##loss.data[0]
            mean_losses[iter_] = np.mean(losses[max(0,iter_-100):iter_])
            
            pred = output.data.cpu().numpy()[0]
            gt = target.data.cpu().numpy()[0]
            
            # Apply the mask to the predictions and ground truth
            masked_pred = pred * combined_mask.cpu().numpy()
            masked_gt = gt * combined_mask.cpu().numpy()

            rmse_plot[iter_] = metrics(np.concatenate([p.ravel() for p in masked_pred]), np.concatenate([p.ravel() for p in masked_gt]).ravel())
            mean_rmse_plot[iter_] = np.mean(rmse_plot[max(0,iter_-100):iter_])
            
            if iter_ % 100 == 0:
                if iter_ % 1000 == 0 and iter_ != 0:
                    try:
                        os.mkdir(DATA_FOLDER)
                    except FileExistsError:
                        pass
                clear_output()
                rgb = np.asarray(np.transpose(data.data.cpu().numpy()[0],(1,2,0)), dtype='float32')
                pred = output.data.cpu().numpy()[0]
                gt = target.data.cpu().numpy()[0]
                masked_pred = pred * combined_mask.cpu().numpy()
                masked_gt = gt * combined_mask.cpu().numpy()
                
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\Mean RMSE in m: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(), -norm_param_depth * metrics(np.concatenate([p.ravel() for p in masked_pred]), np.concatenate([p.ravel() for p in masked_gt]).ravel()))) ##loss.data[0]
               
                # Plot loss
                fig1, ax1 = plt.subplots(figsize=(14.0, 8.0))
                ax1.plot(mean_losses[:iter_], 'blue')
                ax1.set_title('Training Loss')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Loss')
                ax1.grid(color='black', linestyle='-', linewidth=0.5)

                # Plot accuracy
                fig2, ax2 = plt.subplots(figsize=(14.0, 8.0))
                ax2.plot(-norm_param_depth * mean_rmse_plot[:iter_], 'red')
                ax2.set_title('Mean RMSE in m')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Accuracy')
                ax2.grid(color='black', linestyle='-', linewidth=0.5)
    
                plt.show()
    
                if iter_ % 1000 == 0 and iter_ != 0:
                    fig1.savefig(DATA_FOLDER + "/train_loss_{}_out_of_{}".format(e, epochs))
                    fig2.savefig(DATA_FOLDER + "/validation_accuracy_{}_out_of_{}".format(e, epochs))
            
        
                fig = plt.figure(figsize=(14.0, 8.0))
                fig.add_subplot(131)
                plt.imshow(rgb)
                plt.title('RGB')
                fig.add_subplot(132)
                plt.imshow(gt[0,:,:], cmap='viridis_r', vmin=0, vmax=1)
                plt.title('Ground truth')
                fig.add_subplot(133)
                plt.title('Prediction')
                plt.imshow(masked_pred[0,:,:],  cmap='viridis_r', vmin=0, vmax=1)
                plt.suptitle('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\nLoss: {:.6f}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item()))
                plt.show()

                if iter_ % 1000 == 0 and iter_ != 0:
                    # plt.savefig(MAIN_FOLDER + model_folder +"output_data_filled_irfanview_no_shades_10/diagram_{}_out_of_{}".format(e,epochs))
                    fig.savefig(DATA_FOLDER + "/train_images_{}_out_of_{}".format(e, epochs))
                    # plt.savefig("Train_epoch_{}/{}_{}/{}_({:.0f}%).png".format(e, epochs, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader)))
            iter_ += 1
            
            del(data, target, loss)      
            
        if e % save_epoch == 0:
            try:
                os.mkdir(epoch_folder)
            except FileExistsError:
                pass

            # We validate with the largest possible stride for faster computing
            acc = test(net, test_ids, all=False)
            torch.save(net.state_dict(),epoch_folder + 'model_epoch{}'.format(e))
    torch.save(net.state_dict(), epoch_folder + 'model_final')


def test(net, test_ids, all=True):
    # Use the network on the test set
    test_images = ((np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') - norm_param[0]) / (norm_param[1] - norm_param[0]) for id in test_ids)

    test_labels = [1 / norm_param_depth * np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='float32') for id in test_ids]
    eroded_labels = [1 / norm_param_depth * np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='float32') for id in test_ids]
    all_preds = []
    all_gts = []
    all_masked_preds = []
    all_masked_gts = []

    # Switch the network to inference mode
    net.eval()
    
    mse = None
    
    ratio = crop_size / WINDOW_SIZE[0]
    k = 0
    for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
        img = scipy.ndimage.zoom(img, (ratio, ratio, 1), order=1)
        gt = scipy.ndimage.zoom(gt, (ratio, ratio), order=1)
        gt_e = scipy.ndimage.zoom(gt_e, (ratio, ratio), order=1)
        
        # Pad the image, ground truth, and eroded ground truth with reflection
        img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
        gt = np.pad(gt, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
        gt_e = np.pad(gt_e, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')

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
        gt = gt[pad_size:-pad_size, pad_size:-pad_size]
        gt_e = gt_e[pad_size:-pad_size, pad_size:-pad_size]

        # Display the result
        clear_output()
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(np.asarray(255 * img, dtype='uint8'))
        fig.add_subplot(1, 3, 2)
        plt.imshow(pred)  
        fig.add_subplot(1, 3, 3)
        plt.imshow(gt)
        fig.savefig(f"agia_napa/img/s2/test_image_result_{int(test_ids[k])}.png")
        k = k+1    
        
        # Generate mask for non-annotated pixels in depth data 
        gt_mask = (gt_e != 0).astype(np.float32) 
        gt_mask = torch.from_numpy(gt_mask) 
        gt_mask = gt_mask.unsqueeze(0)
        gt_mask = gt_mask.reshape(crop_size, crop_size)
        gt_mask = gt_mask.to(device) 

        img_mask = (img != 0).astype(np.float32) 
        img_mask = np.mean(img_mask, axis=2)
        img_mask = torch.from_numpy(img_mask)  
        #img_mask = img_mask.reshape(crop_size, crop_size)
        img_mask = img_mask.to(device) 
        
        combined_mask = img_mask*gt_mask
      
        masked_pred = pred * combined_mask.cpu().numpy()
        masked_gt_e = gt_e * combined_mask.cpu().numpy()
        all_preds.append(pred)
        all_gts.append(gt_e)
        
        all_masked_preds.append(masked_pred)
        all_masked_gts.append(masked_gt_e)       

        clear_output()

        metrics(masked_pred.ravel(), masked_gt_e.ravel(), norm_param_depth)
        # print(f"RSME: {rmse*-norm_param_depth}")
        rmse = metrics(np.concatenate([p.ravel() for p in all_masked_preds]), np.concatenate([p.ravel() for p in all_masked_gts]).ravel(), norm_param_depth)

    # Returning all predictions and ground truths if 'all' is set to True
    if all:
        return rmse, all_preds, all_gts
    else:
        return rmse  # Returning the final MSE for the test set    


if dataset=="S2":
    net.load_state_dict(torch.load('./bathymetry_s2_an'))
elif dataset=="UAV":
    net.load_state_dict(torch.load('./bathymetry_aerial_an'))
else:
    net.load_state_dict(torch.load('./bathymetry_spot6_an'))

#train_set = dataset(train_ids, cache=CACHE)
#train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

#train(net, optimizer, 10, scheduler)


_, all_preds, all_gts = test(net, test_ids, all=True)
#print(all_preds)
#print(all_gts)

ratio = crop_size / WINDOW_SIZE[0]

for p, id_ in zip(all_preds, test_ids):
    img = p*norm_param_depth
    
    img = scipy.ndimage.zoom(img, (1/ratio, 1/ratio), order=1)
    io.imsave('inference_tile_{}.png'.format(id_), img)
    nlcd02_arr_1, nlcd02_ds_1 = read_geotiff('agia_napa/img/s2/img_410.tif', 3)
    write_geotiff('./inference_tile_{}.tif'.format(id_), img, nlcd02_ds_1)