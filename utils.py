import numpy as np
import torch 
import itertools
from sklearn.metrics import confusion_matrix
import rasterio
import gdal

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 0)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 0)
    y2 = y1 + h
    return x1, x2, y1, y2


def sliding_window(top, step=10, window_size=(20,20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
            
def count_sliding_window(top, step=10, window_size=(20,20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def metrics(predictions, gts, norm_param_depth):
    # Exclude 0 values from calculation
    non_zero_mask = predictions != 0
    
    # Calculate RMSE, MAE, and collect predictions and targets
    rmse = np.sqrt(np.mean(((predictions - gts) ** 2)[non_zero_mask]))
    mae = np.mean(np.abs((predictions - gts)[non_zero_mask]))
    std_dev = np.std((predictions - gts)[non_zero_mask])
    
    print("RMSE : {:.3f}m".format(rmse*-norm_param_depth))
    print("MAE : {:.3f}m".format(mae*-norm_param_depth))
    print("Std_Dev : {:.3f}m".format(std_dev*-norm_param_depth))
    print("---")
    
    return rmse

def read_geotiff(filename, b):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(b)
    arr = band.ReadAsArray()
    return arr, ds

def write_geotiff(filename, arr, in_ds):
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)