
import numpy as np
import random
from utils import *
from torchvision.transforms import Resize

random.seed(1)

DATA_FOLDER = 'agia_napa/img/aerial/img_{}.tif'
LABEL_FOLDER = 'agia_napa/gts/aerial/gts_{}.tif'
ERODED_FOLDER = 'agia_napa/gts/aerial/gts_{}.tif'

class dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                            cache=False, augmentation=True):
        super(dataset, self).__init__()
        
        self.augmentation = augmentation
        self.cache = cache
        
        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]
        
        

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        
        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}
            
    
    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)
    
    def __getitem__(self, i):
        
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            #if dataset == "S2":
            data = np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            data = (data - norm_param[0][:, np.newaxis, np.newaxis]) / (norm_param[1][:, np.newaxis, np.newaxis] - norm_param[0][:, np.newaxis, np.newaxis]) 
            #else:
             #   data = np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32') / norm_param          
                       
  
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label
        
        if dataset == "UAV":
            data[data == 0] = 0.5   
        if dataset == "SPOT6":
            data[data == 0] = 0.5
        if dataset == "S2":
            data[data == 0] = 0.5


        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]

        data_p, label_p = self.data_augmentation(data_p, label_p)
   
        data_p = scipy.ndimage.zoom(data_p, (1,scale,scale), order=0)
        label_p = scipy.ndimage.zoom(label_p, scale, order=0)
   
        
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))

