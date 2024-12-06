import os
import random
from PIL import Image
from PIL import ImageFile
import torch
from torch.utils.data import Dataset
import pandas as pd
import h5py
from scipy import sparse
ImageFile.LOAD_TRUNCATED_IMAGES = True


class OvarianDataset(Dataset):
    """
    This class handles loading data from its directory and passing data to the model in batch_sizes
    """
    def __init__(self, args, root_dir, transform=None, train='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # Create a mapping from class to images
        self.data = {}
     
        self.class_to_idx = {'CC':0, 'EC':1, 'HGSC':2, 'LGSC':3, 'MC':4}

        data_dir = os.path.join(args.csv_path, 'MKobel/patches/1024/Mix')
        eval_path = os.path.join(args.csv_path, 'eval.csv')
        eval_df = pd.read_csv(eval_path, dtype={'name': str})['name'].tolist()

        manifest = os.path.join(args.csv_path, 'manifest.csv')
        manifest_df = pd.read_csv(manifest, dtype={'slide_id': str})

        self.data = {}
        class_names = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']
        i = -1
        
        for class_name in class_names:
            class_dirs = os.path.join(data_dir, class_name) 
            if os.path.exists(class_dirs):
                i += 1
                if self.class_to_idx[class_name] not in self.data:
                    self.data[self.class_to_idx[class_name]] = []
                for fnames in os.listdir(class_dirs):
                    if self.train == 'train' and (fnames not in eval_df):  
                        for pname in os.listdir(os.path.join(class_dirs, fnames, '1024/20')):
                            path = os.path.join(class_dirs, fnames, '1024/20', pname)
                            self.data[self.class_to_idx[class_name]].append(path)
                    elif self.train == 'test' and (fnames in eval_df):
                        for pname in os.listdir(os.path.join(class_dirs, fnames, '1024/20')):
                            path = os.path.join(class_dirs, fnames, '1024/20', pname)
                            self.data[self.class_to_idx[class_name]].append(path)
               
        self.flat_data = [(img, cls_i) for cls_i in self.data for img in self.data[cls_i]]


    def __len__(self):
        # Return the total number of images across all domains and classes
        return len(self.flat_data)

    def __getitem__(self, idx):
        img_path, class_idx = self.flat_data[idx]
        with Image.open(img_path) as image:
            image_tensor = self.transform(image)
        return image_tensor, class_idx
