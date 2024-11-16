import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import h5py

class OvarianDataset(Dataset):
    def __init__(self, args, root_dir, transform=None, train='train', saved_features='False'):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.saved_features = saved_features
        
        # Create a mapping from class to domains and images
        self.data = {}
     
        self.class_to_idx = {'CC':0, 'EC':1, 'HGSC':2, 'LGSC':3, 'MC':4}

        data_dir = os.path.join(self.root_dir, 'AMC/patches/1024/Mix')
        evel_csv = os.path.join(self.root_dir, 'AMC', 'AMC'.lower(),'s1024_p200/20x_val_split.1_2_train_3_eval.csv')
        eval_df = pd.read_csv(evel_csv, dtype={'name': str})['name'].tolist()

        manifest = os.path.join(self.root_dir, 'AMC', 'AMC'.lower(),'manifest.csv')
        manifest_df = pd.read_csv(manifest, dtype={'slide_id': str})

        self.data = {}
        # class_dirs = [os.path.join(domain_dir, d) for d in os.listdir(domain_dir) if os.path.isdir(os.path.join(domain_dir, d))]
        class_names = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']
        i = -1
        if self.saved_features == 'True':
            self.data = {0:[], 1:[], 2:[], 3:[], 4:[]}
            feature_dir = os.path.join(self.root_dir,'AMC', 'features/embeddings', 'AMC', args.model)
            slides = os.listdir(feature_dir)
            for slide in slides:
                class_name = manifest_df[manifest_df['slide_id'] == slide[:-3]]['subtype'].values[0]
                if self.train == 'train' and (slide[:-3] not in eval_df):
                    ppath = feature_dir + '/' +slide
                    f = h5py.File(ppath, 'r')
                    vectors = torch.tensor(f['features']['20x']).to(torch.float32)
                    for ind, pvector in enumerate(vectors):
                        self.data[self.class_to_idx[class_name]].append(pvector)
                elif self.train == 'eval' and (slide[:-3] in eval_df):
                    for pvector in torch.load(slide):
                        self.data[self.class_to_idx[class_name]].append(pvector)
                elif self.train == 'test':
                    for pvector in torch.load(slide):
                        self.data[self.class_to_idx[class_name]].append(pvector)
        else: 
            for class_name in class_names:
                class_dirs = os.path.join(data_dir, class_name) 
                i += 1
                if self.class_to_idx[class_name] not in self.data:
                    self.data[self.class_to_idx[class_name]] = []
                for fnames in os.listdir(class_dirs):
                    if self.train == 'train' and (fnames not in eval_df):  
                        for pname in os.listdir(os.path.join(class_dirs, fnames, '1024/20')):
                            path = os.path.join(class_dirs, fnames, '1024/20', pname)
                            # item = (path, class_index, fnames)
                            self.data[self.class_to_idx[class_name]].append(path)
                    elif self.train == 'eval' and (fnames in eval_df):
                        for pname in os.listdir(os.path.join(class_dirs, fnames, '1024/20')):
                            path = os.path.join(class_dirs, fnames, '1024/20', pname)
                            # item = (path, class_index, fnames)
                            self.data[self.class_to_idx[class_name]].append(path)
                    elif self.train == 'test':
                        for pname in os.listdir(os.path.join(class_dirs, fnames, '1024/20')):
                            path = os.path.join(class_dirs, fnames, '1024/20', pname)
                            # item = (path, class_index, fnames)
                            self.data[domain_name][self.class_to_idx[class_name]].append(path)
    
        self.flat_data = [(img, cls_i) for cls_i in self.data for img in self.data[cls_i]]


    def __len__(self):
        # Return the total number of images across all domains and classes
        return sum(len(self.flat_data))

    def __getitem__(self, idx):
        img_path, class_idx = self.flat_data[idx]
        with Image.open(img_path) as image:
            image_tensor = self.transform(image)
        return image_tensor, class_idx

