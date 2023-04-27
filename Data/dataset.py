import os
import csv
import torch
import numpy as np
from torch import nn
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split

device = 'cuda'
torch.manual_seed(3407)

class MRI_dataset(Dataset):
    def __init__(self, subj, data_type, brain_type, vis_transform, txt_transform, data_dir, csv_file_path):
        self.subj = format(subj, '02')
        self.data_dir = os.path.join(data_dir, 'subj'+self.subj)
        self.brain_type = brain_type
        self.vis_transform = vis_transform
        self.txt_transform = txt_transform

        if data_type == 'train':
            self.img_dir = os.path.join(self.data_dir, 'training_split', 'training_images')
            self.fmri_dir = os.path.join(self.data_dir, 'training_split', 'training_fmri')
            self.csv_file_path = os.path.join(csv_file_path,'subj' + self.subj, 'subj' +self.subj + '_train.csv')

        if data_type == 'test':
            self.img_dir = os.path.join(self.data_dir, 'test_split', 'test_images')
            self.csv_file_path = os.path.join(csv_file_path,'subj' + self.subj, 'subj' +self.subj + '_test.csv')
        
        self.imgs_paths = sorted(list(Path(self.img_dir).iterdir()))
        self.mri_array = self.read_fMRI_data()
        self.image_list = self.read_image_list()
        self.text_dict = self.read_text_csv_file()

        self.mri_dim = self.mri_array.shape[-1]

    def read_image_list(self):
        img_list = os.listdir(self.img_dir)
        img_list.sort()
        return img_list
    
    def read_fMRI_data(self):
        if self.brain_type == 'left':
            lh_fmri = np.load(os.path.join(self.fmri_dir, 'lh_training_fmri.npy'))
            return lh_fmri
        if self.brain_type == 'right':
            rh_fmri = np.load(os.path.join(self.fmri_dir, 'rh_training_fmri.npy'))
            return rh_fmri
    
    def read_text_csv_file(self):
        text_dict = {}
        with open(self.csv_file_path,'r') as data:
            for line in csv.reader(data):
                text_dict[line[0]] = line[1]
        return text_dict

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        img_path = self.imgs_paths[idx]
        img_name = str(img_path).split("/")[-1].replace('.png', '')

        img = Image.open(img_path).convert('RGB')

        img = self.vis_transform["eval"](img).to(device)

        mri = self.mri_array[idx]
        
        caption = self.text_dict[img_name]
        sen = self.txt_transform["eval"](caption)

        return img, sen, mri

def train_test_split(train_dataset, batch_size, shuffle=False):
    train_size = int(0.8 * len(train_dataset))
    eval_size = int(0.1 * len(train_dataset))
    test_size = len(train_dataset) - train_size - eval_size

    train_data, eval_data, test_data = random_split(train_dataset, [train_size, eval_size,test_size])
    train_data = torch.utils.data.ConcatDataset([train_data, eval_data])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

    return train_loader, eval_loader, test_loader