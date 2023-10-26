"""
Define a Dataset model that inherits from Pytorch's base class.
"""
import os 
import random
import numpy as np
import torch
#import cv2
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.io import loadmat



class Data(Dataset):
    def __init__(self, filename_x='data_25', filename_y='data_125',
                 directory="12jie/", transform=transforms.ToTensor(), flag1=0):
        b=list(range(0,298))
        random.shuffle(b)

        if flag1 == 0:
            x = loadmat(os.path.join(directory, filename_x))['array']
            y = loadmat(os.path.join(directory, filename_y))['labell']
           # x = loadmat(os.path.join(directory, filename_x))['record']
           # y = loadmat(os.path.join(directory, filename_y))['biaoqian']
            #x = loadmat(os.path.join(directory, filename_x))['input']
           #y = loadmat(os.path.join(directory, filename_y))['output']

            x = x.transpose(2, 0, 1)#维度变化一下
            y = y.transpose(2, 0, 1)

            self.transform = transform
            self.data = {
                'X': x,
                'Y': y
            }

            # Save data shapes for creating models.
            self.input_dim = x.shape[-2:]
            self.xx=x
            self.output_dim = y.shape[-2:]
            self.output_dim_fk = list(self.output_dim)
            self.output_dim_fk[-1] = self.output_dim_fk[-1] // 2 + 1


        elif flag1 == 1:

            zz = loadmat(os.path.join(directory, filename_x))['processed_blind_test']
            zz = zz.squeeze(-3)

            self.data = {
                'zz': zz
            }
            self.zz = zz

        elif flag1 == 2:
            ww = loadmat(os.path.join(directory, filename_x))['rray']
            ww = ww.transpose(2, 0, 1)


            #ww = ww.squeeze(-3)

            self.data = {
                'ww': ww
            }
            self.ww = ww
            self.maxxx = np.max(abs(ww))



        '''
        # Transform makes sure that type is torch and that the
        # dimensions are (NxHxW).
        x_transformed = transforms(x)
        y_transformed = transforms(y)
        '''





        
    def __len__(self):
        return self.data['X'].shape[0]
        
    def __getitem__(self, idx):
        sample = {
            'x': self.data['X'][idx],
            'y': self.data['Y'][idx],
            'max':np.max(self.data['Y'][idx])
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        x, y = sample['x'], sample['y']
        max=sample['max']

        return {
            'x': torch.from_numpy(x.copy()).unsqueeze(0),
            'y': torch.from_numpy(y.copy()).unsqueeze(0),
            'max':max
        }


class RandomHorizontalFlip(object):
    def __init__(self, flip_p=0.5):#依概率p垂直翻转
    #def __init__(self, flip_p=0.5):
        """
        Randomly flip a sample horizontally.
        """
        self.flip_p = flip_p

    def __call__(self, sample):
        x, y = sample['x'], sample['y']

        if random.random() < self.flip_p:
            x = np.fliplr(x)
            y = np.fliplr(y)

        return {
            'x': x,
            'y': y,


        }
