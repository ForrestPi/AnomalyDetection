# data loader 
import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F



class MVTecAD(data.Dataset):
    """Dataset class for the MVTecAD dataset."""

    def __init__(self, image_dir, transform):
        """Initialize and preprocess the MVTecAD dataset."""
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, index):
        """Return one image"""
        filename = "{:03}.png".format(index)
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image)

    def __len__(self):
        """Return the number of images."""
        return len(os.listdir(self.image_dir))


def return_MVTecAD_loader(image_dir, batch_size=256, train=True):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize((512, 512)))
    transform.append(T.RandomCrop((128,128)))
    transform.append(T.RandomHorizontalFlip(p=0.5))
    transform.append(T.RandomVerticalFlip(p=0.5))    
    transform.append(T.ToTensor())
    transform = T.Compose(transform)

    dataset = MVTecAD(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=train)
    return data_loader