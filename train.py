import os
from tqdm import tqdm
import torch
import torchio as tio
from sub_dataset import train_ds,valid_ds


import gc
import monai
from train_valid_functions import train



device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = monai.networks.nets.EfficientNetBN("efficientnet-b0",spatial_dims = 3,in_channels = 1,num_classes = 11,pretrained = True)

if __name__ == '__main__':
    train(model,train_ds,valid_ds,resume=True)




