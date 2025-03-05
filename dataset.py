import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torch
import cv2

from util import make_hm_regr

image_size = 512
scale = 4

#image = image.astype(np.float32)/255
#image = image.transpose([2,0,1])
# Transforms totensor already does normalization and hwc to chw
image_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class CenternetDataset(Dataset):

    def __init__(self, image_dir,label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir,image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image,(image_size,image_size))
        image = image_transforms(image) 
        #
        label_path = os.path.join(self.label_dir,os.path.splitext(image_name)[0] + '.txt')
        if not os.path.exists(label_path):
            box_data = []
        else:
            with open(label_path,'r') as f:
                box_data = [[float(ch) for ch in line.strip().split(' ')[1:]] for line in f.readlines()]
        #
        center_hm,regression_map = make_hm_regr(box_data) 
        return image,torch.from_numpy(center_hm),torch.from_numpy(regression_map)
 