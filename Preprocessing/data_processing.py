from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch
import cv2

class MyDataset(Dataset):
    def __init__(self,image_folder,mask_folder,transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transforms.ToTensor()

        self.image_files = os.listdir(image_folder)
        self.mask_files = os.listdir(mask_folder)

        # self.image_files.sort()
        # self.mask_files.sort()

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        self.image_path = os.path.join(self.image_folder,self.image_files[index])
        self.mask_path = self.image_path.replace('images','masks')
        name = self.mask_files[index].split('.')[0]

        img = Image.open(self.image_path)
        mask = Image.open(self.mask_path)
        mask = mask.convert('L')

        img = img.crop((0,img.size[1]/10,img.size[0]-img.size[0]/15,img.size[1]-img.size[1]/10))
        mask = mask.crop((0,mask.size[1]/10,mask.size[0]-mask.size[0]/15,mask.size[1]-mask.size[1]/10))
        img.save(r'VIS/external/img/'+name+'.jpg')
        # mask.save(r'/data2/nqy/Uterus/Uterus_crop/validation/mask/'+name+'.jpg')

        img = np.array(img)
        (h,w) = img.shape[:2]
        img = cv2.resize(img,(320,256))
        img = Image.fromarray(img)
        # img.save(r'/data2/nqy/Uterus/Code/Segmentation/Train/Uterus/VIS/mask/'+name+'.jpg')

        mask = np.array(mask)
        
        mask1 = np.array(mask!=0,dtype=np.uint8)

        mask0 = np.ones_like(mask) - mask1
        final_mask = np.stack([mask0,mask1],2)
        final_mask = cv2.resize(final_mask,(320,256))

        if self.transform:
            img = self.transform(img)
            mask = torch.tensor(final_mask,dtype=torch.float32)
    
        mask = mask.permute(2,0,1)
        return name,(h,w),img,mask