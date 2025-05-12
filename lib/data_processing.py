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
        mask_endo = Image.open(self.mask_path)
        mask_endo = mask_endo.convert('L')

        # mask = cv2.imread(self.mask_path)
        # mask[mask==127]=255
        # # 转换为灰度图
        # gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        # # 将图片二值化
        # _, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # # min_area_threshold = 100
        # if len(contours) != 0:
        #     max_ = max([len(contours[i]) for i in range(len(contours))])
        #     for j in range(len(contours)):
        #             if len(contours[j]) == max_:
        #                 # 外接矩形
        #                 x, y, w, h = cv2.boundingRect(contours[j])
        #             # 在原图上画出预测的矩形
        #             # cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), 10)
        #             # cv2.imwrite(r'/data2/nqy/Uterus/Code/Classification/data/0626/crop/mask.jpg',mask)
        #     crop_img = img.crop((x,y,x+w,y+h))
        #     crop_mask = mask_endo.crop((x,y,x+w,y+h))

        img = np.array(img)
        (h,w) = img.shape[:2]
        img = cv2.resize(img,(256,256))
        img = Image.fromarray(img)
        # img.save(r'/data2/nqy/Uterus/Code/Segmentation/Train/Endometrium/VIS/img/'+name+'.jpg')

        mask = np.array(mask_endo)
        
        mask1 = mask.copy()

        mask1[mask1 > 170] = 255
        mask1[mask1 < 36] = 0
        mask1[mask1 == 255] = 0
        mask1[mask1 != 0] = 1
        # mask1 = mask1 * 255
        # # 高斯模糊
        # mask1 = cv2.GaussianBlur(mask1, (31, 31), 0)
        # # 应用阈值操作
        # _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
        # mask1 = cv2.resize(mask1,(320,256))
        # cv2.imwrite(r'/data2/nqy/Uterus/Code/Segmentation/Train/Uterus/VIS/mask/'+name+'.jpg', mask1)

        mask0 = np.ones_like(mask) - mask1
        final_mask = np.stack([mask0,mask1],2)
        final_mask = cv2.resize(final_mask,(256,256))

        if self.transform:
            img = self.transform(img)
            mask = torch.tensor(final_mask,dtype=torch.float32)
    
        mask = mask.permute(2,0,1)
        # Convert tensor to NumPy array for saving
        # final_mask_np = mask.numpy()

        # # Save each channel of the mask as separate images if needed
        # cv2.imwrite(r'/data2/nqy/Uterus/Code/Segmentation/Train/Uterus/VIS/mask/'+name+'_0.jpg', final_mask_np[0] * 255)
        # cv2.imwrite(r'/data2/nqy/Uterus/Code/Segmentation/Train/Uterus/VIS/mask/'+name+'_1.jpg', final_mask_np[1] * 255)

        # # Optionally, you can save the entire mask as a single image
        # # Combine the channels back if needed
        # combined_mask = np.stack([final_mask_np[0], final_mask_np[1], final_mask_np[1]], axis=-1)
        # cv2.imwrite(r'/data2/nqy/Uterus/Code/Segmentation/Train/Uterus/VIS/mask/'+name+'.jpg', combined_mask * 255)
        return name,(h,w),img,mask