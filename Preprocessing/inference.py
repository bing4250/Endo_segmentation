import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from data_processing import MyDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.transforms import functional as TF
from Model import Unet
# import wandb

if __name__ == '__main__':
    weight_path = r'checkpoint\Unet-0814-0.9366707801818848.pth'

    test_image_folder = r'external\junior\junior'
    test_mask_folder = r'external\junior\junior_mask'
    # save_folder = r'/VIS/out_save/' + os.path.split(weight_path)[-1].replace('.pth','')
    save_folder = r'VIS/external/mask' 
    threshold_folder = r'VIS/out_threshold/'+ os.path.split(weight_path)[-1].replace('.pth','')
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(threshold_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = Unet(3,2)
    net.to(device)

    net.load_state_dict(torch.load(weight_path,map_location='cpu'))

    test_dataset = MyDataset(test_image_folder,test_mask_folder,transform=transforms)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

    net.eval()

    with torch.no_grad():
        for index, (name,shape,test_image, test_mask) in enumerate(tqdm(test_loader)):
            name = name[0]
            picname = name + '.jpg'
            h = shape[0][0]
            w = shape[1][0]
            test_image, test_mask = test_image.to(device), test_mask.to(device)
            test_output = net(test_image)
            # print(test_output.shape)
            # test_predicted = (test_output>0.5).float()
            test_output_probs = nn.functional.sigmoid(test_output)
            # test_predicted = torch.argmax(test_output_probs, dim=1)
            test_output_probs[:,1][test_output_probs[:,1]>=0.5] = 1
            test_output_probs[:,1][test_output_probs[:,1]<0.5] = 0
            # 插值
            output = F.interpolate(test_output_probs,size=(h,w),mode='nearest')
            
            ut = output[0,1:2,:,:]

            save_image(ut,os.path.join(save_folder,picname))

            # 子宫
            ut_cpu = ut.cpu().numpy()
            ut_image = ut_cpu[0]*255
            
            kernel = np.ones((21, 21), np.uint8)

            # 应用开运算
            opened_ut = cv2.morphologyEx(ut_image, cv2.MORPH_OPEN, kernel)

            # 应用闭运算
            closed_ut = cv2.morphologyEx(opened_ut, cv2.MORPH_CLOSE, kernel)

            blurred_ut = cv2.GaussianBlur(closed_ut, (31, 31), 0)
            
            # 应用阈值操作
            _, thresholded_ut = cv2.threshold(blurred_ut, 127, 255, cv2.THRESH_BINARY)

            thresholded_ut[thresholded_ut==255] = 5
            cv2.imwrite(threshold_folder+'/'+picname,thresholded_ut)
