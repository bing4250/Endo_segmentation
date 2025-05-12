import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from lib.data_processing import MyDataset
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
# from UnetPlusPlus import UnetPlusPlus
# from DSC import DSC
from lib.Network_Res2Net_GRA_NCD import Network
# import wandb

if __name__ == '__main__':
    weight_path = r'./Net_dice_best.pth'

    transform = transforms.ToTensor()

    test_image_folder = '/data/rs/jq/data/endo_seg/test/crop_images'
    test_mask_folder = '/data/rs/jq/data/endo_seg/test/crop_masks'
    save_folder = '/data/rs/jq/code/SINet-V2-main/outsave/' + os.path.split(weight_path)[-1].replace('.pth','')
    threshold_folder = '/data/rs/jq/code/DSC-PyTorch-master/Endometrium/VIS/out_threshold/'+ os.path.split(weight_path)[-1].replace('.pth','')
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(threshold_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # net = DSC()
    net = Network()
    net.to(device)

    net.load_state_dict(torch.load(weight_path,map_location='cpu'))

    test_dataset = MyDataset(test_image_folder,test_mask_folder,transform=transform)
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
            # predict_4, predict_3, predict_2, predict_1, predict_0, predict_g, predict_f = test_output
            S_g_pred, S_5_pred, S_4_pred, S_3_pred = test_output
            # print(test_output.shape)
            # test_predicted = (test_output>0.5).float()
            test_output_probs = nn.functional.sigmoid(S_3_pred)
            # test_predicted = torch.argmax(test_output_probs, dim=1)
            test_output_probs[test_output_probs>=0.5] = 1
            test_output_probs[test_output_probs<0.5] = 0
            # 插值
            output = F.interpolate(test_output_probs,size=(h,w),mode='nearest')
            
            # endo = output[0,1:2,:,:]
            endo = output

            save_image(endo,os.path.join(save_folder,picname))

            # # 子宫
            # endo_cpu = endo.cpu().numpy()
            # endo_image = endo_cpu[0]*255
            
            # kernel = np.ones((21, 21), np.uint8)

            # # 应用开运算
            # opened_endo = cv2.morphologyEx(endo_image, cv2.MORPH_OPEN, kernel)

            # # 应用闭运算
            # closed_endo = cv2.morphologyEx(opened_endo, cv2.MORPH_CLOSE, kernel)

            # blurred_endo = cv2.GaussianBlur(closed_endo, (31, 31), 0)
            
            # # 应用阈值操作
            # _, thresholded_endo = cv2.threshold(blurred_endo, 127, 255, cv2.THRESH_BINARY)

            # # thresholded_ut[thresholded_ut==255] = 5
            # cv2.imwrite(threshold_folder+'/'+picname,thresholded_endo)
