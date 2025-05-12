import torch
import torch.nn as nn
from tqdm import tqdm
# from torch.cuda.amp import autocast as autocast
from torch import optim
# import wandb
import cv2
import matplotlib.pyplot as plt
import datetime
from data_processing import MyDataset
from torch.utils.data import DataLoader

from torchvision import transforms
from Model import Unet
import os

def dice_iou(predicted,mask):
    intersection = torch.sum(predicted*mask)
    union = torch.sum(predicted) + torch.sum(mask)
    dice = (2*intersection)/(union+1e-8)
    iou = dice/(2-dice)
    return dice.item(),iou.item()

def dice_coefficient(y_true, y_pred, smooth=0.00001):
    intersection = torch.sum(y_true[0,1] * y_pred[0,1])
    union = torch.sum(y_true[0,1]) + torch.sum(y_pred[0,1])

    # 子宫
    dice = (2 * intersection) / (union + smooth)

    return dice

def calculate_ls(output,target,fn):
    loss = 0
    for i in range(len(output)):
        loss += fn(output[i],target)
    loss = loss/len(output)
    return loss


def train(net,device,epochs,train_loader,val_loader,criterion,optimizer,model_name,deep_supervision = False):
    max_val_dice = 0.93
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[88, 168, 388], gamma=0.1)

    train_loss_arr = []
    val_loss_arr = []

    for epoch in tqdm(range(epochs)):
        loss_total = 0
        val_loss_total = 0

        net.train()
        # print('\ntrain! ')
        for name,(h,w),image,mask in tqdm(train_loader):
            optimizer.zero_grad()
            
            image,mask = image.to(device),mask.to(device)
            mask = torch.squeeze(mask)
            output = net(image)
            if deep_supervision: 
                loss = calculate_ls(output,mask,criterion)
            else:
                if mask.shape[0] != output.shape[0]:
                    mask = torch.unsqueeze(mask,dim=0)
                loss = criterion(output,mask)

            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_total += loss.item()
        
        loss_append = loss_total/len(train_loader)
        train_loss_arr.append(loss_append)

        net.eval()

        # print('\nval!')
        with torch.no_grad():
            dice_scores = []
            iou_scores = []

            for name,(h,w),val_image, val_mask in tqdm(val_loader):
                val_image, val_mask = val_image.to(device), val_mask.to(device)
                # val_mask = torch.squeeze(val_mask)
                
                # 获取模型输出并应用 softmax
                val_output = net(val_image)
                # val_loss = criterion(val_output,val_mask)
                # val_output_probs = nn.functional.sigmoid(val_output)
                # val_output_probs[:,1][val_output_probs[:,1]>=0.5] = 1
                # val_output_probs[:,1][val_output_probs[:,1]<0.5] = 0
                # val_output_probs[:,2][val_output_probs[:,2]>=0.5] = 1
                # val_output_probs[:,2][val_output_probs[:,2]<0.5] = 0
                if deep_supervision:
                    val_loss = calculate_ls(val_output,val_mask,criterion)
                    val_output_probs = nn.functional.sigmoid(val_output[3])
                    val_output_probs[:,1][val_output_probs[:,1]>=0.5] = 1
                    val_output_probs[:,1][val_output_probs[:,1]<0.5] = 0

                else:
                    val_loss = criterion(val_output,val_mask)
                    val_output_probs = nn.functional.sigmoid(val_output)
                    val_output_probs[:,1][val_output_probs[:,1]>=0.5] = 1
                    val_output_probs[:,1][val_output_probs[:,1]<0.5] = 0

                # val_predicted = torch.argmax(val_output_probs, dim=1)
                # val_predicted = (val_output>0.5).float()

                # 计算 Dice, IoU, mIoU
                # dice, iou, miou = calculate_dice_iou_miou(val_predicted, val_mask, num_classes=2)
                dice = dice_coefficient(val_mask,val_output_probs)
                iou = dice/(2-dice)

                dice_scores.append(dice)
                iou_scores.append(iou)
                val_loss_total += val_loss.item()

        # 计算平均值
        average_dice = sum(dice_scores) / len(dice_scores)
        average_iou = sum(iou_scores) / len(iou_scores)
        val_loss_append = val_loss_total/len(val_loader)
        val_loss_arr.append(val_loss_append)


        checkpint_path = f'checkpoint/{model_name}'
        os.makedirs(checkpint_path,exist_ok=True)

        if (average_dice >= max_val_dice):
            torch.save(net.state_dict(), os.path.join(checkpint_path, f'{model_name}-0814-{average_dice}.pth'))
            print('保存权重')
            max_val_dice = average_dice

        print(f'Epoch:{epoch+1}/{epochs}:loss:{loss}, val loss:{val_loss}, dice:{average_dice}, iou:{average_iou}')
        # wandb.log({'loss': loss, 'val_loss': val_loss,'dice1':average_dice1, 'iou':average_iou1,'dice2':average_dice2,'iou2':average_iou2})

        epochs_l = range(1,len(train_loss_arr)+1)
        train_loss_cpu = [loss for loss in train_loss_arr]
        # train_loss_cpu = [loss.detach().numpy() for loss in train_loss_cpu]
        val_loss_cpu = [loss for loss in val_loss_arr]
        # val_loss_cpu = [loss.detach().numpy() for loss in val_loss_cpu]

        # Plotting loss
        plt.cla()
        plt.figure()
        plt.plot(epochs_l, train_loss_cpu, 'b', label='Training loss')
        plt.plot(epochs_l, val_loss_cpu, 'r', label='Validation loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'VIS/loss/{model_name}_Loss.png')
        plt.close()

        print(f'max dice:{max_val_dice}')


def test(net,test_loader,criterion,device,model_name):
    net.eval()

    # print('\ntest!')
    with torch.no_grad():
        dice_scores = []
        iou_scores = []
        for name, (h,w), test_image, test_mask in tqdm(test_loader):
            test_image, test_mask = test_image.to(device), test_mask.to(device)
            # test_mask = torch.squeeze(test_mask)
            test_output = net(test_image)
            test_loss = criterion(test_output,test_mask)
            # test_predicted = (test_output>0.5).float()
            test_output_probs = nn.functional.sigmoid(test_output)
            # test_predicted = torch.argmax(test_output_probs, dim=1)
            test_output_probs[:,1][test_output_probs[:,1]>=0.5] = 1
            test_output_probs[:,1][test_output_probs[:,1]<0.5] = 0
            
            # 计算 Dice, IoU, mIoU
            # dice, iou, miou = calculate_dice_iou_miou(test_predicted, test_mask, num_classes=2)
            dice = dice_coefficient(test_mask,test_output_probs)
            iou = dice/(2-dice)
            iou = dice/(2-dice)

            dice_scores.append(dice)
            iou_scores.append(iou)

            # visualize_segmentation(test_image, test_mask, test_output_probs,index,model_name)



    # 计算平均值
    average_dice = sum(dice_scores) / len(dice_scores)
    average_iou = sum(iou_scores) / len(iou_scores)
    print(f'Test loss: {test_loss}, dice: {average_dice},iou: {average_iou}')


transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(degrees=15),  # 随机旋转
    transforms.RandomResizedCrop(size=(256, 320), scale=(0.8, 1.0)),  # 随机裁剪和缩放
    transforms.ToTensor()
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False)  # 随机擦除,
])


def DataProcess(train_image_folder,train_mask_folder,val_image_folder,val_mask_folder):
    train_dataset = MyDataset(train_image_folder,train_mask_folder,transform=transforms)
    val_dataset = MyDataset(val_image_folder,val_mask_folder,transform=transforms)

    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)

    return train_loader,val_loader

if __name__ == '__main__':
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="Uterus-Unet",
        
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": 1e-4,
    #     "architecture": "Controlnet",
    #     "dataset": "Uterus_dataset",
    #     "epochs": 30,
    #     }
    # )
     	
    train_image_folder = r'Data/train/images'
    train_mask_folder = r'Data/train/masks'
    val_image_folder = r'Data/validation/images'
    val_mask_folder = r'Data/validation/masks'


    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

    net = Unet(3,2)
    net.to(device)

    model_name = "Unet"
    weight_path = r'checkpoint/Unet/Unet_.pth'
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path,map_location='cpu'))

    epochs = 500

    train_loader,val_loader = DataProcess(train_image_folder,train_mask_folder,val_image_folder,val_mask_folder)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(),1e-3,weight_decay=1e-8)

    train(net,device,epochs,train_loader,val_loader,criterion,optimizer,model_name=model_name,deep_supervision=False)