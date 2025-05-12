import torch
import torch.nn as nn
import torch.nn.functional as F
   
class DoubleConv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.double_conv(x)
    
class Down(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Down,self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(ch_in,ch_out)
        )

    def forward(self,x):
        return self.down(x)
    
class Up(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Up,self).__init__()
        self.up = nn.ConvTranspose2d(ch_in,ch_in//2,kernel_size=2,stride=2)
        self.conv = DoubleConv(ch_in,ch_out)

    def forward(self,x1,x2):
        x1 = self.up(x1)

        diffY = torch.tensor(x2.size()[2]-x1.size()[2])
        diffX = torch.tensor(x2.size()[3]-x1.size()[3])

        x1 = F.pad(x1,[diffX,diffX-diffX//2,diffY,diffY-diffY//2])
        x = torch.cat([x2,x1],dim=1)

        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(OutConv,self).__init__()
        self.outconv = nn.Conv2d(ch_in,ch_out,kernel_size=1)
        
    def forward(self,x):
        return self.outconv(x)

class Unet(nn.Module):
    def __init__(self,ch_in,n_classes):
        super(Unet,self).__init__()

        self.inc = DoubleConv(ch_in,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,1024)

        self.up1 = Up(1024,512)
        self.up2 = Up(512,256)
        self.up3 = Up(256,128)
        self.up4 = Up(128,64)

        self.outc = OutConv(64,n_classes)
    
    def forward(self,x):
        x1= self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)

        x = self.outc(x)
        return x



