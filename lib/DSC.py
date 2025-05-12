# import os 
# os.environ['CUDA_VISIBLE_DEVICES']='3'
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from lib.irnn import irnn



def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

class Spacial_IRNN(nn.Module):
    def __init__(self,in_channels,alpha=1.0):
        super(Spacial_IRNN,self).__init__()
        self.left_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.right_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.up_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.down_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.left_weight.weight  = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))
        self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))
        self.up_weight.weight    = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))
        self.down_weight.weight  = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))

    def forward(self,input):
        return irnn.apply(input,self.up_weight.weight,self.right_weight.weight,self.down_weight.weight,self.left_weight.weight, self.up_weight.bias,self.right_weight.bias,self.down_weight.bias,self.left_weight.bias)

class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.out_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels,4,kernel_size=1,padding=0,stride=1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out

class DSC_Module(nn.Module):
    def __init__(self,in_channels,out_channels,attention=1,alpha=1.0):
        super(DSC_Module,self).__init__()
        self.out_channels = out_channels
        self.irnn1 = Spacial_IRNN(self.out_channels,alpha)
        self.irnn2 = Spacial_IRNN(self.out_channels,alpha)
        self.conv_in = conv1x1(in_channels,in_channels)
        self.conv2 = conv1x1(in_channels*4,in_channels)
        self.conv3 = conv1x1(in_channels*4,out_channels)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        
        
    
    def forward(self,x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv_in(x)
        top_up,top_right,top_down,top_left = self.irnn1(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv2(out)
        top_up,top_right,top_down,top_left = self.irnn2(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        
        return out

class LayerConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(LayerConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)

        return x




class Predict(nn.Module):
    def __init__(self, in_planes=32, out_planes=1, kernel_size=1):
        super(Predict, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size)

    def forward(self, x):
        y = self.conv(x)

        return y
    

if __name__ == '__main__':
    device = device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # x = torch.Tensor(4,3,256,256).to(device)
    # x = model(x)
    # print(len(x))
    # print(x[0].shape,x[1].shape,x[2].shape,x[3].shape,x[4].shape,x[5].shape,x[6].shape)

 