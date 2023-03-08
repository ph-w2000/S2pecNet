import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import random
import numpy as np


class Residual_block(nn.Module):
    def __init__(self, in_planes, planes, index):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_planes)
        self.conv1 = nn.ConvTranspose2d(in_channels=in_planes,
                               out_channels=planes,
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.conv2 = nn.ConvTranspose2d(in_channels=planes,
                               out_channels=planes,
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        
        self.conv3 = nn.ConvTranspose2d(in_channels=planes,
                               out_channels=planes,
                               kernel_size=(1, 4),
                               padding=0,
                               stride=(1,3))
        

    def forward(self, x):
        out = self.bn1(x)
        out = self.selu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride==1:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=stride,  bias=False)
            self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=1, stride=1, padding=1, bias=False)
            
        else:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=stride, output_padding=1, bias=False)
            self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=1, stride=1, padding=1,  bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out


class RawDecoder(nn.Module):    
    def __init__(self):
        super().__init__()
        block1 = Residual_block(64,64,1)
        block2 = Residual_block(64,64,2)
        block3 = Residual_block(64,64,3)
        block4 = Residual_block(64,32,4)
        block5 = Residual_block(32,32,5)
        block6 = Residual_block(32,1,6)

        self.decoder_conv = nn.Sequential(
            block1,
            block2,
            block3,
            block4,
            block5,
            block6,
            nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=(3, 3), padding=(0,0),output_padding=1, bias=False),
            nn.ConvTranspose2d(1, 1, kernel_size=(1, 85), stride=1, padding=(0,0),bias=False),
            nn.Conv2d(1, 1, kernel_size=(70, 1), stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.SiLU()
        )
        
        
    def forward(self, x):
        x = self.decoder_conv(x).squeeze()
        return x


class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.dec_lin = nn.Linear(29, 117)
        
        block1 = PreActBlock(64,64,1)
        block2 = PreActBlock(64,32,1)
        block3 = PreActBlock(32,32,1)
        block4 = PreActBlock(32,16,1)
        block5 = PreActBlock(16,16,1)
        block6 = PreActBlock(16,8,2)
        block7 = PreActBlock(8,8,1)
        block8 = PreActBlock(8,8,1)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 8, stride=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            block1,
            block2,
            block3,
            block4,
            block5,
            block6,
            block7,
            block8,
            nn.ConvTranspose2d(8, 1, kernel_size=(3, 9), stride=(1, 3), padding=(1, 0), bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )  
        
        
    def forward(self, x):
        x = self.dec_lin(x)
        x = self.decoder_conv(x)   
        x = torch.sigmoid(x)
        return x