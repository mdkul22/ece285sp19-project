import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
#import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import nntools as nt

class YoloLoss(nt.NeuralNetwork):

    def __init__(self):
        super(YoloLoss, self).__init__()
        
    
    def compute_iou(self, box1, box2):
        """
            cited from: https://github.com/xiongzihua/pytorch-YOLO-v1/blob/master/yoloLoss.py
        """
        N = box1.size(0)
        M = box2.size(0)
        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou   
    
    def return_loss(self, gtset, y):
        
        # localisation loss calculation
        lambda_coord = 5
        y[:][:][1] - gtset[:][:
        
        
        
    
class Yolo(YoloLoss):

    def __init__(self):
        super(Yolo, self).__init__()
        self.model = nn.Sequential(
                    nn.Conv2d(3, 64, 7, 2, 3),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(192, 192, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(256, 128, 1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(256, 256, 1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(512, 256, 1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(512, 256, 1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(512, 256, 1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(512, 256, 1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(512, 512, 1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(512, 1024, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(1024, 512, 1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(1024, 1024, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(1024, 512, 1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(1024, 1024, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(1024, 1024, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(1024, 1024, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(1024, 1024, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True)
                    )
        self.weight_init(self.model)
        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Linear(1024, 4096))
        self.classifier.append(nn.ReLU(inplace=True))
        self.classifier.append(nn.Dropout(p=0.5))
        self.classifier.append(nn.Linear(4096, 30))
        
 
    def weight_init(self, ms):
        for m in ms.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
    
    def forward(self, x):
        h = self.model(x)
        for k in range(len(self.classifier)):
            h = self.classifier[k](h)
        print(h.shape)
        return h
    
    def loss(self):
        pass
        
if __name__ == '__main__':
    x = Network()             
