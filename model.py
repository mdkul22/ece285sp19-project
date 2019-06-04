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

    def __init__(self,n_batch):
        super(YoloLoss, self).__init__()
        self.B = 2
        self.C = 20       
        self.use_gpu=1        
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.n_batch=n_batch    
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
    
    def return_loss(self, pred_tensor, target_tensor):
        
        print("pred_tensor: ", pred_tensor.shape)
        print("target_tensor: ", target_tensor.shape)
        # localisation loss calculation
        n_elements = self.B * 5 + self.C
        batch = target_tensor.size(0)
        target_tensor = target_tensor.view(batch,-1,n_elements)
        #print(target_tensor.size())
        #print(pred_tensor.size())
        pred_tensor = pred_tensor.view(batch,-1,n_elements)
        coord_mask = target_tensor[:,:,5] > 0
        noobj_mask = target_tensor[:,:,5] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)

        coord_target = target_tensor[coord_mask].view(-1,n_elements)
        coord_pred = pred_tensor[coord_mask].view(-1,n_elements)
        class_pred = coord_pred[:,self.B*5:]
        class_target = coord_target[:,self.B*5:]
        box_pred = coord_pred[:,:self.B*5].contiguous().view(-1,5)
        box_target = coord_target[:,:self.B*5].contiguous().view(-1,5)

        noobj_target = target_tensor[noobj_mask].view(-1,n_elements)
        noobj_pred = pred_tensor[noobj_mask].view(-1,n_elements)

        # compute loss which do not contain objects
        if self.use_gpu:
            noobj_target_mask = torch.cuda.ByteTensor(noobj_target.size())
        else:
            noobj_target_mask = torch.ByteTensor(noobj_target.size())
        noobj_target_mask.zero_()
        for i in range(self.B):
            noobj_target_mask[:,i*5+4] = 1
        noobj_target_c = noobj_target[noobj_target_mask] # only compute loss of c size [2*B*noobj_target.size(0)]
        noobj_pred_c = noobj_pred[noobj_target_mask]
        noobj_loss = F.mse_loss(noobj_pred_c, noobj_target_c, size_average=False)

        # compute loss which contain objects
        if self.use_gpu:
            coord_response_mask = torch.cuda.ByteTensor(box_target.size())
            coord_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        else:
            coord_response_mask = torch.ByteTensor(box_target.size())
            coord_not_response_mask = torch.ByteTensor(box_target.size())
        coord_response_mask.zero_()
        coord_not_response_mask = ~coord_not_response_mask.zero_()
        for i in range(0,box_target.size()[0],self.B):
            box1 = box_pred[i:i+self.B]
            box2 = box_target[i:i+self.B]
            iou = self.compute_iou(box1[:, :4], box2[:, :4])
            max_iou, max_index = iou.max(0)
            if self.use_gpu:
                max_index = max_index.data.cuda()
            else:
                max_index = max_index.data
            coord_response_mask[i+max_index]=1
            coord_not_response_mask[i+max_index]=0

        # 1. response loss
        box_pred_response = box_pred[coord_response_mask].view(-1, 5)
        box_target_response = box_target[coord_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response[:, 4], size_average=False)
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) +\
                   F.mse_loss(box_pred_response[:, 2:4], box_target_response[:, 2:4], size_average=False)
        # 2. not response loss
        box_pred_not_response = box_pred[coord_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coord_not_response_mask].view(-1, 5)

        # compute class prediction loss
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)

        # compute total loss
        total_loss = self.lambda_coord * loc_loss + contain_loss + self.lambda_noobj * noobj_loss + class_loss
        return total_loss
        
    def criterion(self,y,d):
        return self.return_loss(y,d)    
    
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


class VGGTransfer(YoloLoss):
    
    def __init__(self, num_classes,n_batch,fine_tuning=False):
        super(VGGTransfer, self).__init__(n_batch)
        vgg = tv.models.vgg16_bn(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = fine_tuning
        self.features = vgg.features
        self.classifier = nn.Sequential(nn.Linear(25088,4096),nn.ReLU(True),nn.Dropout(),nn.Linear(4096,1470),)

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0),-1)
        y = self.classifier(f)
        return y    



if __name__ == '__main__':
    x = Network()             
