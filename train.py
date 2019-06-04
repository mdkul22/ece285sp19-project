import torch
import nntools as nt
import os
import numpy as np
import torch.nn as nn
import torch.utils.data as td
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
from dataloader import VOCDataset, myimshow
import model
class statsmanager(nt.StatsManager):
    def __init__self():
        super(statsmanager,self).__init__()

    def init(self):
        super(statsmanager,self).init()
        self.m_ap=0

    def accumulate(self,loss,x,y,d):
        #Do m_ap calculations
        pass

    def summarize(self):
        loss=super(statsmanager,self).summarize()
        return {'loss':loss}

def plot(self,fig,ax1, ax2 ,im):
    ax1.set_title('Image')
    x,y=train_set[0]
    myimshow(x,ax=ax1)
    ax2.set_title('Loss')
    ax2.plot([exp1.history[k]['loss']for k in range(exp1.epoch)])
    plt.tight_layout()
    fig.canvas.draw()

lr=1e-3 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
vgg = model.VGGTransfer(20)
#vgg.to(device)         
adam=torch.optim.Adam(vgg.parameters(),lr=lr)
stats_manager=statsmanager()
train_set=VOCDataset('/home/mdk/Desktop/mlip_285/VOCdevkit/VOC2012')
x,y=train_set[0]
exp1=nt.Experiment(vgg,train_set,train_set,adam,stats_manager,batch_size=4,output_dir="run1",perform_validation_during_training=False)
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
exp1.run(num_epochs=1,plot=lambda exp:plot(exp,fig=fig,ax1=ax1, ax2=ax2 ,im=x))

