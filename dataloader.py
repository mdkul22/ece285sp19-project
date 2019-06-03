import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as td
import torchvision as tv
import xml.etree.ElementTree as ET
from PIL import Image
from matplotlib import pyplot as plt


class VOCDataset(td.Dataset):
    def __init__(self, root_dir, mode="train", image_size=(375, 500)):
        super(VOCDataset, self).__init__()
        self.image_size = image_size
        self.mode = mode
        #self.data = pd.read_csv(os.path.join(root_dir, "%s.xml" % mode))
        self.annotations_dir = os.path.join(root_dir, "Annotations")
        self.images_dir = os.path.join(root_dir, "JPEGImages")
        
        # os.listdir returns list in arbitrary order
        self.image_names = os.listdir(self.images_dir)
        self.image_names = [image.rstrip('.jpg') for image in self.image_names]
        self.voc_dict = {
                        'person':1, 'bird':2, 'cat':3, 'cow':4, 'dog':5, 
                        'horse':6, 'sheep':7, 'aeroplane':8, 'bicycle':9,
                        'boat':10, 'bus':11, 'car':12, 'motorbike':13, 'train':14, 
                        'bottle':15, 'chair':16, 'diningtable':17, 
                        'pottedplant':18, 'sofa':19, 'tvmonitor':20
                        }

        
        
    def __len__(self):
        return len(self.image_names)

    def __repr__(self):
        return "VOC2012Dataset(mode={}, image_size={})". \
            format(self.mode, self.image_size)
    
    def __getitem__(self, idx):
        # Get file paths for image and annotation (label)
        img_path = os.path.join(self.images_dir, \
                                "%s.jpg" % self.image_names[idx])
        lbl_path = os.path.join(self.annotations_dir, \
                                "%s.xml" % self.image_names[idx])   
        
        # Get objects and bounding boxes from annotations
        lbl_tree = ET.parse(lbl_path)
        objs = []
        
        
        for obj in lbl_tree.iter(tag='object'):
            name = obj.find('name').text
            for box in obj.iter(tag='bndbox'):
                xmax = box.find('xmax').text
                xmin = box.find('xmin').text
                ymax = box.find('ymax').text
                ymin = box.find('ymin').text
            attr = (self.voc_dict[name], (int(xmin)+int(xmax))/2,(int(ymin)+int(ymax))/2, int(xmax)-int(xmin), int(ymax)-int(ymin), 1)
            objs.append(attr)
         
        lab_mat=torch.zeros([len(objs),20,5])

        for i in range(len(objs)):
            for x in range(20):
                if objs[i][0] == x+1:
                    lab_mat[i][x][0]=torch.tensor([[1]])
            for x in range(20):
                lab_mat[i][x][1]=torch.tensor([[objs[i][1]]])
                lab_mat[i][x][2]=torch.tensor([[objs[i][2]]])
                lab_mat[i][x][3]=torch.tensor([[objs[i][3]]])
                lab_mat[i][x][4]=torch.tensor([[objs[i][4]]])

        objs = np.array(objs)
        # Open and normalize the image
        img = Image.open(img_path).convert('RGB')
        transform = tv.transforms.Compose([
            #tv.transforms.Resize(self.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        x = transform(img)
        _, img_sx, img_sy = x.shape
        d = objs
        target = torch.zeros((7,7,31))
        target = target.numpy() 
        for i in range(len(objs)):
            for j in range(20):
                if lab_mat[i][j][0] == 1:
                    xi = int(lab_mat[i][j][1].numpy() % 7)
                    yi = int(lab_mat[i][j][2].numpy() % 7)
                    if target[xi][yi][30] == 1:
                        target[xi][yi][5] = lab_mat[i][j][1].numpy()/img_sx
                        target[xi][yi][6] = lab_mat[i][j][2].numpy()/img_sy
                        target[xi][yi][7] = lab_mat[i][j][3].numpy()/img_sx
                        target[xi][yi][8] = lab_mat[i][j][4].numpy()/img_sy
                        target[xi][yi][9] = 1  
                        target[xi][yi][9+objs[i][0]] = 1
                    else:
                        target[xi][yi][0] = lab_mat[i][j][1].numpy()/img_sx
                        target[xi][yi][1] = lab_mat[i][j][2].numpy()/img_sy
                        target[xi][yi][2] = lab_mat[i][j][3].numpy()/img_sx
                        target[xi][yi][3] = lab_mat[i][j][4].numpy()/img_sy
                        target[xi][yi][4]   = 1
                        target[xi][yi][30]  = 1  # indicates filled slot              
                        target[xi][yi][9+objs[i][0]] = 1
        
        self.target = torch.from_numpy(np.delete(target, 30, 2))
        return x, d

    def number_of_classes(self):
        #return self.data['class'].max() + 1
        # TODO: make more flexible
        return 20
    def print_target(self):
        print(self.target)
    
def myimshow(image, ax=plt):
    image = image.to('cpu').detach().numpy()
    image = np.moveaxis(image, [0,1,2], [2,0,1])
    image = (image + 1)/2
    image[image<0] = 0
    image[image>1] = 1
    h = ax.imshow(image)        
    ax.axis('off')
    return h

if __name__ == '__main__':
    xmlparse = VOCDataset('/home/mdk/Desktop/mlip_285/VOCdevkit/VOC2012')
    print(xmlparse[0])
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    #print(axes)
    x, y = xmlparse[5]
    xmlparse.print_target()
    x1, y1 = xmlparse[1]
    myimshow(x, ax=ax1)
    myimshow(x1, ax=ax2)
    plt.show()
    
