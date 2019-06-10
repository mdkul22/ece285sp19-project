Description
===========
This is project Multi Object Detection using YOLO developed by team MADS composed of Mayunk Kulkarni, Darren Eck, Sivasankar Palaniappan.
This implementation has been done over three networks VGG, Resnet and YOLO net. VGG has been implemented with both Adam and SGD optimization.
We trained over the PascalVOC2012 training data set and tested it on the VOC 2007 and 2012 test set.

Requirements
============
Install the following packages as follows:
$ pip install --user os
$ pip install --user numpy
$ pip install --user torch
$ pip install --user torchvision
$ pip install --user matplotlib
$ pip install --user pillow
$ pip install --user PIL

Below are the model/network links which can be downloaded.

run5 is for the vgg2 net in Demo.ipynb - which is VGG with Adam optimization.
run4 is for the vgg net in Demo.ipynb - which is the VGG with SGD optimization
newloss is for the yolo net in Demo.ipynb - which is the yolo net

Download the nets and unzip them in the folder where this repo is being cloned.

https://drive.google.com/a/eng.ucsd.edu/file/d/1w5NEFdZxOUt-pU9ONU4qQO-uet0fFSbR/view?usp=sharing - Model for run5 
https://drive.google.com/a/eng.ucsd.edu/file/d/1Gva8oRWo1nhjTI1O8ympwnT_O-piRlsm/view?usp=sharing - Model for run4
https://drive.google.com/a/eng.ucsd.edu/file/d/1zD2JaH54mEkBMBJbEmvFGG7w4PheVQy4/view?usp=sharing - Model for newloss

Code organization
=================
Demo.ipynb -- Run a demo of our code 
- View the ground truth and predicted bounding boxes of various objects in an image , along with the labels for the following:
    - Yolo net
    - VGG with Adam 
    - VGG with SGD
dataloader.py -- Python notebook to load the images for training, validation and test set and converting them to Tensors. 
- Also does the preprocessing of the tensor to obtain the target tensor.
test.ipynb -- Notebook which is used for predicting and testing an image.
model.py -- Contains the classes for all the models which have been implemented.
nntools.py -- File for running the experiment and storing and loading the checkpoints.
yolov1_scratch.ipynb -- notebook for unsuccessful implementation of yolo loss function.
train_notebooks/train.ipynb -- Notebook for training the yolo net.
train_notebooks/VGG_Pretrain_SGD.ipynb -- Notebook for training the VGG model with SGD optimization.
train_notebooks/vgg_adam.ipynb -- Notebook for training the VGG model with Adam optimization.
train_notebooks/train_resnet.ipynb -- Notebook for training the resnet with 18 layers.
train_notebooks/train_resnet34.ipynb -- Notebook for training the resnet with 34 layers.
Debugging/train.py -- Python file for debugging train.ipynb
Debugging/trainres.py -- Python file for debugging train_resnet.ipynb.
Debugging/trainres34.py -- Python file for debugginh train_resnet34.ipynb.
