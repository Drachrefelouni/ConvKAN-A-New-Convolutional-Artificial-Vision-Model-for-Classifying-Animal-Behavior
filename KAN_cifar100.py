#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:58:40 2024

@author: achref
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score
import torch.nn.functional as F
from cnn_kan_architecte import *
from kan import KANLinear
import time
from implementation import *
torch.cuda.empty_cache()

num_classes = 100
num_epochs = 10  # Adjusted for demonstration
batch_size = 128
mot="cnn_kan1_cifar100"


model = KAN(num_classes)
#for test with more than layer of kan
#model = KAN3(num_classes) CNN-KAN with 3 layer
#model = KAN4(num_classes) CNN-KAN with 4 layer
#model = KAN5(num_classes) CNN-KAN with 5 layer
# Move your model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Wrap your model with DataParallel
#model = nn.DataParallel(model)

# Define your optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Load your dataset

transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform)


train_loader, val_loader,test_loader=manage_data_set(train_dataset,test_dataset,batch_size)
train_losses,val_losses=excute(model,device,train_loader,optimizer,criterion,val_loader,num_epochs,mot)


model.load_state_dict(torch.load(mot+".pth"))
pred_array = evaluate(model, device, test_loader, criterion,num_classes,mot)


