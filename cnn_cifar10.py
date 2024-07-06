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

num_classes = 10
num_epochs = 10 # Adjusted for demonstration
batch_size = 256
mot="cnn_cnn_cifar10"


# Instantiate your new model
model = CNN(num_classes)

# Move your model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
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

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)


train_loader, val_loader,test_loader=manage_data_set(train_dataset,test_dataset,batch_size)
train_losses,val_losses=excute(model,device,train_loader,optimizer,criterion,val_loader,num_epochs,mot)


model.load_state_dict(torch.load(mot+".pth"))
pred_array = evaluate(model, device, test_loader, criterion,num_classes,mot)

