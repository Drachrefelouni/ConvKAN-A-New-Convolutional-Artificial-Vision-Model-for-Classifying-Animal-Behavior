#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:08:51 2024

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


# Define your training loop
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Start measuring time
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        # End measuring time
   
    train_loss /= len(train_loader.dataset)
    return train_loss

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    return val_loss, accuracy

def evaluate(model, device, test_loader, criterion,num_classes,mot):
    model.eval()
    test_loss = 0
    correct = 0
    pred_list = []
    target_list = []
    score_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            pred_list.extend(pred.cpu().numpy())
            target_list.extend(target.cpu().numpy())
            score_list.extend(output.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    pred_array = np.array(pred_list)
    target_array = np.array(target_list)
    score_array = np.array(score_list)

    precision = precision_score(target_array, pred_array, average='macro', zero_division=0)
    recall = recall_score(target_array, pred_array, average='macro', zero_division=0)

    # Calculate mean average precision (mAP)
    target_one_hot = np.eye(num_classes)[target_array]  # Convert targets to one-hot encoding
    map_score = average_precision_score(target_one_hot, score_array, average='macro')

    print(mot+': Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Precision: {:.4f}, Recall: {:.4f}, mAP: {:.4f}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), precision, recall, map_score))
    results = {
        "precision": precision,
        "recall": recall,
        "map_score": map_score
    }
    
    torch.save(results, mot + "_resultat.pth")
    return pred_array

# Training and validation


def excute(model,device,train_loader,optimizer,criterion,val_loader,num_epochs,mot):
    train_losses = []
    val_losses = []
    val_accuracies = []
    t=0
    best_val_loss = float('inf')
    list_t=[]
    for epoch in range(1, num_epochs + 1):
        # Reset timers at the start of each epoch
       # model.reset_timers()
        
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss, val_accuracy = validate(model, device, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        
        print('Epoch: {} Train Loss: {:.6f} Validation Loss: {:.6f}  '.format(
            epoch, train_loss, val_loss ))
        
        # Print cumulative times at the end of each epoch
       # print(f'Epoch: {epoch}')
       # print(f'Cumulative execution time for self.nn1(x): {model.cumulative_time1:.6f} seconds')
       # print(f'Cumulative execution time for self.nn2(x): {model.cumulative_time2:.6f} seconds')
       # list_t.append(model.cumulative_time1)
        
#        t=t+model.cumulative_time1
        # Save the model with the lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), mot+".pth")
            
            #print(f'Saving model with validation loss: {val_loss:.6f}')
    #torch.save(list_t, mot+"_time_kan.pth")
    return  train_losses,val_losses

def manage_dataset_Caltech(dataset):
    # Define the train-validation-test split ratio
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1
    TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE
    
    # Calculate sizes
    train_size = int(TRAIN_SIZE * len(dataset))
    val_size = int(VAL_SIZE * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Define batch size
    BATCH_SIZE = 16
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader,val_loader,test_loader

def manage_data_set(train_dataset,test_dataset,batch_size):
        # Split the training dataset into training and validation datasets
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader,test_loader