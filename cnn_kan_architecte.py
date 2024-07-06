import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score
import torch.nn.functional as F
from cnn_kan_architecte import *  # Make sure this file is available
from kan import KANLinear  # Make sure this file is available
from tictoc import tic,toc
import time
from torchvision import datasets, transforms, models



class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.selu(self.fc1(x))
        x = self.fc2(x)
        return x


 

    
class KAN(nn.Module):
    def __init__(self, num_classes):
        super(KAN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.kan1 = KANLinear(64 * 8 * 8, 256)
        self.kan2 = KANLinear(256, num_classes)
        
        # Initialize cumulative time counters
        self.cumulative_time1 = 0
        self.cumulative_time2 = 0

    def reset_timers(self):
        self.cumulative_time1 = 0
        self.cumulative_time2 = 0

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)

        # Timing the execution of self.kan1(x)
        start_time_fc1 = time.time()
        x = self.kan1(x)
        end_time_fc1 = time.time()
        fc1_time = end_time_fc1 - start_time_fc1
        self.cumulative_time1 += fc1_time

        # Timing the execution of self.kan2(x)
        start_time_fc2 = time.time()
        x = self.kan2(x)
        end_time_fc2 = time.time()
        fc2_time = end_time_fc2 - start_time_fc2
        self.cumulative_time2 += fc2_time

        return x
    


class KAN1(nn.Module):
    def __init__(self, num_classes):
        super(KAN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.kan1 = KANLinear(128 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = F.selu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.kan1(x)
        return x

    
class KAN(nn.Module):
    def __init__(self, num_classes):
        super(KAN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.kan1 = KANLinear(64 * 8 * 8, 256)
        self.kan2 = KANLinear(256, num_classes)
    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.kan1(x)
        x = self.kan2(x)
        return x      
    
class KAN3(nn.Module):
    def __init__(self, num_classes):
        super(KAN3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.kan1 = KANLinear(64 * 8 * 8, 256)
        self.kan2 = KANLinear(256, 256)
        self.kan3 = KANLinear(256, num_classes)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.kan1(x)
        x = self.kan2(x)
        x = self.kan3(x)
        return x     
    
class KAN4(nn.Module):
    def __init__(self, num_classes):
        super(KAN4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.kan1 = KANLinear(64 * 8 * 8, 256)
        self.kan2 = KANLinear(256, 256)
        self.kan3 = KANLinear(256, 256)
        self.kan4 = KANLinear(256, num_classes)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.kan1(x)
        x = self.kan2(x)
        x = self.kan3(x)
        x = self.kan4(x)
        return x         

class KAN5(nn.Module):
    def __init__(self, num_classes):
        super(KAN5, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.kan1 = KANLinear(64 * 8 * 8, 256)
        self.kan2 = KANLinear(256, 256)
        self.kan3 = KANLinear(256, 256)
        self.kan4 = KANLinear(256, 256)
        self.kan5 = KANLinear(256, num_classes)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.kan1(x)
        x = self.kan2(x)
        x = self.kan3(x)
        x = self.kan4(x)
        x = self.kan5(x)
        return x         
    
    
class KANCNN(nn.Module):
    def __init__(self, num_classes):
        super(KANCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.kan1 = KANLinear(64 * 8 * 8, 256)
        self.kan2 = nn.Linear(256, num_classes)
        
        # Initialize cumulative time counters
        self.cumulative_time1 = 0
        self.cumulative_time2 = 0

    def reset_timers(self):
        self.cumulative_time1 = 0
        self.cumulative_time2 = 0

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)

        # Timing the execution of self.kan1(x)
        start_time_fc1 = time.time()
        x = self.kan1(x)
        end_time_fc1 = time.time()
        fc1_time = end_time_fc1 - start_time_fc1
        self.cumulative_time1 += fc1_time

        # Timing the execution of self.kan2(x)
        start_time_fc2 = time.time()
        x = self.kan2(x)
        end_time_fc2 = time.time()
        fc2_time = end_time_fc2 - start_time_fc2
        self.cumulative_time2 += fc2_time

        return x
    
class CNNKAN(nn.Module):
    def __init__(self, num_classes):
        super(CNNKAN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.kan1 = nn.Linear(64 * 8 * 8, 256)
        self.kan2 = KANLinear(256, num_classes)
        
        # Initialize cumulative time counters
        self.cumulative_time1 = 0
        self.cumulative_time2 = 0

    def reset_timers(self):
        self.cumulative_time1 = 0
        self.cumulative_time2 = 0

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)

        # Timing the execution of self.kan1(x)
        start_time_fc1 = time.time()
        x = self.kan1(x)
        end_time_fc1 = time.time()
        fc1_time = end_time_fc1 - start_time_fc1
        self.cumulative_time1 += fc1_time

        # Timing the execution of self.kan2(x)
        start_time_fc2 = time.time()
        x = self.kan2(x)
        end_time_fc2 = time.time()
        fc2_time = end_time_fc2 - start_time_fc2
        self.cumulative_time2 += fc2_time

        return x
    
    
