import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, use_dropout=False, use_batchnorm=False):
        super(LeNet5, self).__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        self.conv1 = nn.Conv2d(1, 6, 5)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(6) 
        self.conv2 = nn.Conv2d(6, 16, 5)
        if use_batchnorm:
            self.bn2 = nn.BatchNorm2d(16) 
        self.fc1 = nn.Linear(16*4*4, 120)
        if use_dropout:
            self.dropout1 = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(120, 84)
        if use_dropout:
            self.dropout2 = nn.Dropout(0.5) 
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        if self.use_batchnorm:
            x=self.bn1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        if self.use_batchnorm:
            x=self.bn2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x=self.dropout1(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x=self.dropout2(x)
        x = self.fc3(x)
        return x