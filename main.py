
# coding: utf-8


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


class Digit(nn.Module):
    def __init__(self):
        super(Digit, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.pool4 = nn.AvgPool2d(3)
        self.fc = nn.Linear(512, 10)
        
        
    def forward(self, images):
        output = self.pool1(F.relu(self.conv1(images)))
        output = self.pool2(F.relu(self.conv2(output)))
        output = self.pool3(F.relu(self.conv3(output)))
        output = self.fc(self.pool4(output).view(-1, 512))
        
        return output



class DigitDataset(Dataset):
    def __init__(self):
        self.train = train
    
    def __getitem__(self, index):
        features = train.iloc[index, 1:].values
        target = train.iloc[index, 0].values
        print(type(features))
        return features, target
    
    def __len__(self):
        return self.train.shape[0]


digit = Digit()
digit = digit.cuda()

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

optimizer = torch.optim.Adam(digit.parameters(), lr=1e-3)


DDS = DigitDataset()

train_iter = DataLoader(DDS, batch_size=128, shuffle=True, num_workers=6)


for k, (features, targets) in enumerate(train_iter):
    print(features.size())
    print(targets.size())
    pred_train = digit(Variable(features).cuda())
    y_train = Variable(targets.cuda())
    loss = loss_fn(train_pred, y_train)



