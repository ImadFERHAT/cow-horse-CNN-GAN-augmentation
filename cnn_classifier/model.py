import torch
from torch import nn

class myCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.FirstLayer = nn.Conv2d(3,16,5)
        self.SecondLayer = nn.Conv2d(16,32,5)
        self.ThirdLayer = nn.MaxPool2d(2)
        self.FourthLayer = nn.Linear(32*124*124, 2)

    def forward(self,x):
        x = self.FirstLayer(x)
        x = self.SecondLayer(x)
        x = self.ThirdLayer(x)
        x = x.view(x.size(0),-1)
        x = self.FourthLayer(x)
        return x
