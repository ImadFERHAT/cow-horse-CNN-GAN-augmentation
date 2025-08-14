import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.FirstLayer = nn.Conv2d(3,16,5)
        self.SecondLayer = nn.Conv2d(16,32,5)
        self.ThirdLayer = nn.Linear(32*56*56,1)
        self.bn = nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm2d(32)
        self.Relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = self.bn(self.FirstLayer(x))
        x = self.Relu(x)
        x = self.bn1(self.SecondLayer(x))
        x = self.Relu(x)
        x = self.flatten(x)
        x = self.ThirdLayer(x)
        return torch.sigmoid(x)
