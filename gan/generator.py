import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.emergencyLayer = nn.Linear(100, 16*56*56)
        self.FirstLayer = nn.ConvTranspose2d(16,32,3)
        self.SecondLayer = nn.ConvTranspose2d(32,64,3)
        self.ThirdLayer = nn.ConvTranspose2d(64,3,5)
        self.Relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(3)

    def forward(self,x):
        x = self.emergencyLayer(x)
        x = x.view(-1,16,56,56)
        x = self.FirstLayer(x)
        x = self.bn(x)
        x = self.Relu(x)
        x = self.bn1(self.SecondLayer(x))
        x = self.Relu(x)
        x = self.bn2(self.ThirdLayer(x))
        return nn.Tanh()(x)
