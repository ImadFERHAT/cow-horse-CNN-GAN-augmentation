import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import myCNN

# Define transformations
trf = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(45),
    transforms.Normalize(0,1),
    transforms.functional.invert
])

# Dataset and dataloaders
train_data_path = "drive/MyDrive/MVP/Newdata/Train"
test_data_path = "drive/MyDrive/MVP/Newdata/Test"
train_set = datasets.ImageFolder(root=train_data_path, transform=trf)
test_set = datasets.ImageFolder(root=test_data_path, transform=trf)

batch_size = 41
train_loader = DataLoader(train_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# Model, loss, optimizer
model = myCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Training and testing functions
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for X, y in dataloader:
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
    print(f"Accuracy: {100*correct/total:.2f}%")
