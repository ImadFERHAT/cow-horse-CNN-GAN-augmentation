import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from generator import Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
trf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])
dataset = datasets.ImageFolder(root="drive/MyDrive/MVP/Newdata", transform=trf)
dataloader = DataLoader(dataset, batch_size=41, shuffle=True)

# Models
G = Generator().to(device)
D = Discriminator().to(device)

# Loss and optimizers
criterion = torch.nn.BCELoss()
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    for real_images, _ in dataloader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        real_labels = torch.ones(batch_size,1).to(device)
        fake_labels = torch.zeros(batch_size,1).to(device)

        # Train Discriminator
        outputs_real = D(real_images)
        noise = torch.randn(batch_size,100).to(device)
        fake_images = G(noise)
        outputs_fake = D(fake_images.detach())
        loss_D = criterion(outputs_real, real_labels) + criterion(outputs_fake, fake_labels)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        outputs_fake = D(fake_images)
        loss_G = criterion(outputs_fake, real_labels)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    # Show a generated image
    with torch.no_grad():
        img = G(torch.randn(1,100).to(device))
        img = (img+1)/2
        plt.imshow(img[0].permute(1,2,0).cpu())
        plt.show()
