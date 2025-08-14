from generator import Generator
from discriminator import Discriminator

import torch, itertools
import torch.nn as nn
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator

device = "cuda" if torch.cuda.is_available() else "cpu"

G_AB = Generator().to(device)
G_BA = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

opt_G = optim.Adam(itertools.chain(G_AB.parameters(),G_BA.parameters()), lr=2e-4, betas=(0.5,0.999))
opt_D_A = optim.Adam(D_A.parameters(), lr=2e-4, betas=(0.5,0.999))
opt_D_B = optim.Adam(D_B.parameters(), lr=2e-4, betas=(0.5,0.999))

L1 = nn.L1Loss()
BCE = nn.MSELoss()

# Training loop (use your dataloader and code)
