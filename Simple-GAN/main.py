from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import *
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim

########### Default Parameters ############
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 3e-4
Z_DIM = 64
IMG_DIM = 28*28*1
BATCH_SIZE = 64
NUM_EPOCHS = 10
###########################################

disc = Discriminator(IMG_DIM).to(DEVICE)
gen = Generator(Z_DIM,IMG_DIM).to(DEVICE)
NOISE = torch.randn((BATCH_SIZE,Z_DIM)).to(DEVICE)

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = MNIST(root='dataset/',transform=transforms,download=True)
loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
opt_disc = optim.Adam(disc.parameters(),lr=LR)
opt_gen = optim.Adam(gen.parameters(),lr=LR)
bce_loss = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

for e in range(NUM_EPOCHS):
    for batch_idx, (real,_) in enumerate(loader):
        real = real.view(-1,784).to(DEVICE)
        batch_size = real.shape[0]

        noise = torch.randn(batch_size,Z_DIM).to(DEVICE)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        loss_disc_real = bce_loss(disc_real,torch.ones_like(disc_real)) 
        disc_fake = disc(fake).view(-1)
        loss_disc_fake = bce_loss(disc_fake,torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_fake+loss_disc_real) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()


        output = disc(fake).view(-1)
        loss_gen = bce_loss(output,torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch: {e}/{NUM_EPOCHS}, Batch: {batch_idx}/{len(loader)} \
                    Disc Loss: {loss_disc:.4f}, Gen Loss: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(NOISE).reshape(-1,1,28,28)
                data = real.reshape(-1,1,28,28)
                fake_img_grid = torchvision.utils.make_grid(fake,normalize=True)
                real_img_grid = torchvision.utils.make_grid(real,normalize=True)

                writer_fake.add_image("Fake Images", fake_img_grid)
                writer_real.add_image("Real Images", real_img_grid)

                step += 1