import torch
from dataset import MyDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from model import Generator, Discriminator

def train_fn(disc_P, disc_M, gen_M, gen_P, loader, opt_disc,opt_gen, l1, mse, d_scaler, g_scaler):
    P_reals = 0
    P_fakes = 0

    loop = tqdm(loader, leave=True)

    for idx, (monet,photo) in enumerate(loop):
        monet = monet.to(config.DEVICE)
        photo = photo.to(config.DEVICE)

    # DISCRIMINATOR
        with torch.cuda.amp.autocast():
            fake_photo = gen_P(monet)
            
            D_P_real = disc_P(photo)
            D_P_fake = disc_P(fake_photo.detach())
            
            P_reals += D_P_real.mean().item()
            P_fakes += D_P_fake.mean().item()
            
            D_P_real_loss = mse(D_P_real,torch.ones_like(D_P_real))
            D_P_fake_loss = mse(D_P_fake,torch.zeros_like(D_P_fake))
            
            D_P_loss = D_P_fake_loss + D_P_real_loss

            fake_monet = gen_M(photo)
            
            D_M_real = disc_M(monet)
            D_M_fake = disc_M(fake_monet.detach())
            
            D_M_real_loss = mse(D_M_real,torch.ones_like(D_M_real))
            D_M_fake_loss = mse(D_M_fake,torch.zeros_like(D_P_fake))
            
            D_M_loss = D_M_fake_loss + D_M_real_loss

            D_loss = (D_P_loss+D_M_loss)/2
        
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
    
    # GENERATOR        
        with torch.cuda.amp.autocast():
            D_P_fake = disc_P(fake_photo)
            D_M_fake = disc_M(fake_monet)

            loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))
            loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))

            # cycle consistency loss

            cycle_monet = gen_M(fake_photo)
            cycle_photo = gen_P(fake_monet)

            cycle_monet_loss = l1(monet,cycle_monet)
            cycle_photo_loss = l1(photo,cycle_photo)

            # identity loss 

            identity_monet = gen_M(monet)
            identity_photo = gen_P(photo)

            identity_monet_loss = l1(monet,identity_monet)
            identity_photo_loss = l1(photo,identity_photo)

            G_loss = (
                loss_G_M
                + loss_G_P
                + cycle_monet_loss * config.LAMBDA_CYCLE
                + cycle_photo_loss * config.LAMBDA_CYCLE
                + identity_monet_loss * config.LAMBDA_IDENTITY
                + identity_photo_loss * config.LAMBDA_IDENTITY
            )
        
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_photo*0.5+0.5,f"./saved_images/photo/photo_{idx}.png")
            save_image(fake_monet*0.5+0.5,f"./saved_images/monet/monet_{idx}.png")

        loop.set_postfix()

def main():
    disc_P = Discriminator(in_channels=3).to(config.DEVICE)
    disc_M = Discriminator(in_channels=3).to(config.DEVICE)

    gen_P = Generator(img_channels=3,num_residuals=9).to(config.DEVICE)
    gen_M = Generator(img_channels=3,num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_P.parameters()) + list(disc_M.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5,0.999),
    )
    opt_gen = optim.Adam(
        list(gen_M.parameters()) + list(gen_P.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5,0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_P, 
            gen_P,
            opt_gen,
            lr = config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_M, 
            gen_M,
            opt_gen,
            lr = config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_P, 
            disc_P,
            opt_disc,
            lr = config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_M, 
            disc_M,
            opt_disc,
            lr = config.LEARNING_RATE
        )
    
    dataset = MyDataset(
        root_monet = config.TRAIN_DIR+"Dataset/Photo", 
        root_photo = config.TRAIN_DIR+"Dataset/Monet",
        transform= config.transforms
    )

    loader = DataLoader(
        dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = False,
        num_workers = config.NUM_WORKERS,
        pin_memory = True,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_P, disc_M, gen_M, gen_P, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_P, opt_gen, filename=config.CHECKPOINT_GEN_P)
            save_checkpoint(gen_M, opt_gen, filename=config.CHECKPOINT_GEN_M)
            save_checkpoint(disc_P, opt_disc, filename=config.CHECKPOINT_DISC_P)
            save_checkpoint(disc_M, opt_disc, filename=config.CHECKPOINT_DISC_M)


if __name__ == "__main__":
    main()