'''
Generator: 
    - U-Net Architecture
    - Has an Encoder and Decoder
    - Uses Skip Connections

    Encoder:
    - Input:  256 x 256 x 3
    - Final Output: 1 x 1 x 512
    - Uses 8 Blocks: Conv2d(stride=2) -> BatchNorm2d -> LeakyReLU
    
    Bottleneck: 1 x 1 x 512

    Decoder: 
    - Input: 1 x 1 x 512 
    - Output: 256 x 256 x 3
    - Uses 8 Blocks: ConvTranspose2d -> BatchNorm2d -> ReLU
    * Dropout for the first 3 blocks only
Discriminator:

'''

import torch.nn as nn
import torch
class ConvBlock(nn.Module):
    def __init__(self,in_features,out_features, use_dropout=False, isEncoder=True):
        super(ConvBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, 4, 2, 1, bias=False, padding_mode='reflect')
            if isEncoder 
            else nn.ConvTranspose2d(in_features, out_features, 4, 2, 1, bias=False),

            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2) if isEncoder else nn.ReLU(),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.isEncoder = isEncoder

    def forward(self,x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1,padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        self.e2 = ConvBlock(64, 128, isEncoder=True)
        self.e3 = ConvBlock(128, 256, isEncoder=True)
        self.e4 = ConvBlock(256, 512, isEncoder=True)
        self.e5 = ConvBlock(512, 512, isEncoder=True)
        self.e6 = ConvBlock(512, 512, isEncoder=True)
        self.e7 = ConvBlock(512, 512, isEncoder=True)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4,2,1),
            nn.ReLU(),
        )

        self.d1 = ConvBlock(512, 512, isEncoder=False, use_dropout=True)
        self.d2 = ConvBlock(1024, 512, isEncoder=False, use_dropout=True)
        self.d3 = ConvBlock(1024, 512, isEncoder=False, use_dropout=True)
        self.d4 = ConvBlock(1024, 512, isEncoder=False)
        self.d5 = ConvBlock(1024, 256, isEncoder=False)
        self.d6 = ConvBlock(512, 128, isEncoder=False)
        self.d7 = ConvBlock(256, 64, isEncoder=False)
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self,x):
        down1 = self.e1(x)
        down2 = self.e2(down1)
        down3 = self.e3(down2)
        down4 = self.e4(down3)
        down5 = self.e5(down4)
        down6 = self.e6(down5)
        down7 = self.e7(down6)
        
        bottleneck = self.bottleneck(down7)
        
        up1 = self.d1(bottleneck)
        up2 = self.d2(torch.cat([up1, down7], 1))
        up3 = self.d3(torch.cat([up2, down6], 1))
        up4 = self.d4(torch.cat([up3, down5], 1))
        up5 = self.d5(torch.cat([up4, down4], 1))
        up6 = self.d6(torch.cat([up5, down3], 1))
        up7 = self.d7(torch.cat([up6, down2], 1))
        
        return self.d8(torch.cat([up7, down1], 1))


class Block(nn.Module):
    def __init__(self,in_features,out_features,stride):
        super(Block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, 4, stride, 1,bias=False,padding_mode="reflect"),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        block1 = Block(64, 128, stride=2)
        block2 = Block(128, 256, stride=2)
        block3 = Block(256, 512, stride=1)
        block4 = nn.Conv2d(512, 1, 4,stride=1,padding=1,padding_mode="reflect")

        self.model = nn.Sequential(
            block1,
            block2,
            block3,
            block4
        )

    def forward(self,x,y):
        x = torch.cat([x,y],1)
        x = self.initial(x)
        x = self.model(x)
        return x





def disc_test():
    x_disc = torch.randn((1, 3, 256, 256))
    y_disc = torch.randn((1, 3, 256, 256))
    model_disc = Discriminator()
    preds_disc = model_disc(x_disc, y_disc)
    print(f"====== Discriminator Model: ======= \n{model_disc}")
    print(preds_disc.shape)

def gen_test():
    x_gen = torch.randn((1, 3, 256, 256))
    model_gen = Generator()
    preds_gen = model_gen(x_gen)
    print(f"====== Generator Model: ======= \n{model_gen}")
    print(preds_gen.shape)
    print("\n")


if __name__ == "__main__":
    gen_test()
    disc_test()
        