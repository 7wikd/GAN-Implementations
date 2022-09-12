import torch
import torch.nn as nn

'''
Building Blocks
ConvBlock: Basic Structure of the block, highly modular
ResidualBlock: Each singular block from the 'B' Residual Blocks in the original paper
UpSampleBlock: 
'''

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, disc=False, use_act=True, use_bn=True, **kwargs):
        super().__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (nn.LeakyReLU(0.2, inplace=True) if disc else nn.PReLU(num_parameters=out_channels))
    
    def forward(x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels.in_channels*scale_factor**2,3,1,1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_channels)
    
    def forward(x):
        return self.act(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(in_channels, in_channels,kernel_size=3,stride=1,padding=1)
        self.block2 = ConvBlock(in_channels, in_channels,kernel_size=3,stride=1,padding=1, use_act=False)

    def forward(x):
        out = self.block1(x)
        out = self.block2(out)
        return out+x


c