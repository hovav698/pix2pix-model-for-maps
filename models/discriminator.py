from torch import nn
import torch
from models.blocks import downsample

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            downsample(6, 64, 4, False),
            downsample(64, 128, 4),
            downsample(128, 256, 4),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, 4, stride=1),
            nn.Sigmoid()
        )

    def forward(self, inp, tar):
        x = torch.cat([inp, tar], dim=1)
        output = self.model(x)

        return output
