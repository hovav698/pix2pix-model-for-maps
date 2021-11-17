from torch import nn
import torch
from models.blocks import downsample, upsample


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        down_layers = [
            downsample(3, 64, 4, apply_batchnorm=False),
            downsample(64, 128, 4),
            downsample(128, 256, 4),
            downsample(256, 512, 4),
            downsample(512, 512, 4),
            downsample(512, 512, 4),
            downsample(512, 512, 4),
            downsample(512, 512, 4)
        ]

        self.down_stack = nn.ModuleList(down_layers)

        up_layers = [
            upsample(512, 512, 4, apply_dropout=True),
            upsample(1024, 512, 4, apply_dropout=True),
            upsample(1024, 512, 4, apply_dropout=True),
            upsample(1024, 512, 4),
            upsample(1024, 256, 4),
            upsample(512, 128, 4),
            upsample(256, 64, 4)
        ]

        self.up_stack = nn.ModuleList(up_layers)

        self.last = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, img):
        x = img
        skips = []
        for down in self.down_stack:
            x = down(x)
            # print("Down:",x.shape)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            # print("up:",x.shape)
            x = torch.cat([x, skip], dim=1)

        x = self.last(x)
        x = self.tanh(x)

        return x