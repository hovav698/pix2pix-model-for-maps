from torch import nn

#the convolution blocks that used in the generator and discriminator models

class downsample(nn.Module):
    def __init__(self, input_size, filters, kernel_size, apply_batchnorm=True):
        super(downsample, self).__init__()
        modules = []
        modules.append(nn.Conv2d(input_size, filters, kernel_size, stride=2, padding=1, bias=False))

        if apply_batchnorm:
            modules.append(nn.BatchNorm2d(filters))

        modules.append(nn.LeakyReLU())
        self.sequential = nn.Sequential(*modules)

    def forward(self, img):
        return self.sequential(img)


class upsample(nn.Module):
    def __init__(self, input_size, filters, kernel_size, apply_dropout=False):
        super(upsample, self).__init__()
        modules = []
        modules.append(nn.ConvTranspose2d(input_size, filters, kernel_size, stride=2, bias=False, padding=1))
        modules.append(nn.BatchNorm2d(filters))
        modules.append(nn.LeakyReLU())

        if apply_dropout:
            modules.append(nn.Dropout(0.5))

        self.sequential = nn.Sequential(*modules)

    def forward(self, img):
        return self.sequential(img)
