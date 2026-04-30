import torch.nn as nn
import torch
class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim=1, depth=5):
        super(Discriminator, self).__init__()
        layers = []
        for i in range(depth - 1):
            layers.append(ConvLayer(in_dim, 2 * in_dim))
            in_dim = 2 * in_dim
        layers.append(ConvLayer(in_dim, out_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
