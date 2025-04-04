import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=1, base_channels=64):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels*2, 3, padding=1)
        self.enc3 = nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1)

        self.middle = nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1)

        self.dec3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.out = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x_noisy, cond_img):
        x = torch.cat([x_noisy, cond_img], dim=1)

        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(F.max_pool2d(e1, 2)))
        e3 = F.relu(self.enc3(F.max_pool2d(e2, 2)))

        mid = F.relu(self.middle(e3))

        d3 = F.relu(self.dec3(mid))
        d2 = F.relu(self.dec2(d3 + e2))
        out = self.out(d2 + e1)

        return out
