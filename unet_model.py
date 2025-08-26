import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet3Level(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters
        # Encoder
        self.enc1 = DoubleConv(in_channels, f)        # 200x300 -> 200x300
        self.pool1 = nn.MaxPool2d(2)                  # -> 100x150
        self.enc2 = DoubleConv(f, f*2)                # -> 100x150
        self.pool2 = nn.MaxPool2d(2)                  # -> 50x75
        self.enc3 = DoubleConv(f*2, f*4)              # -> 50x75
        self.pool3 = nn.MaxPool2d(2)                  # -> 25x37 or 25x37

        # Bottleneck
        self.bottleneck = DoubleConv(f*4, f*8)        # -> 25x37

        # Decoder
        self.up3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)  # -> 50x74 or 50x74
        self.dec3 = DoubleConv(f*8, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)  # -> 100x148
        self.dec2 = DoubleConv(f*4, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)    # -> 200x296
        self.dec1 = DoubleConv(f*2, f)

        # Final conv: привести к одному каналу
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder + skip connections (выровнять размеры при необходимости)
        u3 = self.up3(b)
        if u3.size() != e3.size():
            u3 = F.interpolate(u3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        if u2.size() != e2.size():
            u2 = F.interpolate(u2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.size() != e1.size():
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.final_conv(d1)
        return out
