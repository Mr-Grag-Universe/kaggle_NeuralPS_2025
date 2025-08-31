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


import torch
import torch.nn as nn
import torch.nn.functional as F

def get_group_norm(num_channels, num_groups=8):
    # Ensure num_groups divides num_channels
    g = min(num_groups, num_channels)
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)

class DoubleConvGN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            get_group_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            get_group_norm(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNet3LevelWithFeatures(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32,
    num_scalar_features=0, det_array_length=0, device='cpu'):
        """
        Now: one feature projection per encoder/decoder level (levels: enc1, enc2, enc3, bottleneck, dec3, dec2, dec1).
        For simplicity we reuse projection modules symmetrically: projections for encoder levels (enc1..enc3),
        one for bottleneck, and one for decoder levels (d3,d2,d1) — total 7 proj modules but to keep params small
        we'll create 4 distinct MLPs: proj_enc1, proj_enc2, proj_enc3, proj_bottleneck and reuse them where appropriate.
        """
        super().__init__()
        f = base_filters
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.det_array_length = det_array_length
        self.num_scalar_features = num_scalar_features
        # encoder
        self.enc1 = DoubleConvGN(in_channels + f, f)   # reserve space for broadcasted feat concat
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConvGN(f + f, f*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConvGN(f*2 + f, f*4)
        self.pool3 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = DoubleConvGN(f*4 + f, f*8)

        # decoder
        self.up3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConvGN(f*8 + f, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConvGN(f*4 + f, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConvGN(f*2 + f, f)

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

        # Feature projection MLPs: one per meaningful spatial level
        proj_dim = f  # channels to broadcast

        if num_scalar_features + det_array_length > 0:
            def make_mlp():
                return nn.Sequential(
                    nn.LazyLinear(proj_dim*2),
                    nn.ReLU(inplace=True),
                    nn.Linear(proj_dim*2, proj_dim),
                    nn.ReLU(inplace=True),
                )
            # create distinct MLPs for encoder levels + bottleneck + decoder group reuse
            self.proj_enc1 = make_mlp()
            self.proj_enc2 = make_mlp()
            self.proj_enc3 = make_mlp()
            self.proj_bottleneck = make_mlp()
            # reuse enc3/bottleneck MLPs for decoder levels (you can make separate ones if desired)
            self.proj_dec3 = make_mlp()
            self.proj_dec2 = make_mlp()
            self.proj_dec1 = make_mlp()
        else:
            self.proj_enc1 = self.proj_enc2 = self.proj_enc3 = self.proj_bottleneck = None
            self.proj_dec3 = self.proj_dec2 = self.proj_dec1 = None

    def resample_det_array_to_targetW(self, det_array, target_W):
        if det_array is None:
            return None
        if det_array.dim() == 2:
            det = det_array.unsqueeze(1)  # (B,1,L0)
        else:
            det = det_array
        det = det.to(self.device)
        det_res = F.interpolate(det, size=target_W, mode='linear', align_corners=False)  # (B,1,target_W)
        return det_res.squeeze(1)  # (B, target_W)

    def resample_det_array_to_L0(self, det_array, L0):
        if det_array is None:
            return None
        if det_array.dim() == 2:
            det = det_array.unsqueeze(1)  # (B,1,curL)
        else:
            det = det_array
        det = det.to(self.device)
        det_L0 = F.interpolate(det, size=L0, mode='linear', align_corners=False)  # (B,1,L0)
        return det_L0.squeeze(1)  # (B, L0)

    def _make_proj_map(self, proj_mlp, scalar_feats, det_array_L0, H, W):
        """
        Utility to build broadcast map using a specific proj_mlp. proj_mlp can be None.
        det_array_L0: (B, L0) or None
        """
        if proj_mlp is None:
            B = scalar_feats.shape[0] if scalar_feats is not None else (det_array_L0.shape[0] if det_array_L0 is not None else 1)
            return torch.zeros(B, self.base_filters, H, W, device=self.device, dtype=torch.float32)

        # determine batch size
        if scalar_feats is not None:
            B = scalar_feats.shape[0]
        elif det_array_L0 is not None:
            B = det_array_L0.shape[0]
        else:
            B = 1

        # prepare scalar features
        if scalar_feats is None:
            sf = torch.zeros(B, 0, device=self.device)
        else:
            sf = scalar_feats.to(self.device).float()

        # prepare det part: if det_array_L0 is None, create zeros of length self.det_array_length
        if det_array_L0 is None:
            det_part = torch.zeros(B, self.det_array_length, device=self.device) if self.det_array_length > 0 else torch.zeros(B, 0, device=self.device)
        else:
            det_part = det_array_L0.to(self.device).float()

        feat_concat = torch.cat([sf, det_part], dim=1)  # (B, F_s + L0)
        proj = proj_mlp(feat_concat)  # (B, proj_dim)
        proj = proj.view(proj.shape[0], proj.shape[1], 1, 1)
        proj_map = proj.expand(-1, -1, H, W).contiguous()
        return proj_map

    def forward(self, x, scalar_features=None, det_array=None, target_width=None):
        """
        x: (B, C, H, W)
        scalar_features: (B, F_s) or None
        det_array: (B, L0) or None
        target_width: if provided, use for resampling det_array; else use x.shape[-1]
        """
        H, W = x.shape[-2:]
        if target_width is None:
            target_width = W

        # resample det_array global forms
        det_res_to_W = None
        det_res_L0 = None
        if det_array is not None:
            det_res_to_W = self.resample_det_array_to_targetW(det_array, target_width)  # (B, target_W)
            det_res_L0 = self.resample_det_array_to_L0(det_array, self.det_array_length)  # (B, L0)

        # encoder level 1: full resolution, use proj_enc1
        feat_map_full = self._make_proj_map(self.proj_enc1, scalar_features, det_res_L0, H, W)  # (B, f, H, W)
        x_in = torch.cat([x, feat_map_full], dim=1)
        e1 = self.enc1(x_in)
        p1 = self.pool1(e1)

        # level 2
        H2, W2 = p1.shape[2], p1.shape[3]
        det_lvl2 = None
        if det_res_to_W is not None:
            det_lvl2 = F.interpolate(det_res_to_W.unsqueeze(1), size=W2, mode='linear', align_corners=False).squeeze(1)
        feat_map_lvl2 = self._make_proj_map(self.proj_enc2, scalar_features, det_lvl2 if det_lvl2 is not None else det_res_L0, H2, W2)
        p1_in = torch.cat([p1, feat_map_lvl2], dim=1)
        e2 = self.enc2(p1_in)
        p2 = self.pool2(e2)

        # level 3
        H3, W3 = p2.shape[2], p2.shape[3]
        det_lvl3 = None
        if det_res_to_W is not None:
            det_lvl3 = F.interpolate(det_res_to_W.unsqueeze(1), size=W3, mode='linear', align_corners=False).squeeze(1)
        feat_map_lvl3 = self._make_proj_map(self.proj_enc3, scalar_features, det_lvl3 if det_lvl3 is not None else det_res_L0, H3, W3)
        p2_in = torch.cat([p2, feat_map_lvl3], dim=1)
        e3 = self.enc3(p2_in)
        p3 = self.pool3(e3)

        # bottleneck
        H4, W4 = p3.shape[2], p3.shape[3]
        det_lvl4 = None
        if det_res_to_W is not None:
            det_lvl4 = F.interpolate(det_res_to_W.unsqueeze(1), size=W4, mode='linear', align_corners=False).squeeze(1)
        feat_map_lvl4 = self._make_proj_map(self.proj_bottleneck, scalar_features, det_lvl4 if det_lvl4 is not None else det_res_L0, H4, W4)
        b_in = torch.cat([p3, feat_map_lvl4], dim=1)
        b = self.bottleneck(b_in)

        # decoder
        u3 = self.up3(b)
        if u3.size() != e3.size():
            u3 = F.interpolate(u3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        det_d3 = None
        if det_res_to_W is not None:
            det_d3 = F.interpolate(det_res_to_W.unsqueeze(1), size=e3.shape[3], mode='linear', align_corners=False).squeeze(1)
        feat_map_d3 = self._make_proj_map(self.proj_dec3, scalar_features, det_d3 if det_d3 is not None else det_res_L0, e3.shape[2], e3.shape[3])
        d3 = self.dec3(torch.cat([u3, e3, feat_map_d3], dim=1))

        u2 = self.up2(d3)
        if u2.size() != e2.size():
            u2 = F.interpolate(u2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        det_d2 = None
        if det_res_to_W is not None:
            det_d2 = F.interpolate(det_res_to_W.unsqueeze(1), size=e2.shape[3], mode='linear', align_corners=False).squeeze(1)
        feat_map_d2 = self._make_proj_map(self.proj_dec2, scalar_features, det_d2 if det_d2 is not None else det_res_L0, e2.shape[2], e2.shape[3])
        d2 = self.dec2(torch.cat([u2, e2, feat_map_d2], dim=1))

        u1 = self.up1(d2)
        if u1.size() != e1.size():
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        det_d1 = None
        if det_res_to_W is not None:
            # full-res det map already det_res_to_W (length target_width), but we still pass L0-form to proj for consistency
            det_d1 = det_res_to_W
        feat_map_d1 = self._make_proj_map(self.proj_dec1, scalar_features, det_d1 if det_d1 is not None else det_res_L0, e1.shape[2], e1.shape[3])
        d1 = self.dec1(torch.cat([u1, e1, feat_map_d1], dim=1))

        out = self.final_conv(d1)
        return out
