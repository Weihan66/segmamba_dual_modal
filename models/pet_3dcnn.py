# models/pet_3dcnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )
    def forward(self,x): return self.net(x)

class PET3DCNN(nn.Module):
    """
    Simple UNet-like 3D CNN for PET branch.
    """
    def __init__(self, in_ch=1, base_ch=16, depth=4):
        super().__init__()
        chs = [base_ch * (2**i) for i in range(depth)]
        self.inc = DoubleConv(in_ch, chs[0])
        self.downs = nn.ModuleList()
        for i in range(depth-1):
            self.downs.append(nn.Sequential(
                nn.MaxPool3d(2),
                DoubleConv(chs[i], chs[i+1])
            ))
        self.bottleneck = DoubleConv(chs[-1], chs[-1]*2)
        self.ups = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.ups.append(nn.ConvTranspose3d(chs[i+1]*2 if i < depth-2 else chs[i+1]*2, chs[i], kernel_size=2, stride=2))
            self.ups.append(DoubleConv(chs[i]*2, chs[i]))
        self.out_conv = nn.Conv3d(chs[0], chs[0], kernel_size=1)

    def forward(self, x):
        feats = []
        cur = self.inc(x)
        feats.append(cur)
        for d in self.downs:
            cur = d(cur)
            feats.append(cur)
        cur = self.bottleneck(cur)
        # up
        for k in range(0, len(self.ups), 2):
            up = self.ups[k]
            block = self.ups[k+1]
            cur = up(cur)
            enc = feats[-(k//2 + 2)]
            if cur.shape != enc.shape:
                cur = F.interpolate(cur, size=enc.shape[2:], mode='trilinear', align_corners=False)
            cur = torch.cat([cur, enc], dim=1)
            cur = block(cur)
        out = self.out_conv(cur)
        return out, feats
