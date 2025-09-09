# models/segmamba_like.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    """
    A lightweight SSM/Mamba-inspired block for long-range modelling.
    Implementation idea: apply a depthwise separable conv1d across flattened spatial sequence
    to capture long-range dependencies along the depth axis, combined with pointwise convs.
    This is NOT the original SegMamba code — it's an inspired, runnable module.
    """
    def __init__(self, in_channels, hidden_channels, seq_axis=2, dropout=0.1):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channels)
        self.pointwise1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        # depthwise conv along depth (D) implemented as conv3d with kernel (k,1,1) groups=hidden
        self.depthwise = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(7,1,1), padding=(3,0,0), groups=hidden_channels)
        self.pointwise2 = nn.Conv3d(hidden_channels, in_channels, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout3d(dropout)

    def forward(self, x):
        # x: [B,C,D,H,W]
        y = self.norm(x)
        y = self.pointwise1(y)
        y = self.act(y)
        y = self.depthwise(y)
        y = self.act(y)
        y = self.pointwise2(y)
        y = self.drop(y)
        return x + y  # residual

class SegMambaLike(nn.Module):
    """
    A compact encoder-decoder that uses MambaBlock in encoder for CT branch.
    """
    def __init__(self, in_ch=1, base_ch=16, depth=4):
        super().__init__()
        enc_chs = [base_ch * (2**i) for i in range(depth)]
        self.init_conv = nn.Conv3d(in_ch, enc_chs[0], kernel_size=3, padding=1)
        self.downs = nn.ModuleList()
        for i in range(depth):
            blocks = nn.Sequential(
                MambaBlock(enc_chs[i], enc_chs[i]*2),
                MambaBlock(enc_chs[i], enc_chs[i]*2),
            )
            self.downs.append(blocks)
            if i < depth -1:
                self.downs.append(nn.Conv3d(enc_chs[i], enc_chs[i+1], kernel_size=2, stride=2))  # downsample

        # bottleneck
        self.bottleneck = nn.Sequential(
            MambaBlock(enc_chs[-1], enc_chs[-1]*2),
            MambaBlock(enc_chs[-1], enc_chs[-1]*2)
        )

        # simple decoder
        self.ups = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            up = nn.ConvTranspose3d(enc_chs[i+1], enc_chs[i], kernel_size=2, stride=2)
            self.ups.append(up)
            self.ups.append(nn.Sequential(
                MambaBlock(enc_chs[i]*2, enc_chs[i]*2),
                MambaBlock(enc_chs[i]*2, enc_chs[i]*2),
            ))

        self.out_conv = nn.Conv3d(enc_chs[0], enc_chs[0], kernel_size=1)

    def forward(self, x):
        # encoder
        x = self.init_conv(x)
        enc_feats = []
        cur = x
        i = 0
        # iterate using pairs in self.downs: block then maybe downsample
        j = 0
        while j < len(self.downs):
            blocks = self.downs[j]
            cur = blocks(cur)
            enc_feats.append(cur)
            j += 1
            if j < len(self.downs):
                down = self.downs[j]
                cur = down(cur)
                j += 1
        # bottleneck
        cur = self.bottleneck(cur)
        # decoder
        for k in range(0, len(self.ups), 2):
            up = self.ups[k]
            blocks = self.ups[k+1]
            cur = up(cur)
            # skip connection with corresponding enc_feats (reverse)
            enc = enc_feats[-(k//2 + 1)]
            # if shapes mismatch, center crop
            if cur.shape != enc.shape:
                # simple resize
                cur = F.interpolate(cur, size=enc.shape[2:], mode='trilinear', align_corners=False)
            cur = torch.cat([cur, enc], dim=1)
            cur = blocks(cur)
        out = self.out_conv(cur)
        return out, enc_feats  # return multiscale features for fusion
