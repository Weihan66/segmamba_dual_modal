# models/dual_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segmamba_like import SegMambaLike
from .pet_3dcnn import PET3DCNN
from .swincross_cma import CrossModalAttention

class FusionHead(nn.Module):
    def __init__(self, in_ch, mid_ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_ch),
            nn.GELU(),
            nn.Conv3d(mid_ch, mid_ch//2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_ch//2),
            nn.GELU(),
            nn.Conv3d(mid_ch//2, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)

class DualModalNet(nn.Module):
    """
    Overall network:
      CT branch: SegMambaLike -> multiscale features
      PET branch: PET3DCNN -> multiscale features
      For each scale, apply CrossModalAttention to pair of features
      Fuse top-level features for segmentation head.
    """
    def __init__(self, ct_in_ch=1, pet_in_ch=1, base_ch=16, depth=4):
        super().__init__()
        self.ct_enc = SegMambaLike(in_ch=ct_in_ch, base_ch=base_ch, depth=depth)
        self.pet_enc = PET3DCNN(in_ch=pet_in_ch, base_ch=base_ch, depth=depth)
        # CMA modules for selected scales (use last 3 scales)
        # pick channel sizes accordingly (approximate)
        self.cmas = nn.ModuleList()
        ch_scale = [base_ch * (2**i) for i in range(depth)]
        for ch in ch_scale[-3:]:
            self.cmas.append(CrossModalAttention(in_ch=ch, token_dim=min(128, ch*4), heads=4))
        # final fusion head
        # unify shapes: use highest resolution feature (first enc_feats from ct/pet)
        self.fusion_conv = FusionHead(in_ch=ch_scale[0]*2, mid_ch=ch_scale[0]*4)

    def forward(self, ct, pet):
        # get features: out, enc_feats
        ct_top, ct_feats = self.ct_enc(ct)      # ct_top: [B,C,D,H,W], ct_feats list from shallow->deep
        pet_top, pet_feats = self.pet_enc(pet)  # similar
        # choose last 3 scales for cma (align indices)
        fused_feats = []
        num = len(self.cmas)
        for i in range(num):
            idx_ct = -(i+1)
            idx_pet = -(i+1)
            f_ct = ct_feats[idx_ct]
            f_pet = pet_feats[idx_pet]
            cma = self.cmas[-(i+1)]
            fct, fpet = cma(f_ct, f_pet)
            fused_feats.append((fct, fpet))
        # fuse by upsampling deep fused features to highest resolution and concatenate with ct_top & pet_top
        # take fused top-level features (first item of fused_feats corresponds to deepest). We'll upsample and sum.
        up_fused = None
        target_shape = ct_top.shape[2:]
        for fct, fpet in fused_feats:
            # sum the two modality features
            fsum = fct + fpet
            # resize to target
            fsum = F.interpolate(fsum, size=target_shape, mode='trilinear', align_corners=False)
            if up_fused is None:
                up_fused = fsum
            else:
                up_fused = up_fused + fsum
        # include ct_top and pet_top (project to same channels if needed)
        ct_proj = ct_top
        pet_proj = pet_top
        if ct_proj.shape[1] != pet_proj.shape[1]:
            # simple conv to match channels
            pet_proj = F.interpolate(pet_proj, size=ct_proj.shape[2:], mode='trilinear', align_corners=False)
        fused = torch.cat([ct_proj, pet_proj], dim=1)
        if up_fused is not None:
            fused = fused + up_fused
        out = self.fusion_conv(fused)
        out = torch.sigmoid(out)
        return out
