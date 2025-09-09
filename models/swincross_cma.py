# models/swincross_cma.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    A SwinCross-like Cross-Modal Attention (CMA) module.
    Implementation: project CT & PET features to tokens, perform multihead cross-attention (CT queries, PET keys/vals and vice versa),
    then fuse outputs and project back to 3D shape.
    This is a simplified, runnable variant.
    """
    def __init__(self, in_ch, token_dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.token_dim = token_dim
        self.heads = heads
        self.to_q = nn.Conv3d(in_ch, token_dim, kernel_size=1)
        self.to_k = nn.Conv3d(in_ch, token_dim, kernel_size=1)
        self.to_v = nn.Conv3d(in_ch, token_dim, kernel_size=1)
        self.proj_out = nn.Conv3d(token_dim, in_ch, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.scale = (token_dim // heads) ** -0.5

    def flatten_hw(self, x):
        # x: [B,C,D,H,W] -> [B, L, C] where L = D*H*W
        B, C, D, H, W = x.shape
        x = x.view(B, C, -1).permute(0,2,1).contiguous()
        return x, (D,H,W)

    def unflatten(self, x, shape3d):
        # x: [B,L,C] -> [B,C,D,H,W]
        D, H, W = shape3d
        x = x.permute(0,2,1).contiguous()
        x = x.view(x.shape[0], x.shape[1], D, H, W)
        return x

    def mha_cross(self, q, k, v):
        # q,k,v: [B,L,dim]
        B, Lq, Dq = q.shape
        _, Lk, Dk = k.shape
        h = self.heads
        qh = q.view(B, Lq, h, Dq//h).transpose(1,2)  # [B,h,Lq,dh]
        kh = k.view(B, Lk, h, Dk//h).transpose(1,2)
        vh = v.view(B, Lk, h, Dk//h).transpose(1,2)
        attn = torch.matmul(qh, kh.transpose(-2,-1)) * self.scale  # [B,h,Lq,Lk]
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, vh)  # [B,h,Lq,dh]
        out = out.transpose(1,2).contiguous().view(B, Lq, Dq)
        return out

    def forward(self, feat_ct, feat_pet):
        # feat_*: [B,C,D,H,W] - assume same spatial dims or resize
        if feat_ct.shape != feat_pet.shape:
            feat_pet = F.interpolate(feat_pet, size=feat_ct.shape[2:], mode='trilinear', align_corners=False)
        # project
        q_ct = self.to_q(feat_ct)
        k_ct = self.to_k(feat_ct)
        v_ct = self.to_v(feat_ct)
        q_pet = self.to_q(feat_pet)
        k_pet = self.to_k(feat_pet)
        v_pet = self.to_v(feat_pet)
        # flatten
        q_ct_f, shape = self.flatten_hw(q_ct)
        k_ct_f, _ = self.flatten_hw(k_ct)
        v_ct_f, _ = self.flatten_hw(v_ct)
        q_pet_f, _ = self.flatten_hw(q_pet)
        k_pet_f, _ = self.flatten_hw(k_pet)
        v_pet_f, _ = self.flatten_hw(v_pet)
        # cross attention: CT queries PET keys/vals and PET queries CT keys/vals
        ct_attn = self.mha_cross(q_ct_f, k_pet_f, v_pet_f)
        pet_attn = self.mha_cross(q_pet_f, k_ct_f, v_ct_f)
        # fuse: average and project back
        fused_ct = (ct_attn + q_ct_f) * 0.5
        fused_ct = fused_ct.view(fused_ct.shape[0], fused_ct.shape[2], -1).permute(0,2,1)
        fused_ct = self.unflatten(fused_ct, shape)
        fused_ct = self.proj_out(fused_ct)
        fused_ct = self.dropout(fused_ct)
        # similarly for pet (we will return fused_ct and fused_pet)
        fused_pet = (pet_attn + q_pet_f) * 0.5
        fused_pet = fused_pet.view(fused_pet.shape[0], fused_pet.shape[2], -1).permute(0,2,1)
        fused_pet = self.unflatten(fused_pet, shape)
        fused_pet = self.proj_out(fused_pet)
        fused_pet = self.dropout(fused_pet)
        # residual add
        out_ct = feat_ct + fused_ct
        out_pet = feat_pet + fused_pet
        return out_ct, out_pet
