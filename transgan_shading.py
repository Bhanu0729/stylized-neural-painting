# transgan_shading.py
import torch, math
import torch.nn as nn
import torch.nn.functional as F

# ---------- MLP --------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

# ---------- Transformer block -------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, int(dim*mlp_ratio))
    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x

# ---------- PixelShuffle upsampling --------------------------------
def upsample(x, h, w):
    B, N, C = x.shape
    x = x.transpose(1,2).view(B, C, h, w)    # (B,C,H,W)
    x = F.pixel_shuffle(x, 2)                # (B,C/4,2H,2W)
    B, C2, H2, W2 = x.shape
    return x.view(B, C2, H2*W2).transpose(1,2), H2, W2

# ---------- TransGAN ShadingNet ------------------------------------
class TransShadingNet(nn.Module):
    """
    Input : (B, 12, 1, 1) — stroke parameters
    Output: (B, 3, 128, 128) RGB + (B, 1, 128, 128) alpha
    """
    def __init__(self, rdrr, dim_base=128, depth=(4, 4, 2)):
        super().__init__()
        self.out_size = 128

        # encode 12D stroke vector to token
        self.token_fc = nn.Linear(12, dim_base*8)

        # Transformer stages
        self.stage1 = nn.ModuleList([TransformerBlock(dim_base*8) for _ in range(depth[0])])
        self.stage2 = nn.ModuleList([TransformerBlock(dim_base*2) for _ in range(depth[1])])
        self.stage3 = nn.ModuleList([TransformerBlock(dim_base//2) for _ in range(depth[2])])

        # Final RGB and Alpha heads
        self.to_rgb   = nn.Linear(dim_base//2, 3)
        self.to_alpha = nn.Linear(dim_base//2, 1)

        self.start_H = self.start_W = 8  # 8×8 → 128×128 via 2×2 upsampling and bicubic

    def forward(self, z):   # z: (B, 12, 1, 1)
        B = z.size(0)
        z = z.view(B, 12)
        x = self.token_fc(z).view(B, -1, 1).transpose(1, 2)  # (B,1,C)

        H = W = self.start_H

        # Stage 1 (8×8)
        for blk in self.stage1: x = blk(x)
        x = x.repeat(1, H*W, 1)

        # Stage 2 (16×16)
        x, H, W = upsample(x, H, W)
        for blk in self.stage2: x = blk(x)

        # Stage 3 (32×32)
        x, H, W = upsample(x, H, W)
        for blk in self.stage3: x = blk(x)

        # Final upsample to 128×128
        x = x.transpose(1,2).view(B, -1, H, W)  # (B,C,32,32)
        x = F.interpolate(x, size=self.out_size, mode='bicubic', align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B,128,128,C)

        # Predict RGB and Alpha
        rgb   = torch.sigmoid(self.to_rgb(x))     # (B,128,128,3)
        alpha = torch.sigmoid(self.to_alpha(x))   # (B,128,128,1)

        # Permute to match output shape
        rgb   = rgb.permute(0, 3, 1, 2)            # (B,3,H,W)
        alpha = alpha.permute(0, 3, 1, 2)          # (B,1,H,W)

        return rgb, alpha
