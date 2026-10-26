import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DiTConfig:
    embedding_size: int = 192
    num_heads: int = 6
    head_size: int = 36     # not 192 / 6 = 32: 3D RoPE needs a multiple of 6, so heads span 216 dims
    ffn_hidden: int = 768
    num_blocks: int = 6
    patch_size: int = 4
    num_frames: int = 8
    img_size: tuple = (32, 32)
    num_classes: int = 10

    def __post_init__(self):
        assert self.head_size % 6 == 0, "head_size must be divisible by 6 (2 dims x 3 axes)"


def create_freqs_3d(T, Hp, Wp, head_dim, base=10000):
    d = head_dim // 3 #  3 axis (t,y,x)
    theta = base ** -(torch.arange(0, d, 2) / d)   # (d//2,)

    t, y, x = torch.meshgrid(torch.arange(T), torch.arange(Hp), torch.arange(Wp),
                             indexing="ij")

    t, y, x = t.flatten(), y.flatten(), x.flatten()      # each (N,) with N = T*Hp*Wp

    angles = torch.cat([t[:, None]*theta,
                        y[:, None]*theta,
                        x[:, None]*theta], dim=-1)       # (N, head_dim//2)

    return angles.cos(), angles.sin()


def apply_rotary_emb(x, cos, sin):
    x_half = x.reshape((*x.shape[:-1], -1, 2)) # B x H x N x D//2 x 2

    even = x_half[..., 0] # B x H x N x D//2
    odd = x_half[..., 1] # B x H x N x D//2

    upper = even * cos - odd * sin # B x H x N x D//2
    lower = even * sin + odd * cos # B x H x N x D//2

    return torch.stack((upper, lower), dim=-1).flatten(-2) # B x H x N x D - alternating


class TimestepEmbedder(nn.Module):
    def __init__(self, embedding_size, hidden_dim=256, base=10000, scale_t=1000):
        super().__init__()
        self.scale_t = scale_t

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size)
        )

        theta = base ** - (torch.arange(0, hidden_dim, 2) / hidden_dim) # (d//2, )
        self.register_buffer("theta", theta, persistent=False)


    def time_encode(self, t):
        freqs = t[:, None] * self.theta[None, :] # (B, d//2)
        return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    def forward(self, t):
        return self.mlp(self.time_encode(t * self.scale_t))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, embedding_size, drop_prob=0.1):
        super().__init__()

        self.drop_prob = drop_prob
        self.use_cfg = drop_prob > 0
        self.null_class = num_classes

        self.embedder = nn.Embedding(num_classes + self.use_cfg, embedding_size)

    def drop_labels(self, labels):
        drop_ids = torch.rand(labels.shape, device=labels.device) < self.drop_prob
        return torch.where(drop_ids, torch.full_like(labels, self.null_class), labels)

    def forward(self, labels):
        if self.use_cfg and self.training:
            labels = self.drop_labels(labels)
        return self.embedder(labels)


class Attention(nn.Module):
    def __init__(self, num_heads, dim, hidden_dim):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.qkv = nn.Linear(dim, num_heads * hidden_dim * 3)
        self.concat = nn.Linear(num_heads * hidden_dim, dim)

    def forward(self, x, cos, sin):
        B, N, _ = x.size() # (B, N, D)

        # query, key, value matrix - (B, H, N, D_v)
        q, k, v = self.qkv(x).reshape(B, N, self.num_heads, 3, self.hidden_dim).transpose(1, 2).unbind(3)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # query * key / sqrt(D_v) for scaling
        qk = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.hidden_dim) # (B, H, N, N) - qk attention
        qk = torch.softmax(qk, -1) # along last dim - attention to pay to each key for the query
        
        # values
        vals = torch.matmul(qk, v).transpose(1, 2).flatten(2, 3) # (B, H, N, D_v) -> (B, N, H*D_v)
        return self.concat(vals)


class DiTBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, head_size, ffn_hidden):
        super().__init__()

        self.ln_mha = nn.LayerNorm(embedding_size, elementwise_affine=False, eps=1e-6)
        self.ln_mlp = nn.LayerNorm(embedding_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(num_heads, embedding_size, head_size)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, ffn_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_hidden, embedding_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_size, 6 * embedding_size)
        )


    def forward(self, x, c, cos, sin):
        shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).unsqueeze(1).chunk(6, -1) # B x 1 x D

        x = x + torch.mul(gate_mha, self.attn(torch.mul(1 + scale_mha, self.ln_mha(x)) + shift_mha, cos, sin))
        x = x + torch.mul(gate_mlp, self.ffn(torch.mul(1 + scale_mlp, self.ln_mlp(x)) + shift_mlp))

        return x


class DitLastLayer(nn.Module):
    def __init__(self, embedding_size, patch_size):
        super().__init__()

        self.ln = nn.LayerNorm(embedding_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size * 2)
        )
        self.fc = nn.Linear(embedding_size, patch_size * patch_size)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).unsqueeze(1).chunk(2, -1)
        x = torch.mul(1 + scale, self.ln(x)) + shift
        return self.fc(x)


class DiT(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()

        self.num_blocks = config.num_blocks
        self.patch_size = config.patch_size
        self.H, self.W = config.img_size
        self.T = config.num_frames

        cos, sin = create_freqs_3d(config.num_frames, self.H//config.patch_size, self.W//config.patch_size, config.head_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        self.embedder = nn.Conv3d(1, config.embedding_size, (1, config.patch_size, config.patch_size), stride=(1, config.patch_size, config.patch_size))

        self.t_embedder = TimestepEmbedder(config.embedding_size)
        self.label_embedder = LabelEmbedder(config.num_classes, config.embedding_size)

        self.blocks = nn.ModuleList([
            DiTBlock(config.embedding_size, config.num_heads, config.head_size, config.ffn_hidden) for _ in range(config.num_blocks)
        ])
        self.last_layer = DitLastLayer(config.embedding_size, config.patch_size)

        self.initialize_weights()

    def patchify_raw(self, x, p):
        B, C, T, H, W = x.size()
        x = x.reshape(B, C, T, H//p, p, W//p, p)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)      # B, T, H', W', p, p, C
        return x.reshape(B, T*(H//p)*(W//p), p*p*C)


    def patchify(self, x):
        # B x C x T x H x W -> B x N x embd
        # N = T * (H/p) * (W/p)

        return self.embedder(x).flatten(2, 4).transpose(1, 2)
    
    def unpatchify(self, x):
        # B x N x patch*patch
        # N = T * (H/p) * (W/p)
        B = x.size()[0]
        x = x.reshape(B, 1, self.T, self.H//self.patch_size, self.W//self.patch_size, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 2, 3, 5, 4, 6) # (B, 1, T, grid, p, grid, p)

        return x.reshape(B, 1, self.T, self.H, self.W)

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.last_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.last_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.last_layer.fc.weight)
        nn.init.zeros_(self.last_layer.fc.bias)
        nn.init.normal_(self.label_embedder.embedder.weight, std=0.02)

    def forward(self, x, y, t):
        x = self.patchify(x) # B x N x D
        c = self.t_embedder(t) + self.label_embedder(y) # B x D

        for i in range(self.num_blocks):
            x = self.blocks[i](x, c, self.cos, self.sin)

        x = self.last_layer(x, c) # B x N x p**2

        return self.unpatchify(x)


def main():
    config = DiTConfig()
    T, (H, W), p = config.num_frames, config.img_size, config.patch_size
    B = 32
    dit = DiT(config)

    imgs = torch.randn(B, 1, T, H, W)
    imgs_out = dit.unpatchify(dit.patchify_raw(imgs, p))

    assert imgs_out.size() == imgs.size()
    assert torch.allclose(imgs, imgs_out)

    x = torch.randn(B, 1, T, H, W)
    y = torch.randint(0,10, (B, ))
    t = torch.rand(B)
    out = dit(x, y, t)

    assert out.size() == x.size()

if __name__ == "__main__":
    main()
