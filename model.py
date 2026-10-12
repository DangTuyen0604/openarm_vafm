"""VA flow-matching policy: DINOv2 + SigLIP (đóng băng) + depth CNN + qpos -> chunk 40x16.

    RGB  -> DINOv2+SigLIP (đóng băng, bf16) -> projector MLP -> 256 token
    Depth-> CNN nhỏ (2 kênh: độ sâu + mask) -> 64 token
    qpos -> MLP -> 1 token
    321 token + pos emb -> context encoder (4 lớp) -> ctx
    x_t (nhiễu), t -> action expert (8 khối: self-attn, cross-attn(ctx), MLP, adaLN-Zero) -> v
    Suy luận: tích phân Euler 10 bước t: 1 -> 0.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# =====================================================================
# Backbone thị giác đóng băng (giống OpenVLA: DINOv2 + SigLIP, lớp áp chót)
# =====================================================================
class FrozenDinoSigLIP(nn.Module):
    def __init__(self, cfg: Config, dtype=torch.bfloat16):
        super().__init__()
        import timm
        pre = cfg.backbone_pretrained and not cfg.openvla_vision_path
        self.dino = timm.create_model(cfg.dino_name, pretrained=pre, num_classes=0, img_size=cfg.img_size)
        self.siglip = timm.create_model(cfg.siglip_name, pretrained=pre, num_classes=0, img_size=cfg.img_size)
        if cfg.openvla_vision_path:
            load_openvla_vision(self, cfg.openvla_vision_path)
        self.register_buffer("dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.dtype = dtype
        self.dino.to(dtype)
        self.siglip.to(dtype)
        for p in self.parameters():
            p.requires_grad_(False)
        self.feat_dim = self.dino.embed_dim + self.siglip.embed_dim          # 1024 + 1152 = 2176
        g1 = tuple(self.dino.patch_embed.grid_size)
        g2 = tuple(self.siglip.patch_embed.grid_size)
        assert g1 == g2, f"Lưới patch DINOv2 {g1} != SigLIP {g2}"
        self.n_tokens = g1[0] * g1[1]                                          # 256 @ 224px
        self.eval()

    def train(self, mode=True):  # luôn ở eval (đóng băng, không dropout)
        return super().train(False)

    @torch.no_grad()
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        x = x01.to(self.dtype)
        xd = (x - self.dino_mean.to(self.dtype)) / self.dino_std.to(self.dtype)
        xs = (x - 0.5) / 0.5
        a = self.dino.get_intermediate_layers(xd, n=[len(self.dino.blocks) - 2])[0]
        b = self.siglip.get_intermediate_layers(xs, n=[len(self.siglip.blocks) - 2])[0]
        return torch.cat([a, b], -1).float()


def load_openvla_vision(bb: FrozenDinoSigLIP, path: str):
    """Nạp trọng số thị giác đã tách từ openvla-7b (tools.py extract-openvla)."""
    from safetensors.torch import load_file
    from timm.models.vision_transformer import checkpoint_filter_fn
    sd = load_file(path)
    for prefix, m in (("featurizer.", bb.dino), ("fused_featurizer.", bb.siglip)):
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        try:
            sub = checkpoint_filter_fn(sub, m)  # nội suy pos_embed nếu img_size khác 224
        except Exception as e:  # noqa: BLE001
            print(f"[model] checkpoint_filter_fn bỏ qua: {e}")
        missing, unexpected = m.load_state_dict(sub, strict=False)
        missing = [k for k in missing if not k.startswith(("head", "fc_norm", "attn_pool"))]
        print(f"[model] OpenVLA {prefix[:-1]}: {len(sub)} tensor, thiếu {len(missing)}, thừa {len(unexpected)}")
        if missing:
            print("   thiếu:", missing[:8])


# =====================================================================
# Các khối mới, được huấn luyện
# =====================================================================
class DepthEncoder(nn.Module):
    """RỦI RO 5: CNN nhỏ (~0.5M tham số) cho depth đã clip + mask -> lưới 8x8 = 64 token."""

    def __init__(self, d_model: int, in_ch: int = 2):
        super().__init__()
        chs = [in_ch, 32, 64, 128, 256]
        layers = []
        for i in range(4):
            k = 5 if i == 0 else 3
            layers += [nn.Conv2d(chs[i], chs[i + 1], k, 2, k // 2),
                       nn.GroupNorm(8, chs[i + 1]), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.out = nn.Conv2d(256, d_model, 1)

    def forward(self, x):                      # (B,2,128,128)
        h = self.out(self.net(x))              # (B,d,8,8)
        return h.flatten(2).transpose(1, 2)    # (B,64,d)


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Mlp(nn.Sequential):
    def __init__(self, d, ratio, drop):
        h = int(d * ratio)
        super().__init__(nn.Linear(d, h), nn.GELU(approximate="tanh"), nn.Dropout(drop), nn.Linear(h, d))


class ExpertBlock(nn.Module):
    """self-attn (giữa 40 bước) -> cross-attn (vào 321 token ngữ cảnh) -> MLP, đều điều biến adaLN-Zero theo t."""

    def __init__(self, d, heads, mlp_ratio, drop):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.sa = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ca = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(d, mlp_ratio, drop)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(self.ada[1].weight)       # adaLN-Zero: khối bắt đầu là hàm đồng nhất
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, ctx, c):
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = self.ada(c).unsqueeze(1).chunk(9, -1)
        h = modulate(self.n1(x), s1, c1)
        x = x + g1 * self.sa(h, h, h, need_weights=False)[0]
        h = modulate(self.n2(x), s2, c2)
        x = x + g2 * self.ca(h, ctx, ctx, need_weights=False)[0]
        h = modulate(self.n3(x), s3, c3)
        return x + g3 * self.mlp(h)


# =====================================================================
# Policy
# =====================================================================
class VAFlowPolicy(nn.Module):
    def __init__(self, cfg: Config, stats: dict | None = None, backbone_dtype=None):
        super().__init__()
        self.cfg = cfg
        d, K, D = cfg.d_model, cfg.chunk, cfg.action_dim
        if backbone_dtype is None:
            backbone_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.backbone = FrozenDinoSigLIP(cfg, backbone_dtype)

        self.projector = nn.Sequential(nn.LayerNorm(self.backbone.feat_dim),
                                       nn.Linear(self.backbone.feat_dim, d), nn.GELU(), nn.Linear(d, d))
        self.depth_enc = DepthEncoder(d)
        self.state_mlp = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, d))
        self.n_rgb = self.backbone.n_tokens
        self.n_depth = (cfg.depth_size // 16) ** 2
        self.n_ctx = self.n_rgb + self.n_depth + 1                       # 256 + 64 + 1 = 321
        self.ctx_pos = nn.Parameter(torch.randn(1, self.n_ctx, d) * 0.02)
        self.ctx_drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, int(d * cfg.mlp_ratio), cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.enc_layers, enable_nested_tensor=False)
        self.ctx_norm = nn.LayerNorm(d)

        self.act_in = nn.Linear(D, d)
        self.act_pos = nn.Parameter(torch.randn(1, K, d) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([ExpertBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
                                     for _ in range(cfg.expert_layers)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.act_out = nn.Linear(d, D)
        for m in (self.final_ada[1], self.act_out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # RỦI RO 3: chuẩn hoá theo từng (bước, khớp) — lưu trong checkpoint
        z = lambda *s: torch.zeros(*s)
        o = lambda *s: torch.ones(*s)
        self.register_buffer("action_mean", z(K, D))
        self.register_buffer("action_std", o(K, D))
        self.register_buffer("state_mean", z(D))
        self.register_buffer("state_std", o(D))
        if stats is not None:
            for k in ("action_mean", "action_std", "state_mean", "state_std"):
                getattr(self, k).copy_(torch.as_tensor(np.asarray(stats[k]), dtype=torch.float32))

    # ---------- chuẩn hoá ----------
    def norm_state(self, s):
        return (s - self.state_mean) / self.state_std

    def norm_action(self, a):
        return (a - self.action_mean) / self.action_std

    def unnorm_action(self, a):
        return a * self.action_std + self.action_mean

    # ---------- mạng ----------
    def encode(self, rgb01, depth2, state_n):
        r = self.projector(self.backbone(rgb01))
        tok = torch.cat([r, self.depth_enc(depth2), self.state_mlp(state_n).unsqueeze(1)], 1)
        assert tok.shape[1] == self.n_ctx, (tok.shape, self.n_ctx)
        return self.ctx_norm(self.encoder(self.ctx_drop(tok + self.ctx_pos)))

    def velocity(self, x_t, t, ctx):
        h = self.act_in(x_t) + self.act_pos
        c = self.time_mlp(timestep_embedding(t))
        for blk in self.blocks:
            h = blk(h, ctx, c)
        shift, scale = self.final_ada(c).unsqueeze(1).chunk(2, -1)
        return self.act_out(modulate(self.final_norm(h), shift, scale))

    @torch.no_grad()
    def sample(self, ctx, steps: int | None = None, generator: torch.Generator | None = None):
        """Euler từ t=1 (nhiễu) về t=0 (hành động chuẩn hoá)."""
        steps = steps or self.cfg.flow_steps
        B = ctx.shape[0]
        x = torch.randn(B, self.cfg.chunk, self.cfg.action_dim, device=ctx.device,
                        generator=generator).to(ctx.dtype)
        dt = -1.0 / steps
        for i in range(steps):
            t = torch.full((B,), 1.0 + i * dt, device=ctx.device)
            x = x + dt * self.velocity(x, t, ctx).to(x.dtype)
        return x.float()

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")}

    def load_trainable(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("backbone.")]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint không khớp. Thiếu: {missing[:5]} Thừa: {unexpected[:5]}")


def sample_flow_t(n, device, generator: torch.Generator | None = None):
    """Như π0: t ~ Beta(1.5, 1) nghiêng về mức nhiễu cao (t=1 là nhiễu).
    CDF của Beta(a, 1) là u^a -> lấy mẫu bằng U^(1/a), cho phép truyền generator (eval cố định)."""
    u = torch.rand(n, device=device, generator=generator) ** (1 / 1.5)
    return 0.001 + 0.999 * u
