import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# Sinusoidal embedding (shared for diffusion step)
# ──────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DiffusionEmbedding(nn.Module):
    """Sinusoidal → Linear → SiLU → Linear (TimeGrad style)."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)   # (B, dim)


# ──────────────────────────────────────────────
# Residual block (CSDI 2D attention + TimeGrad gated dilated conv)
# ──────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """
    Input:
        h:         (B, C, K, L)
        side_info: (B, side_dim, K, L)
    Output:
        residual:  (B, C, K, L)
        skip:      (B, C, K, L)
    """

    def __init__(
        self,
        channels: int = 64,
        side_dim: int = 144,
        nhead_time: int = 8,
        nhead_feat: int = 4,
        dilation: int = 1,
    ):
        super().__init__()
        self.time_attn = nn.TransformerEncoderLayer(
            d_model=channels, nhead=nhead_time, dim_feedforward=channels, batch_first=False
        )
        self.feat_attn = nn.TransformerEncoderLayer(
            d_model=channels, nhead=nhead_feat, dim_feedforward=channels, batch_first=False
        )
        # Dilated conv on time axis, expands to 2*C for gate/filter split
        self.dil_conv = nn.Conv1d(
            channels, 2 * channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.side_proj = nn.Conv2d(side_dim, 2 * channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.skip_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def _time_attn(self, h: torch.Tensor) -> torch.Tensor:
        # h:(B,C,K,L)
        B, C, K, L = h.shape
        x = h.permute(0, 2, 1, 3).reshape(B * K, C, L)            # (B*K, C, L)
        x = x.permute(2, 0, 1)                                     # (L, B*K, C)
        x = self.time_attn(x)
        x = x.permute(1, 2, 0).reshape(B, K, C, L).permute(0, 2, 1, 3)
        return x

    def _feat_attn(self, h: torch.Tensor) -> torch.Tensor:
        # h:(B,C,K,L)
        B, C, K, L = h.shape
        x = h.permute(0, 3, 1, 2).reshape(B * L, C, K)            # (B*L, C, K)
        x = x.permute(2, 0, 1)                                     # (K, B*L, C)
        x = self.feat_attn(x)
        x = x.permute(1, 2, 0).reshape(B, L, C, K).permute(0, 2, 3, 1)
        return x

    def forward(self, h: torch.Tensor, side_info: torch.Tensor):
        B, C, K, L = h.shape
        x = self._time_attn(h)
        x = self._feat_attn(x)

        # Dilated conv on time axis (flatten K into batch)
        r = x.permute(0, 2, 1, 3).reshape(B * K, C, L)             # (B*K, C, L)
        r = self.dil_conv(r)                                        # (B*K, 2C, L)
        r = r.reshape(B, K, 2 * C, L).permute(0, 2, 1, 3)           # (B, 2C, K, L)

        r = r + self.side_proj(side_info)                           # (B, 2C, K, L)

        gate, filt = r.chunk(2, dim=1)
        out = torch.sigmoid(gate) * torch.tanh(filt)                # (B, C, K, L)

        skip = self.skip_proj(out)
        residual = (h + self.out_proj(out)) / math.sqrt(2.0)
        return residual, skip


# ──────────────────────────────────────────────
# ConditionalDenoiser
# ──────────────────────────────────────────────
class ConditionalDenoiser(nn.Module):
    """
    Inputs:
        y_t:    (B, D, pred_len)   noisy target
        t:      (B,)                diffusion step
        x_cond: (B, D, cond_len)   clean condition (history)
    Output:
        eps:    (B, D, pred_len)   predicted noise
    """

    def __init__(
        self,
        d_stocks: int = 10,
        cond_len: int = 60,
        pred_len: int = 20,
        channels: int = 64,
        diff_embed_dim: int = 128,
        feat_embed_dim: int = 16,
        n_blocks: int = 4,
        dilation_cycle: int = 2,
    ):
        super().__init__()
        self.d_stocks = d_stocks
        self.cond_len = cond_len
        self.pred_len = pred_len
        self.total_len = cond_len + pred_len
        self.channels = channels

        # Input projection: (value + observation_mask) 2 channels → C
        self.input_proj = nn.Conv2d(2, channels, kernel_size=1)

        # Diffusion step embedding
        self.diff_embed = DiffusionEmbedding(diff_embed_dim)

        # Feature (stock id) embedding
        self.feat_embed = nn.Embedding(d_stocks, feat_embed_dim)

        side_dim = diff_embed_dim + feat_embed_dim

        # Residual blocks with TimeGrad dilation cycle
        self.blocks = nn.ModuleList([
            ResidualBlock(
                channels=channels,
                side_dim=side_dim,
                dilation=2 ** (i % dilation_cycle),
            )
            for i in range(n_blocks)
        ])

        # Output projection: (C → C) → ReLU → (C → 1)   CSDI style
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def _build_side_info(self, t: torch.Tensor, B: int):
        """Return side_info of shape (B, diff_dim + feat_dim, K, L)."""
        K, L = self.d_stocks, self.total_len
        # Diffusion embedding (B, diff_dim) → broadcast to (B, diff_dim, K, L)
        d_emb = self.diff_embed(t)                                 # (B, diff_dim)
        d_emb = d_emb[:, :, None, None].expand(-1, -1, K, L)

        # Feature embedding (K, feat_dim) → broadcast to (B, feat_dim, K, L)
        feat_ids = torch.arange(K, device=t.device)
        f_emb = self.feat_embed(feat_ids)                           # (K, feat_dim)
        f_emb = f_emb.permute(1, 0)[None, :, :, None].expand(B, -1, -1, L)

        return torch.cat([d_emb, f_emb], dim=1)                    # (B, diff+feat, K, L)

    def forward(self, y_t, t, x_cond):
        B, D, _ = y_t.shape
        assert D == self.d_stocks
        assert y_t.shape[2] == self.pred_len
        assert x_cond.shape[2] == self.cond_len

        # Concatenate along time: (B, D, 80)
        full = torch.cat([x_cond, y_t], dim=2)                     # (B, D, L)

        # Observation mask: 1 for cond (clean), 0 for target (noisy)
        mask = torch.zeros(B, 1, self.total_len, device=y_t.device)
        mask[:, :, : self.cond_len] = 1.0
        mask = mask.expand(-1, D, -1)                              # (B, D, L)

        # Reshape to (B, 2, K=D, L): 2 channels = value + mask
        value_2d = full.unsqueeze(1)                                # (B, 1, D, L)
        mask_2d = mask.unsqueeze(1)                                 # (B, 1, D, L)
        inp = torch.cat([value_2d, mask_2d], dim=1)                 # (B, 2, D, L)

        # Input projection
        h = self.input_proj(inp)                                    # (B, C, K, L)

        # Side info
        side = self._build_side_info(t, B)                          # (B, side_dim, K, L)

        # Residual blocks with skip aggregation
        skips = []
        for block in self.blocks:
            h, skip = block(h, side)
            skips.append(skip)

        z = torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(len(self.blocks))

        # Output projection → (B, 1, K, L)
        z = self.output_proj(z)                                     # (B, 1, K, L)

        # Slice prediction segment and drop channel dim
        eps = z[:, 0, :, self.cond_len :]                           # (B, D, pred_len)
        return eps


# ──────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    B, D, COND, PRED = 4, 10, 60, 20

    model = ConditionalDenoiser(d_stocks=D, cond_len=COND, pred_len=PRED)

    y_t = torch.randn(B, D, PRED)
    t = torch.randint(0, 50, (B,))
    x_cond = torch.randn(B, D, COND)

    eps = model(y_t, t, x_cond)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:     {n_params:,}")
    print(f"y_t:     {y_t.shape}")
    print(f"x_cond:  {x_cond.shape}")
    print(f"eps:     {eps.shape}")
    assert eps.shape == y_t.shape, f"Shape mismatch: {eps.shape} vs {y_t.shape}"

    # Check gradient flow
    loss = eps.square().mean()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"Grad norms: min={min(grad_norms):.2e}, max={max(grad_norms):.2e}")
    print("ConditionalDenoiser smoke test passed.")
