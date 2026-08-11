"""nanoGPT-style decoder-only transformer, per the locked architecture specs:

- RoPE (rotary position embeddings) — no learned absolute positions
- Pre-norm (RMSNorm before attention/FFN)
- GELU activation
- UNTIED input embedding / output head (deliberate, for intervention
  cleanliness: resetting wte must not touch lm_head)
- Attention / MLP / Block are separate nn.Module subclasses, each exposing
  reset_parameters(), so the Week-4 intervention module can re-initialize
  any component in isolation mid-training.

Attention uses torch.nn.functional.scaled_dot_product_attention, which
dispatches to FlashAttention-2 kernels on this GPU (the flash-attn pip
package itself does not build on Windows).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


def build_rope_cache(
    block_size: int, head_dim: int, theta: float, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape (block_size, head_dim // 2)."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(block_size, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_head, T, head_dim). Rotates pairs (x1, x2) = even/odd halves."""
    T = x.size(2)
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.n_layer = cfg.n_layer  # for scaled residual init
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.qkv.weight, mean=0.0, std=0.02)
        # nanoGPT-style: scale residual-path projections by 1/sqrt(2*n_layer)
        nn.init.normal_(
            self.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer)
        )

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_layer = cfg.n_layer
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
        nn.init.normal_(
            self.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def reset_parameters(self) -> None:
        self.norm1.reset_parameters()
        self.attn.reset_parameters()
        self.norm2.reset_parameters()
        self.mlp.reset_parameters()

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        # UNTIED output head (locked spec) — never share weight with wte.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        cos, sin = build_rope_cache(cfg.block_size, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.reset_embedding()
        self.reset_lm_head()

    # ---- per-component resets (seams for the Week-4 intervention module) ----
    def reset_embedding(self) -> None:
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)

    def reset_lm_head(self) -> None:
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def reset_block(self, i: int) -> None:
        self.blocks[i].reset_parameters()

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.drop(self.wte(idx))
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def configure_optimizer(
        self, lr: float, weight_decay: float, betas: tuple[float, float], device: str
    ) -> torch.optim.AdamW:
        # Weight decay on matrices only; none on norms (nanoGPT convention).
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device.startswith("cuda")
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
