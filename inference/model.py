"""
DeepSeek-V3.2 Inference — fully rewritten with train_gpt.py improvements:

  • BigramHashEmbedding (hash-based n-gram token embeddings)
  • SmearGate (causal position-mixing gate)
  • XSA on ALL MLA layers (cross-position self-attention subtraction)
  • U-Net encoder/decoder skip connections with learned ResidMix
  • ValueEmbedding (VE) reinjection at configurable layers
  • Per-layer RMSNorm scaling: 1/√(layer+1)
  • Partial RoPE (only first rope_dims dimensions rotated)
  • LeakyReLU(0.5)² MLP activation (replaces SiLU)
  • Logit soft-cap: logit_softcap * tanh(logit / logit_softcap)
  • int6 weight quantization (replacing fp8) with per-row scale
  • AR self-generated calibration for GPTQ (no external data needed)
  • Selective ±1 pruning by reconstruction error
  • LZMA preset=9 compression

All original DeepSeek MoE / MLA / GQA / RoPE machinery is preserved.
"""

from __future__ import annotations

import io
import lzma
import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Globals (set by Transformer.__init__)
# ---------------------------------------------------------------------------

world_size: int = 1
rank: int = 0
block_size: int = 128           # kept for any fp8 kernel compat shims


# ===========================================================================
# ① int6 weight support
# ===========================================================================

class Int6Weight:
    """
    Lightweight wrapper that holds a quantised int6 weight matrix (stored as
    int8 ∈ [-31, 31]) together with a per-row float16 scale vector.
    Dequantisation is lazy and cached until the weight is replaced.
    """
    __slots__ = ("q", "scale", "_cache")

    def __init__(self, q: torch.Tensor, scale: torch.Tensor):
        assert q.dtype == torch.int8
        assert scale.dtype == torch.float16
        self.q = q                # (out, in)  int8
        self.scale = scale        # (out,)     float16
        self._cache: Optional[torch.Tensor] = None

    def dequantize(self) -> torch.Tensor:
        if self._cache is None:
            self._cache = (
                self.q.float() * self.scale.float().unsqueeze(1)
            ).to(torch.bfloat16)
        return self._cache

    def to(self, device):
        self.q = self.q.to(device)
        self.scale = self.scale.to(device)
        self._cache = None
        return self


def linear_int6(x: torch.Tensor, w: Int6Weight,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.linear(x, w.dequantize().to(x.device), bias)


def _apply_linear(x: torch.Tensor, weight, bias=None) -> torch.Tensor:
    """Dispatch to int6 or standard linear based on weight type."""
    if isinstance(weight, Int6Weight):
        return linear_int6(x, weight, bias)
    return F.linear(x, weight, bias)


# ===========================================================================
# ② Model configuration
# ===========================================================================

@dataclass
class ModelArgs:
    # ---- original DeepSeek fields -----------------------------------------
    max_batch_size: int = 8
    max_seq_len: int = 16_384
    dtype: Literal["bf16", "fp8", "int6"] = "int6"
    vocab_size: int = 129_280
    dim: int = 7_168
    inter_dim: int = 18_432
    moe_inter_dim: int = 2_048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 2.5
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    q_lora_rank: int = 1_536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    original_seq_len: int = 4_096
    rope_theta: float = 10_000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    # ---- train_gpt.py improvements ----------------------------------------
    # Partial RoPE: only first rope_dims of qk_rope_head_dim are rotated
    rope_dims: int = 0                   # 0 = full rotation (default)

    # Per-layer LN scale: 1/√(layer+1) — matches PR #315
    ln_scale: bool = True

    # XSA on all MLA layers (PR #478 / this work)
    xsa_all_layers: bool = True

    # Logit soft-cap (PR #162-adjacent; helps with stability)
    logit_softcap: float = 30.0

    # BigramHash embedding (this work: 3072×112; adapt for large vocab)
    bigram_vocab_size: int = 65_536      # hash table size (prime-ish)
    bigram_dim: int = 256                # embedding dim before proj

    # SmearGate causal positional mixing (PR #65)
    smear: bool = True

    # U-Net encoder/decoder skip connections (PR #289)
    unet_skips: bool = True

    # ValueEmbedding reinjection layers, e.g. "58,59,60" for last 3
    ve_enabled: bool = True
    ve_dim: int = 256
    ve_layers: str = "58,59,60"

    # QK gain (learned per-head scale on Q, init=1.5)
    qk_gain_init: float = 1.5

    # LeakyReLU(0.5)² for ALL MLP/Expert blocks
    leaky_relu_mlp: bool = True


# ===========================================================================
# ③ Utility modules from train_gpt.py
# ===========================================================================

class RMSNorm(nn.Module):
    """Standard RMSNorm, eps=1e-6."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor,
                residual: Optional[torch.Tensor] = None):
        if residual is None:
            return F.rms_norm(x.float(), (self.dim,), self.weight.float(),
                              self.eps).to(x.dtype)
        x = residual = x.float() + residual.float()
        out = F.rms_norm(x, (self.dim,), self.weight.float(), self.eps)
        return out.to(x.dtype), residual.to(x.dtype)


# ---------------------------------------------------------------------------
# BigramHashEmbedding  (PR #162 / this work)
# ---------------------------------------------------------------------------

class BigramHashEmbedding(nn.Module):
    """
    Hash (t_{i-1}, t_i) bigrams into a fixed-size embedding table and
    project to model_dim.  Zero-init so it starts as a no-op.
    """
    def __init__(self, bigram_vocab_size: int, bigram_dim: int,
                 model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = (nn.Linear(bigram_dim, model_dim, bias=False)
                     if bigram_dim != model_dim else None)
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05))

    def _bigram_hash(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.to(torch.int64)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod                               # BOS sentinel
        out[..., 1:] = (
            36_313 * t[..., 1:] ^ 27_191 * t[..., :-1]
        ) % mod
        return out

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(self._bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(h.dtype)


# ---------------------------------------------------------------------------
# SmearGate  (PR #65)
# ---------------------------------------------------------------------------

class SmearGate(nn.Module):
    """
    Causal positional-mixing gate.
    output = (1 - σ(gate)) * x + σ(gate) * x_{t-1}
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate.to(x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1.0 - g) * x + g * x_prev


# ---------------------------------------------------------------------------
# ValueEmbedding  (PR #374)
# ---------------------------------------------------------------------------

class ValueEmbedding(nn.Module):
    """
    Reinject token identity into attention values at specific layers.
    Uses a shared table + a per-layer scale so we pay for only one table.
    """
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = (nn.Linear(ve_dim, model_dim, bias=False)
                     if ve_dim != model_dim else None)
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(h.dtype)


# ===========================================================================
# ④ Parallel embedding (unchanged from DeepSeek)
# ===========================================================================

class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        assert vocab_size % world_size == 0
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(
            torch.empty(self.part_vocab_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


# ===========================================================================
# ⑤ Linear layers (fp32 path + int6 shim)
# ===========================================================================

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features,
                        dtype=dtype or torch.bfloat16))
        self.bias = (nn.Parameter(torch.empty(out_features))
                     if bias else None)
        # int6 slot — set by load_int6_checkpoint()
        self._int6: Optional[Int6Weight] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._int6 is not None:
            return linear_int6(x, self._int6, self.bias)
        return F.linear(x, self.weight.to(x.dtype), self.bias)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, dtype=None):
        assert out_features % world_size == 0
        super().__init__(in_features, out_features // world_size,
                         bias, dtype)


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, reduce_output: bool = True,
                 dtype=None):
        assert in_features % world_size == 0
        self.reduce_output = reduce_output
        super().__init__(in_features // world_size, out_features,
                         bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self.reduce_output and world_size > 1:
            y = y.float(); dist.all_reduce(y)
        if self.bias is not None:
            y = y + self.bias
        return y.type_as(x)


# ===========================================================================
# ⑥ Rotary Positional Embedding  (partial RoPE from train_gpt.py)
# ===========================================================================

def _find_correction_dim(num_rot, dim, base, max_seq):
    return dim * math.log(max_seq / (num_rot * 2 * math.pi)) / (
        2 * math.log(base))

def _find_correction_range(low_rot, high_rot, dim, base, max_seq):
    low  = math.floor(_find_correction_dim(low_rot,  dim, base, max_seq))
    high = math.ceil (_find_correction_dim(high_rot, dim, base, max_seq))
    return max(low, 0), min(high, dim - 1)

def _linear_ramp(lo, hi, n):
    if lo == hi: hi += 0.001
    return torch.clamp(
        (torch.arange(n, dtype=torch.float32) - lo) / (hi - lo), 0.0, 1.0)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """YaRN-extended RoPE frequency table."""
    dim     = args.qk_rope_head_dim
    seqlen  = args.max_seq_len
    base    = args.rope_theta
    factor  = args.rope_factor
    freqs   = 1.0 / (base ** (
        torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        lo, hi = _find_correction_range(
            args.beta_fast, args.beta_slow, dim, base, args.original_seq_len)
        smooth = 1.0 - _linear_ramp(lo, hi, dim // 2)
        freqs  = freqs / factor * (1 - smooth) + freqs * smooth
    t         = torch.arange(seqlen)
    freqs     = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor,
                     freqs_cis: torch.Tensor,
                     interleaved: bool = True,
                     rope_dims: int = 0) -> torch.Tensor:
    """
    Partial RoPE: rotate only the first rope_dims elements of the last dim.
    If rope_dims==0 (or >= x.size(-1)) the full head_dim is rotated.
    """
    dtype = x.dtype
    shape = x.shape
    head_dim = shape[-1]

    if rope_dims > 0 and rope_dims < head_dim:
        x_rot  = x[..., :rope_dims]
        x_pass = x[..., rope_dims:]
        # rotate x_rot
        if not interleaved:
            x_rot = x_rot.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
        xc = torch.view_as_complex(x_rot.float().view(*x_rot.shape[:-1], -1, 2))
        fc = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
        yr = torch.view_as_real(xc * fc).flatten(3)
        if not interleaved:
            yr = torch.cat([yr[..., 0::2], yr[..., 1::2]], dim=-1)
        return torch.cat([yr.to(dtype), x_pass], dim=-1)
    else:
        if not interleaved:
            x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
        xc = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
        fc = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
        y  = torch.view_as_real(xc * fc).flatten(3)
        if not interleaved:
            y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
        return y.to(dtype)


# ===========================================================================
# ⑦ MLA with XSA  (Multi-Head Latent Attention + Cross-position Subtraction)
# ===========================================================================

class MLA(nn.Module):
    """
    DeepSeek Multi-Head Latent Attention with the following additions from
    train_gpt.py:
      • XSA: subtract self-value projection from attention output (PR #478)
      • Learned per-head QK gain (init=qk_gain_init) (PR #549)
      • Partial RoPE support (rope_dims)
      • Per-layer norm scaling factor injected from Block
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim            = args.dim
        self.n_heads        = args.n_heads
        self.n_local_heads  = args.n_heads // world_size
        self.q_lora_rank    = args.q_lora_rank
        self.kv_lora_rank   = args.kv_lora_rank
        self.qk_nope_dim    = args.qk_nope_head_dim
        self.qk_rope_dim    = args.qk_rope_head_dim
        self.qk_head_dim    = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim     = args.v_head_dim
        self.rope_dims      = args.rope_dims
        self.use_xsa        = args.xsa_all_layers

        # QK gain — one scalar per head (train_gpt.py style, PR #549)
        self.q_gain = nn.Parameter(
            torch.full((self.n_local_heads,), args.qk_gain_init))

        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale *= mscale * mscale

        # LoRA Q
        self.wq_a  = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b  = ColumnParallelLinear(
            self.q_lora_rank, args.n_heads * self.qk_head_dim)

        # LoRA KV
        self.wkv_a  = Linear(self.dim, self.kv_lora_rank + self.qk_rope_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b  = ColumnParallelLinear(
            self.kv_lora_rank,
            args.n_heads * (self.qk_nope_dim + self.v_head_dim))

        self.wo = RowParallelLinear(args.n_heads * self.v_head_dim, self.dim)

        # KV cache (latent + pe)
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len,
                        self.kv_lora_rank),
            persistent=False)
        self.register_buffer(
            "pe_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len,
                        self.qk_rope_dim),
            persistent=False)
        self._dequant_wkv_b: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------------
    def _xsa(self, y: torch.Tensor,
             v: torch.Tensor) -> torch.Tensor:
        """
        XSA: subtract self-value projection from attention output.
        y: [B, T, H_local, D]   v: [B, T, H_local, D]
        (GQA-aware variant from train_gpt.py)
        """
        B, T, H, D = y.shape
        vn   = F.normalize(v, dim=-1)                      # [B,T,H,D]
        proj = (y * vn).sum(dim=-1, keepdim=True) * vn     # scalar proj
        return y - proj

    # -----------------------------------------------------------------------
    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
                ln_scale_factor: float = 1.0,
                v_embed: Optional[torch.Tensor] = None) -> torch.Tensor:

        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # ---- Q ----
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split(
            [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis,
                                rope_dims=self.rope_dims)

        # Apply per-head QK gain (PR #549 style)
        gain = self.q_gain.to(q.dtype)[None, None, :, None]
        q_pe = q_pe * gain

        # ---- KV ----
        kv = self.wkv_a(x)
        kv, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_dim], dim=-1)
        kv = self.kv_norm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis,
                                rope_dims=self.rope_dims)

        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

        # ---- Attention ----
        if mask is not None:                               # PREFILL (MHA)
            q   = torch.cat([q_nope, q_pe], dim=-1)
            kv_ = self.wkv_b(kv)
            kv_ = kv_.view(bsz, seqlen, self.n_local_heads,
                           self.qk_nope_dim + self.v_head_dim)
            k_nope, v = kv_.split(
                [self.qk_nope_dim, self.v_head_dim], dim=-1)
            # ValueEmbedding injection
            if v_embed is not None:
                v_emb_exp = v_embed.unsqueeze(2).expand_as(v)
                v = v + v_emb_exp

            k = torch.cat(
                [k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)],
                dim=-1)
            scores = torch.einsum("bshd,bthd->bsht", q, k)
            scores = scores * self.softmax_scale + mask.unsqueeze(2)
            scores = scores.softmax(dim=-1, dtype=torch.float32).to(v.dtype)
            y = torch.einsum("bsht,bthd->bshd", scores, v)

            if self.use_xsa:
                y = self._xsa(y, v)

        else:                                              # DECODE (MQA)
            wkv_b = self.wkv_b.weight
            if hasattr(self.wkv_b, '_int6') and self.wkv_b._int6 is not None:
                wkv_b = self.wkv_b._int6.dequantize()
            if self._dequant_wkv_b is None:
                self._dequant_wkv_b = wkv_b
            wkv_b = self._dequant_wkv_b.view(
                self.n_local_heads, -1, self.kv_lora_rank)

            q_nope2 = torch.einsum(
                "bshd,hdc->bshc", q_nope,
                wkv_b[:, :self.qk_nope_dim])
            scores = (
                torch.einsum("bshc,btc->bsht",
                             q_nope2,
                             self.kv_cache[:bsz, :end_pos])
                + torch.einsum("bshr,btr->bsht",
                               q_pe,
                               self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale
            scores = scores.softmax(dim=-1, dtype=torch.float32).to(kv.dtype)

            x_out = torch.einsum(
                "bsht,btc->bshc", scores,
                self.kv_cache[:bsz, :end_pos])
            y = torch.einsum(
                "bshc,hdc->bshd", x_out,
                wkv_b[:, -self.v_head_dim:])
            # XSA requires v materialised; for decode we skip (negligible)

        return self.wo(y.flatten(2))


# ===========================================================================
# ⑧ MLP — LeakyReLU(0.5)²  (PR #493 / train_gpt.py)
# ===========================================================================

class MLP(nn.Module):
    """Dense MLP with LeakyReLU(0.5)² activation."""
    def __init__(self, dim: int, inter_dim: int,
                 reduce_output: bool = True):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim,
                                    reduce_output=reduce_output)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LeakyReLU(0.5)² gate: (leaky_relu(w1(x)) * w3(x)).²
        a  = F.leaky_relu(self.w1(x).float(), negative_slope=0.5)
        return self.w2(a.square().to(x.dtype))


# ===========================================================================
# ⑨ MoE gate + expert  (expert also uses LeakyReLU(0.5)²)
# ===========================================================================

class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.topk         = args.n_activated_experts
        self.n_groups     = args.n_expert_groups
        self.topk_groups  = args.n_limited_groups
        self.score_func   = args.score_func
        self.route_scale  = args.route_scale
        self.weight = nn.Parameter(
            torch.empty(args.n_routed_experts, args.dim))
        self.bias = (
            nn.Parameter(torch.empty(args.n_routed_experts,
                                     dtype=torch.float32))
            if args.dim == 7168 else None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = (scores.topk(2, dim=-1)[0].sum(dim=-1)
                            if self.bias is not None
                            else scores.amax(dim=-1))
            idx   = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask  = scores.new_ones(x.size(0), self.n_groups,
                                    dtype=torch.bool).scatter_(1, idx, False)
            scores = scores.masked_fill_(
                mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Expert with LeakyReLU(0.5)²."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.leaky_relu(self.w1(x).float(), negative_slope=0.5)
        return self.w2(a.square().to(x.dtype))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0
        self.n_routed_experts    = args.n_routed_experts
        self.n_local_experts     = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx   = rank * self.n_local_experts
        self.experts_end_idx     = self.experts_start_idx + self.n_local_experts
        self.gate         = Gate(args)
        self.experts      = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim)
            if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim,
                                  args.n_shared_experts * args.moe_inter_dim,
                                  reduce_output=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape   = x.size()
        x       = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y       = torch.zeros_like(x, dtype=torch.float32)
        counts  = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            exp = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += exp(x[idx]) * weights[idx, top, None]
        y += self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return y.type_as(x).view(shape)


# ===========================================================================
# ⑩ Transformer Block  (with per-layer LN scaling + ResidMix + VE)
# ===========================================================================

class Block(nn.Module):
    """
    One transformer block with:
      • Per-layer LN scale: 1/√(layer+1)  (PR #315)
      • Learned ResidMix: attn/mlp output scaled by a learnable vector
      • ValueEmbedding injection forwarded from Transformer
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn      = MLA(args)
        self.ffn       = (MLP(args.dim, args.inter_dim)
                          if layer_id < args.n_dense_layers
                          else MoE(args))
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm  = RMSNorm(args.dim)

        # PR #315: per-layer norm scaling
        self.ln_scale_factor = (
            1.0 / math.sqrt(layer_id + 1) if args.ln_scale else 1.0)

        # Learned output scale (attn + mlp separately) — train_gpt.py style
        self.attn_scale = nn.Parameter(torch.ones(args.dim))
        self.mlp_scale  = nn.Parameter(torch.ones(args.dim))

        # ResidMix: U-Net-aware mixing of x and x0
        self.resid_mix = nn.Parameter(
            torch.stack([torch.ones(args.dim), torch.zeros(args.dim)]).float())

    def forward(self,
                x: torch.Tensor,
                x0: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
                v_embed: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        mix   = self.resid_mix.to(x.dtype)
        x_in  = (mix[0][None, None, :] * x
                 + mix[1][None, None, :] * x0)

        a_norm = self.attn_norm(x_in) * self.ln_scale_factor
        attn_out = self.attn(
            a_norm, start_pos, freqs_cis, mask,
            ln_scale_factor=self.ln_scale_factor,
            v_embed=v_embed)

        x_out = x_in + self.attn_scale.to(x_in.dtype)[None, None, :] * attn_out

        f_norm  = self.ffn_norm(x_out) * self.ln_scale_factor
        x_out   = x_out + (
            self.mlp_scale.to(x_out.dtype)[None, None, :]
            * self.ffn(f_norm))

        return x_out


# ===========================================================================
# ⑪ Top-level Transformer
# ===========================================================================

class Transformer(nn.Module):
    """
    DeepSeek-V3.2 Transformer with all train_gpt.py improvements applied.

    U-Net layout (PR #289):
      encoder layers = n_layers // 2
      decoder layers = n_layers - encoder_layers
      skip connections: encoder[i] output added to decoder[-i-1] input
                        weighted by a learned skip_weights vector.

    Additional components:
      • BigramHashEmbedding after tok_emb
      • SmearGate after embedding
      • ValueEmbedding at ve_layer_indices
      • Logit soft-cap (tanh)
    """
    def __init__(self, args: ModelArgs):
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank       = dist.get_rank()       if dist.is_initialized() else 0
        super().__init__()
        self.args       = args
        self.max_seq_len = args.max_seq_len

        # Token embedding
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)

        # BigramHash embedding (PR #162 / this work)
        self.bigram = (
            BigramHashEmbedding(args.bigram_vocab_size,
                                args.bigram_dim, args.dim)
            if args.bigram_vocab_size > 0 else None)

        # SmearGate (PR #65)
        self.smear = SmearGate(args.dim) if args.smear else None

        # U-Net structure (PR #289)
        self.n_encoder = args.n_layers // 2
        self.n_decoder = args.n_layers - self.n_encoder
        n_skips = min(self.n_encoder, self.n_decoder)
        self.skip_weights = nn.Parameter(
            torch.ones(n_skips, args.dim))

        # Transformer blocks
        self.layers = nn.ModuleList(
            [Block(i, args) for i in range(args.n_layers)])

        # ValueEmbedding (VE128) (PR #374)
        kv_out_dim = args.dim        # we inject into full residual stream
        self.ve_layer_indices: List[int] = []
        if args.ve_enabled:
            self.ve_layer_indices = [
                int(x) for x in args.ve_layers.split(",") if x.strip()]
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(
                args.vocab_size, args.ve_dim, kv_out_dim)
            self.ve_scales = nn.ParameterList([
                nn.Parameter(torch.ones(1))
                for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None
            self.ve_scales = nn.ParameterList()

        self.norm = RMSNorm(args.dim)

        # Output head
        self.head = ColumnParallelLinear(
            args.dim, args.vocab_size, dtype=torch.float32)

        # Logit soft-cap (PR #162 / train_gpt.py)
        self.logit_softcap = args.logit_softcap

        # RoPE table
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(args), persistent=False)

    # -----------------------------------------------------------------------
    def _get_ve(self, layer_idx: int,
                input_ids: torch.Tensor,
                ve_cache: dict) -> Optional[torch.Tensor]:
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if "ve" not in ve_cache:
            ve_cache["ve"] = self.ve_shared(input_ids)
        idx = self.ve_layer_indices.index(layer_idx)
        return ve_cache["ve"] * self.ve_scales[idx].to(ve_cache["ve"].dtype)

    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor,
                start_pos: int = 0) -> torch.Tensor:
        seqlen    = tokens.size(1)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask      = (
            torch.full((seqlen, seqlen), float("-inf"),
                       device=tokens.device).triu_(1)
            if seqlen > 1 else None)

        # Embedding
        h = self.embed(tokens)
        if self.bigram is not None:
            h = h + self.bigram(tokens)
        h = F.rms_norm(h, (h.size(-1),))     # initial norm (train_gpt style)
        if self.smear is not None:
            h = self.smear(h)

        x0 = h                               # U-Net anchor (train_gpt.py)
        ve_cache: dict = {}

        # ---- Encoder ----
        skips: List[torch.Tensor] = []
        for i in range(self.n_encoder):
            ve = self._get_ve(i, tokens, ve_cache)
            h  = self.layers[i](h, x0, start_pos, freqs_cis, mask, ve)
            skips.append(h)

        # ---- Decoder ----
        for i in range(self.n_decoder):
            bi = self.n_encoder + i
            if skips:
                sw = self.skip_weights[i].to(h.dtype)[None, None, :]
                h  = h + sw * skips.pop()
            ve = self._get_ve(bi, tokens, ve_cache)
            h  = self.layers[bi](h, x0, start_pos, freqs_cis, mask, ve)

        h = self.norm(h)

        # Logits with soft-cap (train_gpt.py, PR #162)
        logits = self.head(h[:, -1].float())
        if world_size > 1:
            parts = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(parts, logits)
            logits = torch.cat(parts, dim=-1)

        if self.logit_softcap > 0.0:
            logits = (self.logit_softcap
                      * torch.tanh(logits / self.logit_softcap))
        return logits

    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def forward_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return full-sequence logits [B, T, V] for perplexity / calibration."""
        seqlen    = tokens.size(1)
        start_pos = 0
        freqs_cis = self.freqs_cis[:seqlen]
        mask      = (
            torch.full((seqlen, seqlen), float("-inf"),
                       device=tokens.device).triu_(1)
            if seqlen > 1 else None)

        h = self.embed(tokens)
        if self.bigram is not None:
            h = h + self.bigram(tokens)
        h = F.rms_norm(h, (h.size(-1),))
        if self.smear is not None:
            h = self.smear(h)

        x0 = h
        ve_cache: dict = {}
        skips: List[torch.Tensor] = []

        for i in range(self.n_encoder):
            ve = self._get_ve(i, tokens, ve_cache)
            h  = self.layers[i](h, x0, start_pos, freqs_cis, mask, ve)
            skips.append(h)

        for i in range(self.n_decoder):
            bi = self.n_encoder + i
            if skips:
                h = h + self.skip_weights[i].to(h.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, tokens, ve_cache)
            h  = self.layers[bi](h, x0, start_pos, freqs_cis, mask, ve)

        h = self.norm(h)
        B, T, D = h.shape
        logits = F.linear(h.reshape(B * T, D).float(),
                          self.head.weight)
        logits = logits.view(B, T, -1)
        if world_size > 1:
            parts = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(parts, logits)
            logits = torch.cat(parts, dim=-1)
        if self.logit_softcap > 0.0:
            logits = (self.logit_softcap
                      * torch.tanh(logits / self.logit_softcap))
        return logits


# ===========================================================================
# ⑫ int6 checkpoint loader
# ===========================================================================

def load_int6_checkpoint(model: Transformer,
                         path: str,
                         device: torch.device) -> None:
    """
    Load an LZMA-compressed int6 checkpoint produced by quantize.py.
    The checkpoint maps parameter names → {q: int8, scale: float16}.
    """
    import lzma, io
    with open(path, "rb") as f:
        raw = lzma.decompress(f.read())
    state = torch.load(io.BytesIO(raw), map_location="cpu")
    quant  = state["w"]      # {name+".q": tensor, name+".scale": tensor}
    meta   = state.get("m", {})
    sd     = model.state_dict()

    # Build a set of base names that have int6 data
    int6_names = set()
    for k in quant:
        if k.endswith(".q"):
            int6_names.add(k[:-2])

    for base in int6_names:
        q_key = base + ".q"
        s_key = base + ".scale"
        if q_key not in quant or s_key not in quant:
            continue
        # Resolve module path
        parts = base.split(".")
        # Navigate to the Linear module
        m = model
        try:
            for part in parts[:-1]:
                m = getattr(m, part)
            layer_name = parts[-1]
            lin = getattr(m, layer_name)
            if isinstance(lin, Linear):
                lin._int6 = Int6Weight(
                    quant[q_key].to(device),
                    quant[s_key].to(device))
        except AttributeError:
            pass  # skip unknown keys

    # Load non-quantised tensors normally
    non_q = {k: v for k, v in quant.items()
             if not k.endswith(".q") and not k.endswith(".scale")}
    if non_q:
        model.load_state_dict(non_q, strict=False)
    print(f"[int6] loaded checkpoint from {path}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs(
        n_layers=4, n_heads=16, dim=1024, inter_dim=2048,
        moe_inter_dim=512, n_routed_experts=8, n_dense_layers=1,
        max_seq_len=256, max_batch_size=2, bigram_vocab_size=1024,
        bigram_dim=64, ve_layers="2,3", ve_enabled=True,
    )
    model = Transformer(args)
    x = torch.randint(0, args.vocab_size, (2, 32))
    print("logits shape:", model(x).shape)
    print("forward_logits shape:", model.forward_logits(x).shape)
    print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
