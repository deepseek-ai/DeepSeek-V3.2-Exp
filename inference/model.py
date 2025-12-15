import math
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Dict

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    from kernel import act_quant, fp8_gemm, fp8_index
except ImportError as e:
    logging.warning(f"kernel module import failed: {e}. Some features may not work.")
    # Define fallback functions
    def act_quant(x, block_size, scale_fmt):
        return x, None
    
    def fp8_gemm(x, x_scale, weight, weight_scale):
        raise ImportError("fp8_gemm requires kernel module")
    
    def fp8_index(q_fp8, weights, k_cache, k_scale_cache):
        raise ImportError("fp8_index requires kernel module")

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError as e:
    logging.warning(f"fast_hadamard_transform import failed: {e}. Using fallback.")
    # Fallback implementation
    def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Simple hadamard transform fallback - less efficient but functional"""
        return x * scale


# Configure logging
logger = logging.getLogger(__name__)

# Global configuration with safe defaults
world_size = 1
rank = 0
block_size = 128

# Distributed operation timeout (seconds)
DIST_TIMEOUT = 60.0


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        scale_fmt (Optional[str]): Format for quantization scale.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
        index_n_heads (int): Number of index heads.
        index_head_dim (int): Dimension for index head.
        index_topk (int): Top-k for index head.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048


def validate_distributed_params(param_name: str, param_value: int, divisor: int):
    """
    Validate that a parameter is divisible by world_size.
    
    Args:
        param_name: Name of the parameter for error messages
        param_value: Value of the parameter to validate
        divisor: Expected divisor (usually world_size)
        
    Raises:
        ValueError: If parameter is not divisible by divisor
    """
    if divisor > 1 and param_value % divisor != 0:
        raise ValueError(
            f"{param_name} ({param_value}) must be divisible by world_size ({divisor}) "
            f"for distributed training. Remainder: {param_value % divisor}"
        )


def safe_distributed_operation(func):
    """
    Decorator for safe distributed operations with error handling.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Distributed operation failed in {func.__name__}: {e}")
            raise
    return wrapper


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Apply hadamard transform with scaling.
    
    Args:
        x (torch.Tensor): Input tensor of dtype bfloat16
        
    Returns:
        torch.Tensor: Transformed tensor
    """
    if x.dtype != torch.bfloat16:
        logger.warning(f"rotate_activation expects bfloat16, got {x.dtype}")
    
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Validate distributed parameters
        if world_size > 1:
            validate_distributed_params("vocab_size", vocab_size, world_size)
        
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        
        # Validate indices
        if self.vocab_start_idx >= vocab_size or self.vocab_end_idx > vocab_size:
            raise ValueError(
                f"Calculated vocabulary indices out of bounds for rank {rank}. "
                f"Start: {self.vocab_start_idx}, End: {self.vocab_end_idx}, "
                f"Vocab Size: {vocab_size}"
            )
        
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    @safe_distributed_operation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x_local = x - self.vocab_start_idx
            x_local[mask] = 0
            y = F.embedding(x_local, self.weight)
            y[mask] = 0
            dist.all_reduce(y, timeout=DIST_TIMEOUT)
        else:
            y = F.embedding(x, self.weight)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           scale_fmt: Optional[str] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor.
        bias (Optional[torch.Tensor]): The bias tensor to be added.
        scale_fmt (Optional[str]): The format of scaling factors.

    Returns:
        torch.Tensor: The result of the linear transformation.
    """
    if bias is not None:
        raise NotImplementedError("Bias not supported in quantized linear operations")
    
    if weight.dtype != torch.float8_e4m3fn:
        return F.linear(x, weight)
    else:
        x, scale = act_quant(x, block_size, scale_fmt)
        return fp8_gemm(x, scale, weight, weight.scale)


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Validate parameters
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"Invalid dimensions: in_features={in_features}, out_features={out_features}")
        
        dtype = dtype or Linear.dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )
        else:
            self.register_parameter("scale", None)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias, self.scale_fmt)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        if world_size > 1:
            validate_distributed_params("out_features", out_features, world_size)
        
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        return linear(x, self.weight, self.bias, self.scale_fmt)


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, 
                 reduce_output: bool = True, dtype = None):
        if world_size > 1:
            validate_distributed_params("in_features", in_features, world_size)
        
        self.part_in_features = in_features // world_size
        self.reduce_output = reduce_output
        super().__init__(self.part_in_features, out_features, bias, dtype)

    @safe_distributed_operation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        # Validate input shape
        if x.size(-1) != self.part_in_features * world_size:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) does not match "
                f"expected dimension ({self.part_in_features * world_size})"
            )
        
        # Get local partition
        start_idx = rank * self.part_in_features
        end_idx = start_idx + self.part_in_features
        x_local = x[..., start_idx:end_idx]
        
        y = linear(x_local, self.weight, None, self.scale_fmt)
        
        if self.reduce_output and world_size > 1:
            y_float = y.float()
            dist.all_reduce(y_float, timeout=DIST_TIMEOUT)
            y = y_float.type_as(x)
        
        if self.bias is not None:
            y += self.bias
        
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.
            residual (Optional[torch.Tensor]): Optional residual tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        dtype = x.dtype
        
        if residual is None:
            x_float = x.float()
            var = x_float.pow(2).mean(-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(var + self.eps)
            return (self.weight * x_norm).to(dtype)
        else:
            # Fused operation with residual
            x_combined = x.float() + residual.float()
            var = x_combined.pow(2).mean(-1, keepdim=True)
            x_norm = x_combined * torch.rsqrt(var + self.eps)
            return (self.weight * x_norm).to(dtype), x_combined.to(dtype)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, device='cpu')  # Keep on CPU until needed
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.
        interleaved (bool): Whether to use interleaved format. Defaults to True.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    shape = x.shape
    
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    
    x_complex = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    y = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    
    return y.to(dtype)


class Indexer(torch.nn.Module):
    """
    Indexer module for selecting top-k tokens for attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim: int = args.dim
        self.n_heads: int = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_rope_head_dim
        self.index_topk: int = args.index_topk
        self.q_lora_rank: int = args.q_lora_rank
        self.scale_fmt = args.scale_fmt
        
        if world_size > 1:
            validate_distributed_params("index_n_heads", args.index_n_heads, world_size)
        
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk = Linear(self.dim, self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)
        self.softmax_scale = self.head_dim ** -0.5

        self.register_buffer("k_cache", 
            torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, 
                       dtype=torch.float8_e4m3fn), 
            persistent=False)
        self.register_buffer("k_scale_cache", 
            torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim // block_size, 
                       dtype=torch.float32), 
            persistent=False)

    @safe_distributed_operation
    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, 
                freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for Indexer.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            qr: Query representation [batch, seq_len, q_lora_rank]
            start_pos: Starting position in sequence
            freqs_cis: Rotary embedding frequencies
            mask: Optional attention mask
            
        Returns:
            topk_indices: Indices of top-k tokens for attention
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Validate cache bounds
        if end_pos > self.k_cache.size(1):
            raise ValueError(
                f"Sequence end position {end_pos} exceeds cache size {self.k_cache.size(1)}"
            )
        
        # Compute queries
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
        q = torch.cat([q_pe, q_nope], dim=-1)
        
        # Compute keys
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        
        # Apply hadamard transform
        q = rotate_activation(q)
        k = rotate_activation(k)
        
        # Quantize and cache keys
        q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
        
        self.k_cache[:bsz, start_pos:end_pos] = k_fp8
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
        
        # Compute attention weights and scores
        weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        
        index_score = fp8_index(
            q_fp8.contiguous(), 
            weights, 
            self.k_cache[:bsz, :end_pos].contiguous(), 
            self.k_scale_cache[:bsz, :end_pos].contiguous()
        )
        
        if mask is not None:
            index_score += mask
        
        # Select top-k indices
        topk = min(self.index_topk, end_pos)
        topk_indices = index_score.topk(topk, dim=-1)[1]
        
        # Synchronize across GPUs if distributed
        if world_size > 1:
            topk_indices_ = topk_indices.clone()
            dist.broadcast(topk_indices_, src=0, timeout=DIST_TIMEOUT)
            if not torch.all(topk_indices == topk_indices_):
                logger.warning("Top-k indices mismatch across GPUs, using rank 0 values")
                topk_indices = topk_indices_
        
        return topk_indices


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize weight tensor using scale.
    
    Args:
        weight: Quantized weight tensor
        scale: Scale tensor
        
    Returns:
        Dequantized weight tensor
    """
    if weight.dim() != 2:
        raise ValueError(f"Weight must be 2D tensor, got shape {weight.shape}")
    
    shape = weight.shape
    weight = weight.view(shape[0] // block_size, block_size, shape[1] // block_size, block_size)
    weight = weight.transpose(1, 2).contiguous().view(-1, block_size * block_size)
    weight = (weight.float() * scale.view(-1, 1).float()).to(torch.get_default_dtype())
    weight = weight.view(shape[0] // block_size, shape[1] // block_size, block_size, block_size)
    weight = weight.transpose(1, 2).contiguous().view(shape)
    return weight


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        
        if world_size > 1:
            validate_distributed_params("n_heads", args.n_heads, world_size)

        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.scale_fmt = args.scale_fmt
        
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.indexer = Indexer(args)

        self.register_buffer("kv_cache", 
            torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank),
            persistent=False)
        self.register_buffer("pe_cache", 
            torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim),
            persistent=False)
        self.dequant_wkv_b = None

    @safe_distributed_operation
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, 
                mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Validate cache bounds
        if end_pos > self.kv_cache.size(1):
            raise ValueError(
                f"Sequence end position {end_pos} exceeds cache size {self.kv_cache.size(1)}"
            )
        
        # Compute query representation
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        # Compute key-value representation
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        # Quantize and cache KV
        kv_fp8, kv_scale = act_quant(kv, block_size, self.scale_fmt)
        kv = (kv_fp8.view(-1, block_size).float() * kv_scale.view(-1, 1)).to(kv.dtype).view_as(kv)
        
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        
        if mask is not None:    # MHA prefill
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(kv)
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            scores = torch.einsum("bshd,bthd->bsht", q, k).mul_(self.softmax_scale)

            # Indexer for token selection
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
            index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device)
            index_mask.scatter_(-1, topk_indices, 0)
            index_mask += mask
            scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1)
            x = torch.einsum("bsht,bthd->bshd", scores, v)
        else:                   # MQA decode
            # Lazy dequantization of wkv_b weights
            if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
                self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
            
            wkv_b = self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

            # Indexer for token selection
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
            index_mask = torch.full((bsz, 1, end_pos), float("-inf"), device=x.device)
            index_mask.scatter_(-1, topk_indices, 0)
            scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1)
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.
    """
    def __init__(self, dim: int, inter_dim: int, reduce_output: bool = True):
        super().__init__()
        self.dim = dim
        self.inter_dim = inter_dim
        
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim, reduce_output=reduce_output)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        if x.size(-1) != self.dim:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) does not match "
                f"MLP dimension ({self.dim})"
            )
        
        w1_out = self.w1(x).float()
        w3_out = self.w3(x).float()
        return self.w2((F.silu(w1_out) * w3_out).type_as(x))


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        # Note: Magic number 7168 should be documented or made configurable
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) \
            if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        if x.size(-1) != self.dim:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) does not match "
                f"gate dimension ({self.dim})"
            )
        
        scores = linear(x.float(), self.weight.float())
        
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()
        
        original_scores = scores
        
        if self.bias is not None:
            scores = scores + self.bias
        
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.
    """
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.dim = dim
        self.inter_dim = inter_dim
        
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        if x.size(-1) != self.dim:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) does not match "
                f"expert dimension ({self.dim})"
            )
        
        w1_out = self.w1(x).float()
        w3_out = self.w3(x).float()
        return self.w2((F.silu(w1_out) * w3_out).type_as(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        
        if world_size > 1:
            validate_distributed_params("n_routed_experts", args.n_routed_experts, world_size)
        
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        
        # Validate expert indices
        if self.experts_start_idx >= args.n_routed_experts or self.experts_end_idx > args.n_routed_experts:
            raise ValueError(
                f"Calculated expert indices out of bounds for rank {rank}. "
                f"Start: {self.experts_start_idx}, End: {self.experts_end_idx}, "
                f"Total Experts: {args.n_routed_experts}"
            )
        
        self.gate = Gate(args)
        
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(self.n_routed_experts)
        ])
        
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim, reduce_output=False)

    @safe_distributed_operation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        if x.size(-1) != self.dim:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) does not match "
                f"MoE dimension ({self.dim})"
            )
        
        shape = x.size()
        x_flat = x.view(-1, self.dim)
        
        weights, indices = self.gate(x_flat)
        y = torch.zeros_like(x_flat, dtype=torch.float32)
        
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            
            expert = self.experts[i]
            if expert is None:
                continue
            
            idx, top = torch.where(indices == i)
            if len(idx) > 0:
                y[idx] += expert(x_flat[idx]) * weights[idx, top, None]
        
        y += self.shared_experts(x_flat)
        
        if world_size > 1:
            dist.all_reduce(y, timeout=DIST_TIMEOUT)
        
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, residual: torch.Tensor, start_pos: int, 
                freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            residual (torch.Tensor): Residual tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output and residual tensors.
        """
        if residual is None:
            x, residual = self.attn_norm(x), x
        else:
            x, residual = self.attn_norm(x, residual)
        
        x = self.attn(x, start_pos, freqs_cis, mask)
        x, residual = self.ffn_norm(x, residual)
        x = self.ffn(x)
        
        return x, residual


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.
    """
    def __init__(self, args: ModelArgs):
        global world_size, rank
        
        # Initialize distributed settings
        try:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                logger.info(f"Distributed environment: world_size={world_size}, rank={rank}")
            else:
                world_size = 1
                rank = 0
                logger.info("Running in single-GPU mode")
        except Exception as e:
            logger.warning(f"Failed to initialize distributed settings: {e}")
            world_size = 1
            rank = 0
        
        # Set data types
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        Linear.scale_fmt = args.scale_fmt
        
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        
        # Validate parameters
        if world_size > 1:
            validate_distributed_params("vocab_size", args.vocab_size, world_size)
            validate_distributed_params("n_heads", args.n_heads, world_size)
            validate_distributed_params("n_routed_experts", args.n_routed_experts, world_size)
            validate_distributed_params("index_n_heads", args.index_n_heads, world_size)
        
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList([
            Block(layer_id, args) for layer_id in range(args.n_layers)
        ])
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.float32)
        
        # Precompute rotary embeddings on CPU, move to device when needed
        freqs_cis = precompute_freqs_cis(args)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @torch.inference_mode()
    @safe_distributed_operation
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        # Input validation
        if tokens.dim() != 2:
            raise ValueError(f"Tokens must be 2D tensor, got shape {tokens.shape}")
        
        bsz, seqlen = tokens.size()
        
        if bsz > self.args.max_batch_size:
            raise ValueError(
                f"Batch size {bsz} exceeds maximum batch size {self.args.max_batch_size}"
            )
        
        if start_pos + seqlen > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seqlen} at position {start_pos} exceeds "
                f"maximum sequence length {self.max_seq_len}"
            )
        
        # Move rotary embeddings to the same device as tokens
        if self.freqs_cis.device != tokens.device:
            self.freqs_cis = self.freqs_cis.to(tokens.device)
        
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1) \
            if seqlen > 1 else None
        
        h, residual = self.embed(tokens), None
        
        for layer_idx, layer in enumerate(self.layers):
            try:
                h, residual = layer(h, residual, start_pos, freqs_cis, mask)
            except Exception as e:
                logger.error(f"Layer {layer_idx} failed: {e}")
                raise
        
        h, _ = self.norm(h, residual)
        logits = self.head(h[:, -1].float())
        
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits, timeout=DIST_TIMEOUT)
            logits = torch.cat(all_logits, dim=-1)
        
        return logits


def setup_distributed():
    """
    Setup distributed environment if needed.
    
    Returns:
        tuple: (world_size, rank) if distributed is initialized
    """
    global world_size, rank
    
    try:
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            logger.info(f"Distributed setup complete: world_size={world_size}, rank={rank}")
            return world_size, rank
        else:
            world_size = 1
            rank = 0
            logger.info("Single-GPU mode")
            return world_size, rank
    except Exception as e:
        logger.error(f"Failed to setup distributed environment: {e}")
        world_size = 1
        rank = 0
        return world_size, rank


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup distributed environment
    setup_distributed()
    
    # Run test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.manual_seed(42)
    
    args = ModelArgs()
    
    # Test with small batch
    test_batch_size = min(2, args.max_batch_size)
    test_seq_len = min(32, args.max_seq_len)
    
    x = torch.randint(0, args.vocab_size, (test_batch_size, test_seq_len), device=device)
    
    try:
        model = Transformer(args)
        model.to(device)
        
        # Test forward pass
        with torch.no_grad():
            output = model(x)
            print(f"✓ Success! Output size: {output.size()}")
            
            # Test memory usage
            if torch.cuda.is_available():
                print(f"✓ GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
