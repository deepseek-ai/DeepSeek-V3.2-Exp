# DeepSeek-V3.2 Inference — Rewritten with train_gpt.py Improvements

This is a complete rewrite of the DeepSeek-V3.2-Exp inference code
(`model.py`, `generate.py`) with every architectural improvement from
`train_gpt.py` (PR #549 / #609 SOTA submission) applied to the full
MoE model.

---

## What Changed and Why

### 1. int6 Quantisation (replaces fp8) — `model.py`, `quantize.py`

| Original | This work |
|---|---|
| fp8 (e4m3) weight quantisation | int6 per-row quantisation (±31, stored int8) |
| TileLang custom kernel | Standard `F.linear` on dequantised weights |
| Diagonal-Hessian GPTQ-lite | **Full Hessian GPTQ** with Cholesky error compensation + column reordering |
| Train data calibration | **AR self-generated calibration** (64 seqs × 2048 tokens, temp=0.8) |

The full Hessian GPTQ is strictly better than the diagonal approximation:
```
H_Cholesky = cholesky_inverse(cholesky(H + damp·I))
For each column block: q = round(w / scale), err = (w - q·scale) / Hinv_ii
W[:, i:] -= err · Hinv[i, i:]          # error propagation
```

AR self-generated calibration means **zero external data** is accessed
during quantisation: the model autoregressively generates its own
calibration sequences after training completes.

### 2. Selective ±1 Pruning — `quantize.py`

After quantisation, ±1 valued entries are sorted by reconstruction
error (scale²).  A binary search finds the minimum number to zero-out
so the LZMA-compressed bundle fits the target size:

```python
ones_info.sort(key=lambda x: x[2])   # cheapest first
# binary search for lo ∈ [0, len(ones_info)]
```

### 3. LZMA preset=9 — `quantize.py`

```python
lzma.compress(payload, preset=9)      # maximum compression
```

### 4. XSA on All Layers — `model.py` `MLA._xsa()`

Cross-position Self-Attention subtraction (PR #478) applied to **all**
61 MLA layers (previously only last 4).  Forces cross-position
information mixing from layer 0 at zero extra parameter cost:

```python
def _xsa(self, y, v):
    vn   = F.normalize(v, dim=-1)
    proj = (y * vn).sum(dim=-1, keepdim=True) * vn
    return y - proj
```

### 5. BigramHashEmbedding — `model.py` `BigramHashEmbedding`

Hash (t_{i-1}, t_i) bigram pairs into a 65,536-entry table, project
to model_dim, and add to the token embedding.  Zero-initialised so it
starts as a no-op and learns residual bigram statistics:

```python
hash = (36313 * t[1:] XOR 27191 * t[:-1]) % (bigram_vocab_size - 1)
```

### 6. SmearGate — `model.py` `SmearGate`

Causal position-mixing gate (PR #65).  Learns a per-dimension blend
of the current and previous token representations:

```python
out = (1 - σ(gate)) * x + σ(gate) * x_{t-1}
```

### 7. U-Net Encoder/Decoder with Skip Connections — `model.py` `Transformer`

Layers 0..29 = encoder (save activations).
Layers 30..60 = decoder (add weighted skip from encoder mirror):

```python
h = h + skip_weights[i][None, None, :] * skips.pop()
```

Learned `skip_weights` (ones-init, per-dimension) let the decoder
selectively re-use encoder features.

### 8. ResidMix — `model.py` `Block`

Each block mixes its input with the initial embedding `x0` via learned
coefficients (U-Net anchor, train_gpt.py style):

```python
x_in = mix[0] * x + mix[1] * x0
```

### 9. ValueEmbedding (VE) at Last 3 Layers — `model.py` `ValueEmbedding`

Reinjects token identity into the attention value stream at layers
58, 59, 60 (PR #374).  Uses a shared embedding table + per-layer
learned scale:

```python
ve = ve_shared(tokens) * ve_scales[layer_idx]
```

### 10. Per-Layer RMSNorm Scaling — `model.py` `Block`

```python
ln_scale_factor = 1 / sqrt(layer_id + 1)
normed = attn_norm(x) * ln_scale_factor
```

Matches PR #315.  Stabilises deep layers by reducing the effective
norm magnitude as depth increases.

### 11. LeakyReLU(0.5)² MLP Activation — `model.py` `MLP`, `Expert`

Replaces SiLU gating with LeakyReLU(0.5)² (PR #493):

```python
# Original:  w2(silu(w1(x)) * w3(x))
# This work: w2(leaky_relu(w1(x), 0.5)²)
a = F.leaky_relu(w1(x), negative_slope=0.5)
out = w2(a.square())
```

### 12. Logit Soft-Cap — `model.py` `Transformer.forward()`

```python
logits = logit_softcap * tanh(logits / logit_softcap)
```

Prevents logit explosion, improves training stability and calibration
of the final distribution (train_gpt.py, PR #162-adjacent).

### 13. Partial RoPE — `model.py` `apply_rotary_emb()`

Only the first `rope_dims=16` of the 64 RoPE head dimensions are
rotated; the rest pass through unchanged (PR #315):

```python
x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
# rotate x_rope, concatenate x_pass unchanged
```

### 14. Learned Per-Head QK Gain — `model.py` `MLA`

A per-head scalar gain on Q (init = 1.5, PR #549):

```python
gain = self.q_gain[None, None, :, None]   # [1,1,H_local,1]
q_pe = q_pe * gain
```

### 15. Top-p Sampling — `generate.py`

In addition to Gumbel/temperature sampling, nucleus (top-p) sampling
is now supported:

```bash
torchrun ... generate.py --temperature 0.6 --top-p 0.9
```

### 16. Sliding-Window BPB Evaluation — `generate.py` `eval_bpb_sliding()`

Exact sliding-window evaluation matching the leaderboard metric
(stride=64 default).

---

## Run Commands

### Interactive generation (bf16):
```bash
export CONFIG=config_671B_v3.2_enhanced.json
torchrun --nproc-per-node 8 generate.py \
  --ckpt-path /path/to/deepseek-v3.2 \
  --config $CONFIG \
  --interactive
```

### Quantise → int6 + LZMA9, then generate:
```bash
torchrun --nproc-per-node 8 generate.py \
  --ckpt-path /path/to/deepseek-v3.2 \
  --config $CONFIG \
  --quantise --target-mb 15900 --seed 314 \
  --interactive
```

### Quantise only (no generation):
```bash
python quantize.py \
  --ckpt-path /path/to/deepseek-v3.2/model0-mp8.safetensors \
  --config $CONFIG \
  --out model.int6.lzma \
  --target-mb 15900 \
  --seed 314 \
  --calib-seqs 64 --calib-len 2048
```

### Load pre-quantised int6 checkpoint:
```bash
torchrun --nproc-per-node 8 generate.py \
  --ckpt-path model.int6.lzma \
  --config $CONFIG \
  --interactive
```

---

## File Map

| File | Role |
|---|---|
| `model.py` | Full model with all 14 architectural improvements |
| `quantize.py` | GPTQ int6 + AR calibration + selective pruning + LZMA9 |
| `generate.py` | Generation loop, top-p sampling, sliding BPB eval |
| `config_671B_v3.2_enhanced.json` | Config with all new knobs enabled |

Files removed from original:
- `kernel.py` (fp8 TileLang kernels — superseded by int6 path)
- `convert.py` (safetensors sharding — still usable as-is)
