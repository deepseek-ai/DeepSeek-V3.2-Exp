"""
quantize.py — Full Hessian GPTQ int6 quantisation for DeepSeek-V3.2

Ported from train_gpt.py (PR #535 / this work):

  • Full Hessian GPTQ with Cholesky error compensation + column reordering
    (strictly better than diagonal-Hessian / GPTQ-lite)
  • AR Self-Generated calibration: after training the model generates its own
    calibration tokens (N seqs × L tokens, temp=0.8, fixed seed).
    No val data, no train data accessed during quantisation.
  • Percentile-search clip with MSE oracle across 5 candidates
  • Selective ±1 pruning by reconstruction error (binary-search to fit TARGET_MB)
  • LZMA preset=9 compression of the final checkpoint

Usage:
    from quantize import quantize_model, load_int6_checkpoint
    ckpt_bytes = quantize_model(model, target_mb=15.9, seed=314)
    with open("model.int6.lzma", "wb") as f:
        f.write(ckpt_bytes)
"""

from __future__ import annotations

import io
import lzma
import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Transformer, Linear, Int6Weight


# ---------------------------------------------------------------------------
# Config knobs
# ---------------------------------------------------------------------------

CLIP_PERCENTILES   = [0.9990, 0.9995, 0.9999, 0.99999, 1.0]
INT6_CLIP_RANGE    = 31          # ±31 ∈ int8
GPTQ_BLOCK_SIZE    = 128         # columns per GPTQ sweep block
GPTQ_DAMP_RATIO    = 0.01        # H damping: damp = ratio × mean(diag(H))
MIN_NUMEL_QUANTISE = 4_096       # skip tensors smaller than this

# Layers to quantise (all Linear weights except very small ones)
INT6_CATS = {"mlp", "attn", "head", "embed"}


# ===========================================================================
# ① AR Self-Generated Calibration Data
# ===========================================================================

@torch.inference_mode()
def generate_ar_calibration(
    model: Transformer,
    device: torch.device,
    num_seqs: int = 64,
    seq_len:  int = 2048,
    temperature: float = 0.8,
    batch_size:  int = 8,
    seed: int = 42,
) -> list[torch.Tensor]:
    """
    Autoregressively generate calibration sequences from the model itself.
    No external data — fully self-contained (AR self-gen, this work).

    Returns:
        list of [1, seq_len] int64 token tensors
    """
    vocab_size = model.args.vocab_size
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens: list[torch.Tensor] = []

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, num_seqs, batch_size):
            bs     = min(batch_size, num_seqs - batch_start)
            tokens = torch.randint(
                0, vocab_size, (bs, 1), device=device, generator=rng)

            for _ in range(seq_len - 1):
                logits     = model.forward_logits(tokens)
                next_logit = logits[:, -1, :]
                probs      = torch.softmax(next_logit / temperature, dim=-1)
                next_tok   = torch.multinomial(probs, 1, generator=rng)
                tokens     = torch.cat([tokens, next_tok], dim=1)

            for i in range(bs):
                all_tokens.append(tokens[i : i + 1].cpu())

    return all_tokens


# ===========================================================================
# ② Hessian Collection  H = X^T X
# ===========================================================================

def collect_hessians(
    model: Transformer,
    token_seqs: list[torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Forward each calibration sequence through the model, collecting
    H = X^T X for every nn.Linear layer (using forward hooks).
    """
    hessians: dict[str, torch.Tensor] = {}
    hooks:    list = []

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, Linear)):
            continue
        w = (module.weight if isinstance(module, nn.Linear)
             else module.weight)
        cols = w.shape[1]
        pname = name + ".weight"
        hessians[pname] = torch.zeros(cols, cols, dtype=torch.float32,
                                      device="cpu")

        def _make_hook(pn):
            def _hook(mod, inp, _out):
                x = inp[0].detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                hessians[pn] += (x.T @ x).cpu()
            return _hook

        hooks.append(module.register_forward_hook(_make_hook(pname)))

    model.eval()
    with torch.inference_mode(), \
         torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for seq in token_seqs:
            # forward_logits accepts [B, T] tokens
            model.forward_logits(seq.to(device))

    for h in hooks:
        h.remove()

    n_batches = len(token_seqs)
    for pname in hessians:
        H    = hessians[pname]
        H   /= n_batches
        damp = GPTQ_DAMP_RATIO * torch.diag(H).mean().clamp_min(1e-6)
        H   += damp * torch.eye(H.shape[0])
        hessians[pname] = H

    return hessians


# ===========================================================================
# ③ Full Hessian GPTQ int6
# ===========================================================================

def _percentile_clip(t32: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Brute-force MSE oracle over CLIP_PERCENTILES. Returns (best_q, best_scale).
    """
    best_q, best_s, best_err = None, None, float("inf")
    for pct in CLIP_PERCENTILES:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / INT6_CLIP_RANGE).clamp_min(
            1.0 / INT6_CLIP_RANGE).to(torch.float16)
        q = torch.clamp(
            torch.round(t32 / s.float().unsqueeze(1)),
            -INT6_CLIP_RANGE, INT6_CLIP_RANGE).to(torch.int8)
        err = (t32 - q.float() * s.float().unsqueeze(1)).pow(2).mean().item()
        if err < best_err:
            best_q, best_s, best_err = q, s, err
    return best_q, best_s          # type: ignore[return-value]


def quantize_int6_gptq(
    weight: torch.Tensor,
    hessian: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full Hessian GPTQ int6 with Cholesky error compensation + column
    reordering.  Falls back to percentile search when no Hessian is given.

    Returns:
        q     (rows, cols)  int8 ∈ [-31, 31]
        scale (rows,)       float16
    """
    t32 = weight.float()

    if t32.ndim != 2 or hessian is None:
        # 1-D or no Hessian → simple percentile fallback
        if t32.ndim == 1:
            amax  = t32.abs().max().item()
            scale = torch.tensor(
                amax / INT6_CLIP_RANGE if amax > 0 else 1.0,
                dtype=torch.float16)
            q = torch.clamp(
                torch.round(t32 / scale.float()),
                -INT6_CLIP_RANGE, INT6_CLIP_RANGE).to(torch.int8)
            return q, scale
        return _percentile_clip(t32)

    rows, cols = t32.shape
    H    = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0

    # Damping
    damp = GPTQ_DAMP_RATIO * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp

    # Column reordering (most sensitive first)
    perm     = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0.0
    H = H[perm][:, perm]

    # Cholesky inverse
    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    best_q, best_s, best_err = None, None, float("inf")

    for pct in CLIP_PERCENTILES:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s  = (row_clip / INT6_CLIP_RANGE).clamp_min(
            1.0 / INT6_CLIP_RANGE).to(torch.float16)
        sf = s.float()

        Q      = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()

        for i1 in range(0, cols, GPTQ_BLOCK_SIZE):
            i2    = min(i1 + GPTQ_BLOCK_SIZE, cols)
            count = i2 - i1
            W1    = W_work[:, i1:i2].clone()
            Q1    = torch.zeros(rows, count, dtype=torch.int8)
            Err1  = torch.zeros(rows, count)
            Hin1  = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w  = W1[:, i]
                d  = Hin1[i, i]
                q  = torch.clamp(
                    torch.round(w / sf),
                    -INT6_CLIP_RANGE, INT6_CLIP_RANGE).to(torch.int8)
                Q1[:, i]   = q
                err        = (w - q.float() * sf) / d
                W1[:, i:]  -= err.unsqueeze(1) * Hin1[i, i:].unsqueeze(0)
                Err1[:, i] = err

            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        recon = Q.float() * sf.unsqueeze(1)
        mse   = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_s, best_err = Q, s, mse

    # Undo column reordering
    assert best_q is not None
    best_q = best_q[:, inv_perm]
    return best_q, best_s          # type: ignore[return-value]


# ===========================================================================
# ④ Quantise full model state-dict
# ===========================================================================

def _classify(name: str) -> str:
    if "tok_emb" in name or "embed" in name:
        return "embed"
    if ".mlp." in name or "ffn" in name:
        return "mlp"
    if ".attn." in name or "wq" in name or "wkv" in name or "wo" in name:
        return "attn"
    if "head" in name:
        return "head"
    return "other"


def quantise_state_dict(
    sd: dict[str, torch.Tensor],
    hessians: Optional[dict[str, torch.Tensor]] = None,
    int6_cats: set[str] = INT6_CATS,
) -> tuple[dict[str, torch.Tensor], dict]:
    """
    Quantise a flat state-dict (key → tensor) to int6 where appropriate.

    Returns:
        quant_result  — {name+".q": int8, name+".scale": float16, ...passthrough}
        quant_meta    — {name: {"type": "int6"}} for reconstructed info
    """
    result: dict[str, torch.Tensor] = {}
    meta:   dict[str, object] = {}

    for name, t in sd.items():
        t_cpu = t.detach().cpu().float()

        # Only quantise 2-D float tensors of sufficient size
        cat = _classify(name)
        should_quant = (
            t.is_floating_point()
            and t.ndim == 2
            and t.numel() >= MIN_NUMEL_QUANTISE
            and cat in int6_cats
        )

        if not should_quant:
            result[name] = t.detach().cpu()
            continue

        H = (hessians.get(name) if hessians else None)
        q, scale = quantize_int6_gptq(t_cpu, H)
        result[name + ".q"]     = q
        result[name + ".scale"] = scale
        meta[name] = {"type": "int6"}

    return result, meta


# ===========================================================================
# ⑤ Selective ±1 pruning  (this work / PR #609)
# ===========================================================================

def selective_prune(
    quant_result: dict[str, torch.Tensor],
    quant_meta: dict,
    target_bytes: int,
    code_bytes: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Sort ±1-valued quantised weights by their reconstruction error (scale²),
    prune least-impactful ones first until the LZMA-compressed bundle fits
    target_bytes.  Binary-search for the minimal pruning count.

    Returns updated quant_result.
    """
    ones_info: list[tuple[str, int, float]] = []

    for name, info in quant_meta.items():
        if not (isinstance(info, dict) and info.get("type") == "int6"):
            continue
        qk = name + ".q"
        sk = name + ".scale"
        if qk not in quant_result or sk not in quant_result:
            continue
        q, s = quant_result[qk], quant_result[sk]
        if s.ndim == 0 or q.numel() == 0:
            continue
        ones_mask = q.abs() == 1
        if not ones_mask.any():
            continue
        row_idx  = torch.arange(q.shape[0]).unsqueeze(1).expand_as(q)[ones_mask]
        flat_idx = torch.arange(q.numel()).reshape(q.shape)[ones_mask]
        errors   = s.float()[row_idx].pow(2)
        for fi, err in zip(flat_idx.tolist(), errors.tolist()):
            ones_info.append((qk, fi, err))

    if not ones_info:
        print("selective_prune: no ±1 candidates found")
        return quant_result

    ones_info.sort(key=lambda x: x[2])   # cheapest pruning first

    def _compressed_size(n: int) -> tuple[int, dict]:
        tmp = {k: v.clone() for k, v in quant_result.items()}
        for i in range(min(n, len(ones_info))):
            tmp[ones_info[i][0]].view(-1)[ones_info[i][1]] = 0
        buf = io.BytesIO()
        torch.save({"w": tmp, "m": quant_meta}, buf)
        return len(lzma.compress(buf.getvalue(), preset=9)) + code_bytes, tmp

    base_sz, _ = _compressed_size(0)
    print(f"selective_prune: {len(ones_info)} ±1 candidates, "
          f"unpruned={base_sz/1024/1024:.2f}MB "
          f"target={target_bytes/1024/1024:.2f}MB")

    if base_sz <= target_bytes:
        print("selective_prune: already fits, no pruning needed")
        return quant_result

    full_sz, _ = _compressed_size(len(ones_info))
    print(f"selective_prune: full ±1 prune={full_sz/1024/1024:.2f}MB")

    if full_sz > target_bytes:
        print("selective_prune: even full prune not enough; applying all")
        _, pruned = _compressed_size(len(ones_info))
        return pruned

    # Binary search for minimal pruning count
    lo, hi = 0, len(ones_info)
    while lo < hi:
        mid  = (lo + hi) // 2
        sz, _ = _compressed_size(mid)
        if sz <= target_bytes:
            hi = mid
        else:
            lo = mid + 1

    pct = 100.0 * lo / len(ones_info)
    print(f"selective_prune: pruning {lo}/{len(ones_info)} "
          f"({pct:.1f}%) ±1 values to fit target")
    _, pruned = _compressed_size(lo)
    return pruned


# ===========================================================================
# ⑥ Top-level: quantise + calibrate + compress
# ===========================================================================

def quantize_model(
    model: Transformer,
    device: torch.device,
    target_mb: float = 15.9,
    seed: int = 314,
    num_calib_seqs: int = 64,
    calib_seq_len: int = 2_048,
    calib_temp: float = 0.8,
    calib_batch_size: int = 8,
) -> bytes:
    """
    Full quantisation pipeline:
      1. AR self-generated calibration (no external data)
      2. Hessian collection via forward hooks
      3. Full Hessian GPTQ int6 (Cholesky + column reorder)
      4. Selective ±1 pruning
      5. LZMA preset=9 compression

    Returns the compressed bytes ready to write to disk.
    """
    t0 = time.perf_counter()

    # 1. Generate calibration sequences
    print(f"[quant] Generating {num_calib_seqs}×{calib_seq_len} AR "
          f"calibration seqs (temp={calib_temp}, seed={seed}) ...")
    ar_seqs = generate_ar_calibration(
        model, device,
        num_seqs=num_calib_seqs,
        seq_len=calib_seq_len,
        temperature=calib_temp,
        batch_size=calib_batch_size,
        seed=seed,
    )
    print(f"[quant]   done in {time.perf_counter()-t0:.1f}s")

    # 2. Collect Hessians
    t1 = time.perf_counter()
    print("[quant] Collecting Hessians ...")
    # Extract a flat state-dict with unbanked names for Hessian matching
    sd_flat = {k: v.detach().cpu() for k, v in model.named_parameters()}
    hessians = collect_hessians(model, ar_seqs, device)
    print(f"[quant]   {len(hessians)} Hessians collected in "
          f"{time.perf_counter()-t1:.1f}s")

    # 3. Quantise
    t2 = time.perf_counter()
    print("[quant] Running GPTQ int6 ...")
    quant_result, quant_meta = quantise_state_dict(
        sd_flat, hessians=hessians)
    print(f"[quant]   quantised in {time.perf_counter()-t2:.1f}s")

    # 4. Selective pruning
    target_bytes = int(target_mb * 1024 * 1024)
    quant_result = selective_prune(
        quant_result, quant_meta, target_bytes)

    # 5. LZMA compress
    t3 = time.perf_counter()
    buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, buf)
    compressed = lzma.compress(buf.getvalue(), preset=9)
    print(f"[quant] Compressed: {len(compressed)/1024/1024:.2f}MB "
          f"in {time.perf_counter()-t3:.1f}s  "
          f"(total {time.perf_counter()-t0:.1f}s)")
    return compressed


# ===========================================================================
# ⑦ Dequantise for inference reload
# ===========================================================================

def dequantize_state_dict(
    quant_result: dict[str, torch.Tensor],
    quant_meta: dict,
    reference_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Reconstruct a bf16 state-dict from int6 quant_result.
    """
    out: dict[str, torch.Tensor] = {}
    processed: set[str] = set()

    for name, info in quant_meta.items():
        if not (isinstance(info, dict) and info.get("type") == "int6"):
            continue
        qk = name + ".q"
        sk = name + ".scale"
        if qk not in quant_result or sk not in quant_result:
            continue
        q  = quant_result[qk].float()
        s  = quant_result[sk].float()
        if s.ndim > 0:
            dq = q * s.unsqueeze(1)
        else:
            dq = q * s.item()
        # Cast to match reference dtype
        ref_dtype = reference_sd.get(name, torch.empty(0)).dtype
        out[name] = dq.to(ref_dtype if ref_dtype.is_floating_point()
                          else torch.bfloat16)
        processed.add(qk)
        processed.add(sk)

    for k, v in quant_result.items():
        if k not in processed:
            out[k] = v.detach().cpu()

    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path",   required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--out",         default="model.int6.lzma")
    parser.add_argument("--target-mb",   type=float, default=15.9)
    parser.add_argument("--seed",        type=int,   default=314)
    parser.add_argument("--calib-seqs",  type=int,   default=64)
    parser.add_argument("--calib-len",   type=int,   default=2048)
    args = parser.parse_args()

    import json
    from safetensors.torch import load_model
    from model import ModelArgs, Transformer

    device = torch.device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    with open(args.config) as f:
        model_args = ModelArgs(**json.load(f))
    model = Transformer(model_args).to(device)
    load_model(model, args.ckpt_path)

    compressed = quantize_model(
        model, device,
        target_mb=args.target_mb,
        seed=args.seed,
        num_calib_seqs=args.calib_seqs,
        calib_seq_len=args.calib_len,
    )
    with open(args.out, "wb") as f:
        f.write(compressed)
    print(f"Saved to {args.out}  ({len(compressed)/1024/1024:.2f} MB)")
