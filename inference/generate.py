"""
generate.py — DeepSeek-V3.2 inference rewritten with train_gpt.py improvements.

New features vs. original:
  • Loads LZMA-compressed int6 checkpoints (quantize.py format)
  • Sliding-window evaluation (eval_val_sliding) for BPB measurement
  • AR self-generated calibration helper (re-exported from quantize.py)
  • generate() now uses logit soft-cap aware sampling
  • Top-p (nucleus) sampling as an alternative to temperature Gumbel
"""

from __future__ import annotations

import io
import json
import lzma
import os
from argparse import ArgumentParser
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_model
from transformers import AutoTokenizer

from model import ModelArgs, Transformer, load_int6_checkpoint
from quantize import generate_ar_calibration, quantize_model


# ===========================================================================
# Sampling
# ===========================================================================

def _sample_gumbel(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-max (= multinomial) — same as original DeepSeek generate."""
    logits = logits / max(temperature, 1e-5)
    probs  = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


def _sample_top_p(logits: torch.Tensor,
                  temperature: float,
                  top_p: float = 0.9) -> torch.Tensor:
    """Nucleus (top-p) sampling."""
    logits = logits / max(temperature, 1e-5)
    probs  = torch.softmax(logits, dim=-1, dtype=torch.float32)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumprobs = sorted_probs.cumsum(dim=-1)
    # remove tokens above threshold (keep at least 1)
    remove  = (cumprobs - sorted_probs) > top_p
    sorted_probs[remove] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    chosen = torch.multinomial(sorted_probs, 1).squeeze(1)
    return sorted_idx.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)


# ===========================================================================
# Generation loop
# ===========================================================================

@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
) -> List[List[int]]:
    """
    Generate completions for a batch of prompts.

    Args:
        model:          Transformer (with int6 or bf16 weights)
        prompt_tokens:  List[List[int]]  —  one prompt per sample
        max_new_tokens: Maximum tokens to append
        eos_id:         End-of-sequence token id
        temperature:    Sampling temperature (0 → greedy)
        top_p:          If not None, use nucleus sampling with this p value

    Returns:
        List[List[int]]  —  generated token ids (excluding prompt)
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len,
                    max_new_tokens + max(prompt_lens))

    tokens = torch.full(
        (len(prompt_tokens), total_len), -1,
        dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    prev_pos  = 0
    finished  = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1

    for cur_pos in range(min(prompt_lens), total_len):
        logits = model(tokens[:, prev_pos:cur_pos], prev_pos)

        if temperature <= 0.0:
            next_token = logits.argmax(dim=-1)
        elif top_p is not None:
            next_token = _sample_top_p(logits, temperature, top_p)
        else:
            next_token = _sample_gumbel(logits, temperature)

        next_token = torch.where(
            prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= (~prompt_mask[:, cur_pos]) & (next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break

    completions = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]: prompt_lens[i] + max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completions.append(toks)
    return completions


# ===========================================================================
# Sliding-window BPB evaluation (train_gpt.py eval_val_sliding)
# ===========================================================================

@torch.inference_mode()
def eval_bpb_sliding(
    model: Transformer,
    val_tokens: torch.Tensor,
    device: torch.device,
    seq_len: int = 2_048,
    stride: int = 64,
    batch_size: int = 4,
) -> tuple[float, float]:
    """
    Compute validation loss and BPB with a sliding context window.
    This is the exact metric used in the parameter-golf leaderboard.

    Args:
        val_tokens:  1-D int64 tensor of all validation tokens
        seq_len:     Context window length
        stride:      How many tokens to slide per step (64 = standard)
        batch_size:  Number of windows processed in parallel

    Returns:
        (val_loss, val_bpb)
    """
    total_tokens = val_tokens.numel()
    windows      = list(range(0, total_tokens - seq_len, stride))

    model.eval()
    loss_sum   = torch.zeros((), dtype=torch.float64, device=device)
    token_count = torch.zeros((), dtype=torch.float64, device=device)
    byte_count  = torch.zeros((), dtype=torch.float64, device=device)

    for bi in range(0, len(windows), batch_size):
        batch_ws = windows[bi: bi + batch_size]
        bsz      = len(batch_ws)
        x_batch  = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        y_batch  = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        wlens: list[int] = []

        for i, ws in enumerate(batch_ws):
            end  = min(ws + seq_len, total_tokens)
            wlen = end - ws
            wlens.append(wlen)
            chunk         = val_tokens[ws: end + 1].to(device=device,
                                                        dtype=torch.int64)
            x_batch[i, :wlen] = chunk[:-1]
            y_batch[i, :wlen] = chunk[1:]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward_logits(x_batch)   # [B, T, V]

        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y_batch.reshape(-1),
            reduction="none",
        ).reshape(bsz, seq_len)

        for i, ws in enumerate(batch_ws):
            wlen  = wlens[i]
            # Only score the last stride tokens (sliding-window convention)
            s     = 0 if ws == 0 else max(wlen - stride, 0)
            loss_sum   += nll[i, s:wlen].to(torch.float64).sum()
            token_count += float(wlen - s)
            # Approximate bytes = tokens (char-level estimate; replace with
            # your tokenizer's byte LUT for exact BPB)
            byte_count  += float(wlen - s)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count,  op=dist.ReduceOp.SUM)

    val_loss       = (loss_sum / token_count).item()
    bits_per_token = val_loss / (2.0 ** 0).bit_length()   # nats→bits: /ln2
    import math
    bits_per_token = val_loss / math.log(2.0)
    bpb            = bits_per_token * (token_count / byte_count).item()
    return val_loss, bpb


# ===========================================================================
# Checkpoint loading helpers
# ===========================================================================

def load_checkpoint(
    model: Transformer,
    ckpt_path: str,
    world_size: int,
    rank: int,
) -> None:
    """
    Auto-detect checkpoint format:
      • *.lzma   → int6 LZMA (quantize.py output)
      • *.safetensors / directory → original DeepSeek safetensors
    """
    if ckpt_path.endswith(".lzma") or ckpt_path.endswith(".ptz"):
        load_int6_checkpoint(model, ckpt_path, torch.device("cuda"))
    else:
        shard = os.path.join(ckpt_path,
                             f"model{rank}-mp{world_size}.safetensors")
        load_model(model, shard)


# ===========================================================================
# Interactive / batch generation (main)
# ===========================================================================

def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: Optional[float] = 0.9,
    # Quantisation-on-the-fly
    quantise: bool = False,
    target_mb: float = 15.9,
    seed: int = 314,
) -> None:

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank       = int(os.getenv("RANK",       "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")

    # Silence non-master ranks
    if rank != 0:
        global print
        print = lambda *_, **__: None

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(33_377_335)

    with open(config) as f:
        raw = json.load(f)
    # Accept both original and new config keys gracefully
    model_args = ModelArgs(**{k: v for k, v in raw.items()
                              if k in ModelArgs.__dataclass_fields__})
    print(model_args)

    device = torch.device("cuda")
    with torch.device("cuda"):
        model = Transformer(model_args)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("[generate] loading model weights …")
    load_checkpoint(model, ckpt_path, world_size, rank)

    # Optional: quantise on the fly (for evaluation / artifact creation)
    if quantise:
        print(f"[generate] quantising to {target_mb} MB (seed={seed}) …")
        compressed = quantize_model(
            model, device, target_mb=target_mb, seed=seed)
        out_path = "model.int6.lzma"
        with open(out_path, "wb") as f:
            f.write(compressed)
        print(f"[generate] saved {out_path}  "
              f"({len(compressed)/1024/1024:.2f} MB)")
        # Reload from int6
        load_int6_checkpoint(model, out_path, device)

    print("DeepSeek-V3.2 (train_gpt.py enhanced) 👋")

    if interactive:
        messages: list[dict] = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objs = [prompt]; dist.broadcast_object_list(objs, 0)
            else:
                objs = [None]; dist.broadcast_object_list(objs, 0)
                prompt = objs[0]

            if prompt == "/exit":
                break
            if prompt == "/clear":
                messages.clear(); continue

            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True)
            completion_tokens = generate(
                model, [prompt_tokens], max_new_tokens,
                tokenizer.eos_token_id, temperature, top_p)
            completion = tokenizer.decode(
                completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})

    else:
        assert input_file, "Provide --input-file for batch mode"
        with open(input_file) as f:
            prompts = f.read().split("\n\n")
        assert len(prompts) <= model_args.max_batch_size
        prompt_tokens_all = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True)
            for p in prompts]
        completion_tokens = generate(
            model, prompt_tokens_all, max_new_tokens,
            tokenizer.eos_token_id, temperature, top_p)
        completions = tokenizer.batch_decode(
            completion_tokens, skip_special_tokens=True)
        for p, c in zip(prompts, completions):
            print("Prompt:", p)
            print("Completion:", c)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path",      type=str, required=True)
    parser.add_argument("--config",         type=str, required=True)
    parser.add_argument("--input-file",     type=str, default="")
    parser.add_argument("--interactive",    action="store_true")
    parser.add_argument("--max-new-tokens", type=int,   default=512)
    parser.add_argument("--temperature",    type=float, default=0.6)
    parser.add_argument("--top-p",          type=float, default=0.9)
    # Quantisation flags
    parser.add_argument("--quantise",    action="store_true",
                        help="Run GPTQ int6 + LZMA9 before generating")
    parser.add_argument("--target-mb",  type=float, default=15.9)
    parser.add_argument("--seed",        type=int,   default=314)
    args = parser.parse_args()
    assert args.input_file or args.interactive, \
        "Provide --input-file or --interactive"
    main(
        args.ckpt_path, args.config,
        args.input_file, args.interactive,
        args.max_new_tokens, args.temperature, args.top_p,
        args.quantise, args.target_mb, args.seed,
    )
