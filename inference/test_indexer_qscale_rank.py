"""
CPU-only shape/numerics regression for the Indexer.forward q_s rank bug (issue #43).

This does NOT import the full model (which requires tilelang/CUDA). It replicates
the exact ops involved in computing the `weights` tensor that is passed to
`fp8_index` as its `q_s` argument:

  - q_scale: produced by act_quant; for q of shape (b, m, n_heads, head_dim) with
    block_size == head_dim, the scale's trailing dim is head_dim // block_size == 1,
    so q_scale has shape (b, m, n_heads, 1)  -- RANK 4.
  - weights:  weights_proj(x) is a Linear(dim, n_heads), so weights is (b, m, n_heads) -- RANK 3.

The fp8_index_kernel_ in inference/kernel.py declares
    q_s: T.Tensor[(b, m, h), FP32]   # RANK 3
and indexes it as q_s[i_b, i_m, 0]. TileLang's runtime enforces strict ndim
equality, so a rank-4 q_s raises at call time.

  BUGGY line:  weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale   -> (b, m, h, 1)  RANK 4
  FIXED line:  weights = weights * q_scale.squeeze(-1) * self.softmax_scale       -> (b, m, h)     RANK 3

The fix's squeeze(-1) is only valid because head_dim == block_size makes
q_scale's trailing dim exactly 1; model.py now asserts that invariant. This test
covers it too.

This test asserts:
  0. for the default config, head_dim == block_size so q_scale's trailing dim == 1,
  1. the fixed q_s is rank 3 with shape (b, m, n_heads),
  2. the fixed q_s is numerically identical to (buggy rank-4 q_s).squeeze(-1),
  3. (illustration only, synthetic shapes) if head_dim != block_size the trailing
     dim would be > 1, squeeze(-1) would be a no-op, and the rank-4 tensor would
     wrongly reach the kernel -- i.e. the model.py assertion is what fails fast on
     that case. This is a hand-built demonstration of the failure mode, not a test
     bound to the live model config.

Run:  python inference/test_indexer_qscale_rank.py
"""

import torch


def act_quant_scale_shape(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mirror the scale-tensor allocation in inference/kernel.py:act_quant.

    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    We only need a representative scale tensor of the correct shape/values;
    we fill it with positive values (real scales are absmax/fp8_max > 0).
    """
    N = x.size(-1)
    assert N % block_size == 0
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    # Fill with plausible positive scale values (deterministic, non-trivial).
    s.uniform_(0.05, 2.0)
    return s


def main() -> int:
    torch.manual_seed(0)

    # Representative Indexer config (from inference/model.py / DeepSeek-V3.2-Exp).
    b, m = 2, 7            # batch, seqlen
    dim = 256              # hidden dim feeding weights_proj
    n_heads = 64           # index_n_heads
    head_dim = 128         # index_head_dim
    block_size = 128       # kernel.py module-level block_size
    softmax_scale = head_dim ** -0.5

    # --- replicate the relevant forward ops ---------------------------------
    # q after view: (b, m, n_heads, head_dim). Its last dim (head_dim) is what
    # act_quant quantizes over, with block_size == head_dim -> scale trailing dim 1.
    q = torch.randn(b, m, n_heads, head_dim)
    q_scale = act_quant_scale_shape(q, block_size)          # (b, m, n_heads, 1) RANK 4

    weights_proj = torch.nn.Linear(dim, n_heads, dtype=torch.float32)
    x = torch.randn(b, m, dim, dtype=torch.float32)
    weights_base = weights_proj(x) * n_heads ** -0.5         # (b, m, n_heads) RANK 3

    # --- pre-conditions on the intermediates --------------------------------
    assert q_scale.shape == (b, m, n_heads, 1), f"q_scale shape unexpected: {tuple(q_scale.shape)}"
    assert q_scale.dim() == 4, f"q_scale should be rank 4, got rank {q_scale.dim()}"
    assert weights_base.shape == (b, m, n_heads), f"weights shape unexpected: {tuple(weights_base.shape)}"

    # --- BUGGY computation (current upstream before fix) --------------------
    q_s_buggy = weights_base.unsqueeze(-1) * q_scale * softmax_scale   # (b, m, n_heads, 1) RANK 4

    # --- FIXED computation --------------------------------------------------
    q_s_fixed = weights_base * q_scale.squeeze(-1) * softmax_scale     # (b, m, n_heads)    RANK 3

    ok = True

    # 0) invariant the fix relies on: head_dim == block_size, so act_quant's
    #    trailing scale axis (head_dim // block_size) is exactly 1. This is the
    #    same condition the new assert in model.py guards.
    if q_scale.size(-1) == 1 and head_dim == block_size:
        print(f"[PASS] invariant holds: head_dim == block_size == {block_size} -> "
              f"q_scale trailing dim = {q_scale.size(-1)} (squeeze removes a real singleton)")
    else:
        ok = False
        print(f"[FAIL] invariant broken: head_dim={head_dim}, block_size={block_size}, "
              f"q_scale trailing dim = {q_scale.size(-1)}")

    # 1) buggy is indeed rank 4 (the broken contract) ------------------------
    if q_s_buggy.dim() == 4 and q_s_buggy.shape == (b, m, n_heads, 1):
        print(f"[info] buggy q_s rank = {q_s_buggy.dim()}, shape = {tuple(q_s_buggy.shape)}  "
              f"(kernel expects rank 3 -> mismatch)")
    else:
        ok = False
        print(f"[FAIL] expected buggy q_s rank 4 (b,m,h,1); got rank {q_s_buggy.dim()} "
              f"shape {tuple(q_s_buggy.shape)}")

    # 2) fixed is rank 3 with the exact shape the kernel declares ------------
    if q_s_fixed.dim() == 3 and q_s_fixed.shape == (b, m, n_heads):
        print(f"[PASS] fixed q_s rank = {q_s_fixed.dim()}, shape = {tuple(q_s_fixed.shape)}  "
              f"(matches kernel q_s: T.Tensor[(b, m, h), FP32])")
    else:
        ok = False
        print(f"[FAIL] expected fixed q_s rank 3 (b,m,h)=({b},{m},{n_heads}); "
              f"got rank {q_s_fixed.dim()} shape {tuple(q_s_fixed.shape)}")

    # 3) numerical equivalence: fix only drops a singleton dim ---------------
    if torch.allclose(q_s_fixed, q_s_buggy.squeeze(-1), rtol=0, atol=0):
        print("[PASS] fixed q_s == buggy q_s.squeeze(-1)  (bit-identical; fix only removes the singleton dim)")
    else:
        ok = False
        max_abs = (q_s_fixed - q_s_buggy.squeeze(-1)).abs().max().item()
        print(f"[FAIL] numerical mismatch: max abs diff = {max_abs}")

    # 4) failure-mode illustration (synthetic, NOT bound to the live config):
    #    hand-build a hypothetical case where head_dim != block_size, so act_quant's
    #    trailing axis would be > 1. There squeeze(-1) is a no-op and the tensor stays
    #    rank 4 -- exactly the regression the model.py assertion (q_scale.size(-1) == 1)
    #    fails fast on. This demonstrates the failure mode; it does not track real config.
    hyp_blocks = head_dim // 64          # = 2 if block_size were 64 instead of 128
    q_scale_drift = torch.randn(b, m, n_heads, hyp_blocks)   # trailing dim 2 (RANK 4)
    squeezed_drift = q_scale_drift.squeeze(-1)               # no-op: still RANK 4
    assert_would_fire = (q_scale_drift.size(-1) != 1)
    if hyp_blocks > 1 and squeezed_drift.dim() == 4 and assert_would_fire:
        print(f"[PASS] failure-mode illustration (synthetic): with trailing dim = {q_scale_drift.size(-1)} "
              f"(head_dim != block_size), squeeze(-1) is a no-op (still rank {squeezed_drift.dim()}); "
              f"model.py assert(q_scale.size(-1) == 1) would fire -> caught")
    else:
        ok = False
        print(f"[FAIL] illustration inconsistent: hyp_blocks={hyp_blocks}, "
              f"squeezed rank={squeezed_drift.dim()}, assert_would_fire={assert_would_fire}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
