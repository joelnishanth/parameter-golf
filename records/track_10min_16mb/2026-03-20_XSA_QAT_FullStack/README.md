# 11L + XSA + Int6 STE QAT + Full Stack

**Author:** Joel Ponukumatla
**Date:** 2026-03-20
**Score:** val_bpb = pending (pending cloud run on 8xH100)

## Summary

Combines three proven techniques that have never been stacked together:
1. **PR #198 base** (1.1318 BPB): 11L, SmearGate, BigramHash, OrthoInit+muP, WD=0.04, SWA, FA3
2. **XSA from PR #265** (1.1307 BPB): Exclusive Self Attention on last 3 layers — subtracts self-value projection from attention output to reduce self-attention bias in deep layers (arXiv:2603.09078)
3. **Int6 STE QAT from PR #194** (1.1480 BPB): Fake int6 quantization during training with straight-through estimator — model learns quantization-robust weights from step 0

Key insight: PR #198 has a 0.011 BPB post-quant gap (1.1432 pre-quant → 1.1543 post-quant). QAT should close most of this gap. XSA adds a clean ~0.001-0.002 BPB architectural improvement at minimal compute cost (~2ms/step for 3 layers).

## Architecture

| Component | Details |
|-----------|---------|
| **Layers** | 11 transformer layers, 512 dim, 8 heads, 4 KV heads (GQA) |
| **MLP** | 3x expansion (hidden=1536), ReLU² activation |
| **XSA** | Efficient partial XSA on last 3 layers (GQA-aware, no memory overhead) |
| **Quantization** | Int6 STE QAT during training; int6 mixed quant (MLP+attn), int8 (embeddings) at export |
| **Compression** | zstd-22 |
| **SmearGate** | Per-dim learned sigmoid gate (~512 params) |
| **BigramHash** | 2048-bucket hash embedding (dim=128→512, scale=0.05) |
| **Initialization** | Orthogonal + muP (proj scaled by 1/√(2·L)) |
| **Optimizer** | Muon (WD=0.04, momentum=0.99, warmup 0.92→0.99 over 1500 steps) |
| **SWA** | Every 120 steps during warmdown (scale < 0.5) |
| **Attention** | FlashAttention 3 (Hopper) |
| **Sequence** | Train@2048, eval@2048 |
| **Eval** | Sliding window stride=64 |

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All defaults are baked into the script. Equivalent explicit invocation:

```bash
NUM_LAYERS=11 MLP_MULT=3 BIGRAM_VOCAB_SIZE=2048 \
QAT_ENABLED=1 XSA_LAST_N=3 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
SWA_ENABLED=1 SWA_EVERY=120 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Results

Based on component analysis:
- PR #198 base: ~1.1318 BPB (stride=64 sliding, int6 roundtrip)
- + XSA on last 3 layers: ~-0.001 BPB (from PR #265)
- + Int6 STE QAT: ~-0.005 BPB (close most of the 0.011 post-quant gap)
- **Expected: ~1.124–1.128 BPB**

## References

- PR #198 (jfprincz): Base stack — 1.1318 BPB
- PR #265 (unnir): XSA — arXiv:2603.09078
- PR #194 (baudrillardsgh0st): Int6 STE QAT + SWA
