# Consensus Stack

Implements all five techniques that appear across every top open PR (~1.158–1.162 BPB range as of 2026-03-19).

## What changed vs the naive baseline

### 1. int6 quantization + zstd-22 compression
Weights stored in [-31, 31] per row (6-bit) instead of [-127, 127] (8-bit). zstd level 22 replaces zlib level 9.
Net result: ~25% smaller compressed artifact, freeing budget for a wider MLP.

### 2. MLP 3× expansion (hidden=1536)
Enabled by int6 space savings. Default `MLP_HIDDEN=1536` overrides `MLP_MULT=2`.

### 3. FP16 tied embedding passthrough
`tok_emb.weight` skips int6 quantization and is stored as fp16. It serves as both input embedding and output logit head — quantizing it in int6 causes errors to compound in both directions. Drops the embedding quant gap from ~0.007 to ~0.0005 BPB.

### 4. Sliding window evaluation
`eval_val_sliding` slides a window of `train_seq_len` tokens by `val_stride=64` at a time. Only the final `val_stride` tokens of each window (those with the most context) are scored. Every scored token gets up to `seq_len - stride = 960` tokens of context vs. only `rand(0, seq_len)` in the standard non-overlapping eval.

Both standard BPB and sliding window BPB are logged for transparency.

### 5. Co-optimized LR / schedule
Defaults tuned for the int6 + MLP3x regime (from PR #88):
- `MATRIX_LR`: 0.04 → 0.02
- `MUON_MOMENTUM`: 0.95 → 0.99
- `WARMDOWN_ITERS`: 1200 → 3000

## Run command (8×H100)

```bash
RUN_ID=consensus_stack_seed1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Next steps (not yet in this submission)
- int6 STE QAT (PR #89 technique)
- SWA over warmdown checkpoints
- NorMuon optimizer
- MTP auxiliary head
- seq4096 training context
