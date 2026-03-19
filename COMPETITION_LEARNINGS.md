# Parameter Golf — Competition Intelligence

> Last updated: 2026-03-19. Re-scan PRs regularly as the frontier moves daily.

---

## Current state of play

| Source | BPB | Status |
|--------|-----|--------|
| Naive baseline (official, merged) | 1.2244 | Official SOTA (pre-PR activity) |
| FP16 Embed (official, merged) | 1.2197 | Official SOTA |
| 4-hour unlimited run | 1.2074 | Non-record, unlimited compute |
| PR #60 — Sliding Window + 10L + MuonWD | 1.1748 | Open PR, pending review |
| PR #89 — NorMuon + int6 STE + SWA + SW | 1.1622 | Open PR, pending review |
| PR #99 — Int6 + MLP3x + Late-K + SW | 1.1605 | Open PR, pending review |
| PR #88 — Int6 + MLP3x + MTP + seq4096 | 1.1605 | Open PR, pending review |
| PR #106 — Built on #88 + QAT export | 1.1580 | Open PR, pending review |

**Target to win**: beat PR #106 by ≥0.005 → need ≤ **1.153 BPB**.

---

## The consensus stack (5 techniques everyone converges on)

These five appear in virtually every competitive PR. Not having them means starting ~0.06 BPB behind the frontier.

### 1. int6 quantization + zstd compression
- Replaces the baseline's int8 + zlib pipeline
- int6 stores values in [-31, 31] per row (6 bits/param vs 8 bits/param)
- zstd level 22 beats zlib level 9 for int8/int6 weight data
- Saves ~25% artifact space → directly enables a wider MLP
- Requires `pip install zstandard`
- Key change: modify `quantize_state_dict_int8` in `train_gpt.py` to use 6-bit range and swap `zlib.compress` for `zstandard.ZstdCompressor(level=22).compress`

### 2. MLP 3× expansion (hidden=1536)
- Baseline uses `MLP_MULT=2` → hidden=1024
- int6 frees enough space to fit hidden=1536 under 16MB
- Directly improves model capacity per compressed byte
- Controlled via `MLP_HIDDEN=1536` env var (already plumbed in the FP16Embed record script)

### 3. FP16 tied embedding passthrough
- Already in the official SOTA; every top PR keeps this
- `tok_emb.weight` serves double duty as input embedding AND output logit head → most quant-sensitive tensor
- Change: in the quantization loop, add `or name == "tok_emb.weight"` to the fp16 passthrough branch
- Drops the embedding quant gap from ~0.007 BPB to ~0.0005 BPB

### 4. Sliding window evaluation
- Evaluate with a stride < seq_len so each token is scored with more context
- Legal: challenge rules explicitly say "we allow evaluation at any sequence length"
- Current top PRs use stride 64–512; stride 64 gives ~960 tokens of context per token
- Worth ~0.033 BPB by itself — the single largest eval-side improvement
- Implementation: change the val loop to slide a window over the val set with overlap, average per-token losses
- Eval must complete in ≤10 min on 8×H100 (stride 64 at seq_len 1024 is ~97s per PR #88)

### 5. Co-optimized LR / schedule for the int6 + MLP3x regime
- The original warmdown and LR values were tuned for the baseline config, not the int6+MLP3x one
- PR #88 converged on: `MATRIX_LR=0.02`, `MUON_MOMENTUM=0.99`, `WARMDOWN_ITERS=3000`
- These are different from the FP16Embed SOTA values (`MATRIX_LR=0.06`, `WARMDOWN_ITERS=3600`)

---

## Layer 2: Proven add-ons (not universal but validated)

### int6 STE QAT (PR #89, PR #106)
- Fake per-row int6 quantization during the forward pass, straight-through estimator in the backward pass
- Model learns to be robust to quantization from step 0
- Reduces post-quant gap to +0.002 BPB (vs ~0.03 without QAT)
- Implementation: wrap each `CastedLinear.forward` with a fake-quant op during training, disable at eval

### SWA — Stochastic Weight Averaging (PR #89)
- Save checkpoints every N steps during warmdown, average their weights before final eval
- PR #89 used 7 checkpoints
- Free quality improvement; ~0.002–0.005 BPB benefit
- No additional compute cost; just checkpoint management

### NorMuon — Normalized Muon (PR #89)
- Row-normalize the Newton-Schulz update matrix before applying it
- From modded-nanogpt; improves training stability, especially at higher LR
- Small code change to `zeropower_via_newtonschulz5` or the Muon step function

### seq4096 training (PR #88)
- Train with `TRAIN_SEQ_LEN=4096` instead of 1024
- Model sees longer dependencies during training, which pairs well with sliding window eval at test time
- Slight throughput hit (fewer steps in 10 min) but higher quality per step

### MTP auxiliary head (PR #88)
- An extra linear head predicts next-2 and next-3 tokens during training only
- Excluded from the artifact (training-only parameter)
- Provides richer gradient signal at zero artifact cost
- Implementation: add a small detached prediction head after the final norm; add aux loss to main loss with small weight

### Muon weight decay (PR #60)
- `p.mul_(1 - wd * lr)` applied before each Muon step (decoupled weight decay)
- Muon has no built-in regularization; WD=0.02 improves both generalization and quantization robustness
- Also reduces artifact size slightly (more structured/compressible weights)

---

## Our differentiating bets (untried territory)

### Bet A: Narrower sliding window stride (stride=1–16) — LOWEST RISK
- All top PRs use stride 64–512. The quality gain should continue improving with smaller strides.
- At stride=8, every token gets `seq_len - 8` tokens of context (1016 at seq_len=1024)
- Eval time: needs benchmarking. At stride=64 it's ~97s. At stride=8 it would be ~8× longer (~13 min) — likely too slow for 8×H100.
- Stride=32 (~4× slower than stride=64 → ~6 min) is probably the sweet spot to test first.
- Zero architecture change needed, pure eval loop modification.

### Bet B: FP4 QAT (4-bit quantization) — HIGH RISK / HIGH REWARD
- int4 per-row: range [-7, 7] or [-15, 15] with scale
- Would save another ~33% vs int6 → enables MLP 4× or 5×
- Requires very strong STE QAT from step 0 to prevent quality collapse
- Nobody has a clean working version in any open PR
- If it works: enormous capacity budget unlocked

### Bet C: Test-time training at eval — MEDIUM RISK
- Run a few unsupervised gradient steps (predict next token on the test doc itself) before scoring
- Legal: doesn't access training data, just adapts to the specific test document
- PR #77 tried LoRA TTT and got 1.195 BPB — but that was early, without the consensus stack
- The combination of full stack + TTT is unexplored
- Budget: ~5 min of the 10 min eval cap could be spent on TTT gradient steps

---

## Things that have been tried and FAILED

- **SwiGLU activation**: better per-step quality but 45% slower → fewer steps in 10 min → net negative
- **Depth recurrence at 10 min**: needs far more steps to converge; promising for unlimited compute track
- **lzma compression**: worse than zlib for int8 weight data (confirmed by official SOTA author)
- **Higher embed LR** (0.08 vs 0.05): hurts convergence
- **Late-stage QAT** (applied only in last N steps): not enough training signal; overhead not worth it
- **Larger vocab (8192)** so far: PR #78 (1.186), PR #92 (1.194) — competitive but not near frontier
- **8192 vocab + NorMuon + selective quant** (PR #78): 1.186 BPB — decent but behind int6 stack

---

## Key constraint reminders

- Artifact = `len(train_gpt.py in UTF-8)` + `compressed_model_bytes` ≤ **16,000,000 bytes** (decimal, not MiB)
- Training ≤ **10 min** on 8×H100 SXM
- Eval ≤ **10 min** on 8×H100 (separate from training time!)
- No network calls, no external downloads at eval time
- All counted code must live in `train_gpt.py`
- Leaderboard submission requires ≥0.005 nats improvement over current SOTA
- 3 seeds required to demonstrate `p < 0.01`

---

## Architecture quick-reference (baseline)

```
9 layers × 512 dim
8 heads, 4 KV heads (GQA), head_dim=64
MLP: relu^2, hidden=1024 (MLP_MULT=2)
Vocab: 1024 (SentencePiece BPE)
Tied embeddings: yes
Skip connections: encoder/decoder U-Net style
RoPE positional encoding, RMSNorm
Logit softcap: 30.0
```

All hyperparameters are env-var driven (see `Hyperparameters` class in `train_gpt.py` lines 39–87).

---

## File conventions

- All competitive work goes in `records/track_10min_16mb/<date>_<name>/`
- Each record folder must contain: `train_gpt.py`, `README.md`, `submission.json`, at least one `train.log`
- The `train_gpt.py` in the record folder must run self-contained from that directory
- Root `train_gpt.py` stays ≤1500 lines and is not the competition config
