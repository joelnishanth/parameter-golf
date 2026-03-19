# Parameter Golf — repository guide for assistants

This file summarizes how the repo is organized, what the scripts do, and what constraints matter when changing code. The canonical participant-facing narrative is [README.md](README.md).

## What this project is

**OpenAI Model Craft Challenge: Parameter Golf.** Competitors train a small language model whose **submission artifact** (training script bytes + **zlib-compressed int8** serialized weights) must stay under **16,000,000 bytes (decimal MB)**. Leaderboard runs must **train in ≤10 minutes on 8×H100 (SXM)**. Quality is measured as **bits per byte (BPB)** on a fixed FineWeb **validation** split; the metric is **tokenizer-agnostic** (validation uses SentencePiece LUTs for bytes-per-token accounting).

Non-record / unlimited-compute experiments are encouraged under `records/track_non_record_16mb/`.

## Layout

| Path | Role |
|------|------|
| `train_gpt.py` | **CUDA / PyTorch** training: DDP, Muon + AdamW-style groups, bfloat16 autocast, wallclock cap, final int8+zlib artifact + round-trip val check. |
| `train_gpt_mlx.py` | **Apple MLX** port for local iteration on Apple Silicon; mirrors many env-driven hyperparameters; not the official eval stack. |
| `requirements.txt` | Reference deps for CUDA path (RunPod image may preinstall). MLX path uses separate installs per README. |
| `data/` | Dataset download (`cached_challenge_fineweb.py`), optional retokenization (`download_hf_docs_and_tokenize.py`), `tokenizer_specs.json`. See [data/README.md](data/README.md). |
| `data/datasets/fineweb10B_sp1024/` | Expected train/val `*.bin` shards after download (not in git). |
| `data/tokenizers/` | e.g. `fineweb_1024_bpe.model` (not in git until downloaded). |
| `records/track_10min_16mb/<run>/` | **Official leaderboard-style submissions**: `README.md`, `submission.json`, `train.log`, `train_gpt.py` (must run self-contained from that folder). |
| `records/track_non_record_16mb/<run>/` | Same shape; longer compute or exploratory runs. |

## Training entrypoints

### CUDA (`train_gpt.py`)

- Launch with `torchrun`, e.g. `torchrun --standalone --nproc_per_node=8 train_gpt.py` (adjust for GPU count).
- **Hyperparameters** are almost entirely **environment variables** read in the `Hyperparameters` class at the top of the file (data paths, model shape, optimizer LRs, `ITERATIONS`, `MAX_WALLCLOCK_SECONDS` default **600**, `VAL_LOSS_EVERY`, etc.).
- **Wallclock**: training time is tracked excluding validation; when `MAX_WALLCLOCK_SECONDS > 0`, LR warmdown is tied to **remaining wall time** (not only step index).
- **Warmup steps**: optional compile/graph warmup runs then **restores initial weights and optimizer state** so timed training starts clean.
- **End of run**: saves `final_model.pt`, builds **int8-quantized** state, **zlib level 9**, logs `Total submission size int8+zlib`, reloads from disk, runs validation again, logs `final_int8_zlib_roundtrip` / `final_int8_zlib_roundtrip_exact` lines — these are the **submission-relevant** val metrics after quantization.

### MLX (`train_gpt_mlx.py`)

- For Mac / Apple Silicon smoke tests; uses `mlx`, SentencePiece, same shard layout. Extra knobs: `GRAD_ACCUM_STEPS`, `MLX_MAX_MICROBATCH_TOKENS`, `MLX_EAGER_EVAL`.
- Intended as a starting point; parity with CUDA is approximate.

## Data

- Default challenge tokenizer variant in docs: **`sp1024`** (1024-piece SentencePiece vocab).
- Download: `python3 data/cached_challenge_fineweb.py --variant sp1024` (optional `--train-shards N` for smaller local sets).
- Validation is always the **full** published `fineweb_val_*` split (fixed 50k-doc set); training shards are a prefix of a frozen shuffle.

## Submission artifact rules (high level)

- **Counted code** should live in **`train_gpt.py`** (per challenge FAQ in README).
- **Size** = UTF-8 bytes of that code + **compressed** int8 payload (as implemented in `train_gpt.py` — zlib after `torch.save` of quantized dict).
- **Cap**: 16,000,000 bytes total (decimal), not 16 MiB.
- No network/dataset dependency at **eval** time; artifact must be self-contained.

## `submission.json` (records)

Typical fields (see existing runs): `author`, `github_id`, `name`, `blurb`, `date`, `val_loss`, `val_bpb`, `bytes_total`, `bytes_code`. Align with the README submission checklist.

## Conventions for edits

- **`train_gpt.py` and `train_gpt_mlx.py`**: repository policy — keep each **≤ 1500 lines**; they are **starter scripts**, not the home for SOTA configs. Competitive recipes belong under **`records/`** as their own `train_gpt.py` copies.
- **PRs**: small improvements to root trainers are OK; large or niche SOTA work goes in `records/`.
- Match existing style: env-driven `Hyperparameters`, minimal abstraction, heavy reuse of patterns from modded-nanogpt (Muon, etc.). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Quick commands (reference)

```bash
# Data (from repo root)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# CUDA single GPU
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# MLX smoke (example from README)
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

## Glossary

- **BPB / val_bpb**: bits per byte on validation; lower is better; tokenizer-agnostic normalization.
- **GQA**: grouped-query attention (`NUM_KV_HEADS` vs `NUM_HEADS`).
- **Tied embeddings**: `TIE_EMBEDDINGS=1` shares input embedding and output projection where applicable.

---

*This document is maintainer-facing context for tooling and assistants; it does not replace the official challenge rules in README.md.*
