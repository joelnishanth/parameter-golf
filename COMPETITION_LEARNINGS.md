# Parameter Golf — Competition Intelligence

> Last updated: 2026-03-20 night (scan #3: PR #374 is the new leader at 1.1246 BPB). Re-scan PRs regularly as the frontier moves daily.

---

## Current state of play

| Source | BPB | Status |
|--------|-----|--------|
| **PR #374 — 11L + Tight SWA + Shared VE128 + Partial RoPE + LN Scale + XSA4** | **1.1246** | **NEW LEADER** |
| PR #369 — 11L NTK-RoPE + FA3 + Batch524K + XSA4 + EMA | 1.1328 | Record |
| PR #254 — PR#198 stack + TTT (full-weight SGD on val) | 1.1303 | TTT legality debated |
| PR #265 — PR#198 stack + XSA on last 3 layers | 1.1307 | Clean arch improvement |
| PR #198 — 11L + BigramHash + SmearGate + OrthoInit + WD=0.04 + SWA + FA3 | 1.1318 | Former frontier |
| PR #376 — 9L MLP3x Full Stack + Custom Kernel | 1.1401 | Record |
| PR #349 — 11L XSA + EMA + Int5-MLP | 1.1399 | Record |
| PR #180 — 10L Int5-MLP + BigramHash + SWA + WD=0.04 | 1.1428 | Proven |
| PR #194 — 11L Int6 QAT + Per-Dim SmearGate + SWA | 1.1453 | Proven |

**Target to win**: beat PR #374 by ≥0.005 → need ≤ **1.1196 BPB**.

> **Note**: Frontier progression: 1.158 → 1.1318 → 1.1303 → **1.1246** (Mar 20-21). Key new techniques in #374: Tight SWA (scale<0.2), Shared Value Embedding, Partial RoPE (16/64 dims), LN Scale 1/sqrt(layer+1), Late QAT.

---

## PR #374 — The New Leader (1.1246 BPB) — FULL ANATOMY

### Key innovations over PR #198 stack

1. **Tight SWA**: Collect weight averages only when LR warmdown scale < 0.2 (last ~600 steps), every 50 steps. Result: zero SWA penalty (post-SWA BPB = pre-SWA BPB).
2. **Shared Value Embedding (VE)**: One embedding table (vocab→128→kv_dim) re-injects token identity into attention values on layers 9,10. Per-layer learned scales. Gives deeper layers better token discrimination.
3. **Partial RoPE (16/64 dims)**: Only rotate 16 of 64 head dims. Concentrates positional signal, frees 48 dims for content. NTK scaling uses rope_dims in exponent.
4. **LN Scale Factor**: Multiply RMSNorm output by 1/sqrt(layer_idx+1) before attention and MLP. Stabilizes deep layer gradients.
5. **Late QAT (threshold=0.1)**: QAT disabled by default, auto-enables when LR scale drops below 0.1 (~last 10% of warmdown). Minimal training disruption.
6. **XSA on last 4 layers** (not 3): More aggressive self-attention bias removal.
7. **Gradient clip 0.3**: Tighter than standard 1.0.

### Exact run command
```bash
NUM_LAYERS=11 MLP_MULT=3.0 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
SWA_ENABLED=1 SWA_EVERY=50 LATE_QAT_THRESHOLD=0.1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 ADAM_WD=0.04 MUON_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
torchrun --nproc_per_node=8 train_gpt.py
```

### Results
- 6942 steps in 600s at 86.4ms/step
- Pre-quant val_bpb: 1.1407
- Post-SWA val_bpb: 1.1407 (zero SWA penalty)
- Quant gap: 0.008
- **Sliding window val_bpb: 1.1246**
- Artifact: 15,706,024 bytes

---

## PR #198 — The New Frontier (1.1318 BPB) — FULL ANATOMY

### Architecture

```
11 layers × 512 dim
8 heads, 4 KV heads (GQA), head_dim=64
MLP: relu^2, hidden=1536 (MLP_MULT=3)
Vocab: 1024 (SentencePiece BPE)
Tied embeddings: yes
Skip connections: encoder/decoder U-Net style (5 encoder, 6 decoder, 5 skip weights)
RoPE positional encoding (base=10000), RMSNorm
Logit softcap: 30.0
SmearGate: per-dim (512 params)
BigramHash: 2048 buckets, dim=128 projected to 512, learned scale=0.05
Total params: 26,829,913
Artifact size: 15,689,380 bytes (15.7 MB)
```

### Training Config (exact env vars from PR #198 run command)

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Key metrics

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1432 |
| Int6 roundtrip val_bpb (no sliding) | 1.1543 |
| **Int6 sliding val_bpb (stride=64)** | **1.1318** |
| Steps completed (600s cap) | 7,412 |
| Step time | 81 ms |
| 3-seed mean | **1.1326** (seeds: 1.1318, 1.1335, 1.1324) |

### What PR #198 does that our ConsensusStack doesn't

| Feature | Our ConsensusStack | PR #198 | Delta estimate |
|---------|-------------------|---------|---------------|
| Layers | 9 | **11** | ~0.010 BPB |
| SmearGate | No | **Yes (per-dim)** | ~0.003–0.005 BPB |
| BigramHash | No | **Yes (2048 buckets)** | ~0.004–0.008 BPB |
| OrthoInit + muP | No | **Yes** | ~0.002–0.003 BPB |
| WD | 0.02 | **0.04** | ~0.002 BPB |
| SWA | No | **Yes (every 200 steps)** | ~0.003–0.005 BPB |
| FA3 | No | **Yes** | ~0.003 BPB (more steps) |
| QAT | No | No | — |
| Batch tokens | 786K | **786K** | — |
| MATRIX_LR | 0.02 | **0.025** | — |

---

## Code Recipes — Exact implementations from top PRs

### SmearGate (from PR #198)

Per-dimension learned gate blending each token's embedding with the previous token's. Zero-initialized (starts as identity), 512 params total.

```python
class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
```

**Where to wire it**: Apply to `x` in `GPT.forward()` right after token embedding, before the first block. Also apply in `GPT.forward_logits()` for eval consistency.

**Optimizer**: Use scalar LR (0.025) with Adam, betas=(0.9, 0.99).

### BigramHash Embedding (from PR #198)

Hash-based bigram features that encode the (prev_token, curr_token) pair into a learned embedding. Zero-initialized so it starts as a no-op.

```python
class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
```

**PR #198 config**: `BIGRAM_VOCAB_SIZE=2048`, `BIGRAM_DIM=128` (projected to 512). 2048 buckets saves ~300KB vs the default 4096.

**Where to wire it**: Add `bigram_embed(token_ids)` to the token embedding output in `GPT.forward()`.

**Artifact cost**: ~524K params × int6 ≈ ~400KB compressed. Fits easily under 16MB.

### OrthoInit + muP Scaling (from PR #198)

Orthogonal initialization for all 2D linear weights ≥64×64. Output projections are scaled by 1/√(2·num_layers) for muP-style depth scaling.

```python
def _init_weights(self) -> None:
    if self.tie_embeddings:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
    num_layers = len(self.blocks)
    for name, module in self.named_modules():
        if isinstance(module, nn.Linear):
            if getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
            elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                nn.init.orthogonal_(module.weight, gain=1.0)
                if ".proj." in name or name.endswith(".proj"):
                    with torch.no_grad():
                        module.weight.mul_(1.0 / math.sqrt(2 * num_layers))
```

### SWA — Stochastic Weight Averaging (from PR #198)

Checkpoints collected every N steps during warmdown (when LR scale < 0.5), then averaged before final eval.

```python
# State init
swa_state: dict[str, Tensor] | None = None
swa_count = 0

# During training loop (when scale < 0.5)
if args.swa_enabled and scale < 0.5 and step % args.swa_every == 0:
    if swa_state is None:
        swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
        swa_count = 1
    else:
        for name, t in base_model.state_dict().items():
            swa_state[name] += t.detach().cpu()
        swa_count += 1

# After training loop ends
if swa_state is not None and swa_count > 1:
    avg_state = {name: (t / swa_count).to(dtype=base_model.state_dict()[name].dtype)
                 for name, t in swa_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
```

**PR #198 config**: `SWA_ENABLED=1`, `SWA_EVERY=200`. Gets ~8 checkpoints averaged over warmdown. PR #194 uses `SWA_EVERY=50` → ~30 checkpoints.

### Int6 Quantization — Per-row (from PR #198)

```python
def quantize_int6_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1.0 / 31.0).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -32, 31).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / 31.0 if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -32, 31).to(torch.int8)
    return q, scale
```

**Mixed quantization**: `mixed_quantize_int6(state_dict, {"mlp", "attn"})` applies int6 to MLP + attention weights, int8 to embeddings (via fp16 passthrough for tok_emb), fp32 passthrough for control tensors (scales, gates, smear params).

### Int6 STE QAT (from PR #194 — NOT used in PR #198)

```python
def fake_quantize_int6_per_row(w: Tensor) -> Tensor:
    scale = w.detach().abs().amax(dim=-1, keepdim=True).div_(31.0).clamp_(min=1.0 / 31.0)
    w_deq = (w / scale).round().clamp_(-31, 31) * scale
    return w + (w_deq - w).detach()

class CastedLinear(nn.Linear):
    _qat: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self._qat and self.training:
            w = fake_quantize_int6_per_row(w)
        return F.linear(x, w.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)
```

**Note**: PR #198 does NOT use QAT and still gets 1.1318. Its post-quant gap is 0.011 BPB (1.1432 → 1.1543). PR #194 uses QAT with gap 0.021 BPB (1.1666 → 1.1453 post-quant+sliding). QAT helps reduce post-quant gap but PR #198 compensates with OrthoInit + WD=0.04 which make weights more quant-friendly.

### Flash Attention 3 (from PR #198)

```python
from flash_attn_interface import flash_attn_func as flash_attn_3_func

# In CausalSelfAttention.forward():
q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
q = F.rms_norm(q, (q.size(-1),))
k = F.rms_norm(k, (k.size(-1),))
cos, sin = self.rotary(seqlen, x.device, q.dtype)
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)
q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
y = flash_attn_3_func(q, k, v, causal=True)
```

**Requirement**: `pip install flash-attn` (FA3 compatible build). `flash_attn_interface` is the FA3 module. H100 SXM required.

**Impact**: Reduces step time from ~81ms to significantly less than PyTorch SDPA, allowing more steps in 600s.

### Muon with Decoupled Weight Decay (from PR #194)

```python
# Inside Muon.step():
weight_decay = group.get("weight_decay", 0.0)
curr = 0
for p in params:
    g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
    if weight_decay > 0:
        p.mul_(1 - lr * weight_decay)
    p.add_(g, alpha=-lr)
    curr += p.numel()
```

**PR #198**: `MUON_WD=0.04`, `ADAM_WD=0.04` (same for both optimizers).
**PR #194**: `MUON_WEIGHT_DECAY=0.038`.

---

## Head-to-head: PR #198 vs PR #194

Understanding why #198 beats #194 by 0.016 BPB despite #194 having QAT:

| Feature | PR #198 (1.1318) | PR #194 (1.1480) |
|---------|------------------|------------------|
| Layers | 11 | 11 |
| BigramHash | **Yes (2048 buckets)** | No |
| SmearGate | Yes (per-dim) | Yes (per-dim) |
| OrthoInit + muP | **Yes** | Not mentioned |
| QAT | No | **Yes (int6 STE)** |
| SWA every | 200 (~8 ckpts) | 200 (~30 ckpts) |
| MATRIX_LR | **0.025** | 0.02 |
| TIED_EMBED_LR | **0.035** | 0.03 |
| Batch tokens | **786K** | 524K |
| Step time | 81ms | 74ms |
| Steps in 600s | 7,412 | 8,052 |
| Params | 26.8M | 26.5M |
| Artifact | 15.7 MB | 15.3 MB |

**Key takeaway**: BigramHash + OrthoInit + larger batch + higher LR > QAT. The BigramHash alone is likely worth ~0.008–0.012 BPB. OrthoInit + muP scaling improves init quality.

---

## NEW: Exclusive Self Attention / XSA (from PR #265 — 1.1307 BPB)

XSA removes self-attention bias from deep layers by subtracting the self-value projection from the attention output. Only applied to the last N layers where self-attention bias is highest (per arXiv:2603.09078).

**Cost**: ~2ms/step overhead for 3 layers. **Benefit**: ~0.001–0.002 BPB.

```python
# Inside CausalSelfAttention:
def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
    B, T, H, D = y.shape
    Hkv = v.size(-2)
    group = H // Hkv
    y_g = y.reshape(B, T, Hkv, group, D)
    vn = F.normalize(v, dim=-1).unsqueeze(-2)
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
    return (y_g - proj).reshape(B, T, H, D)

def forward(self, x: Tensor) -> Tensor:
    # ... q, k, v, rotary, flash_attn as usual ...
    y = flash_attn_3_func(q, k, v, causal=True)
    if self.use_xsa:
        y = self._xsa_efficient(y, v)
    y = y.reshape(bsz, seqlen, dim)
    return self.proj(y)
```

**Wiring**: In `GPT.__init__`, enable on last N layers:
```python
# Hyperparameter: xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
if xsa_last_n > 0:
    for i in range(max(0, num_layers - xsa_last_n), num_layers):
        self.blocks[i].attn.use_xsa = True
```

**PR #265 config**: `XSA_LAST_N=3` (layers 8, 9, 10 of 11). Default in code is 4. Uses `SWA_EVERY=120` (13 checkpoints). Otherwise identical to PR #198 stack.

---

## NEW: Test-Time Training / TTT (from PR #254 — 1.1303 BPB)

Full-weight SGD on the validation data before scoring. Adapts the model to the test distribution, then uses adapted weights for sliding window eval. **Legality debated** (PR #267 flags this).

**Cost**: ~43s on 8xH100 (total eval ~129s, well under 600s). **Benefit**: ~0.002 BPB on top of #198 stack.

```python
def ttt_adapt(args, base_model, device, val_tokens, rank=0, world_size=1, log_fn=None):
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_seqs = args.ttt_batch_seqs

    # Freeze early blocks for stability
    if args.ttt_freeze_blocks > 0:
        for i, block in enumerate(base_model.blocks):
            if i < args.ttt_freeze_blocks:
                for p in block.parameters():
                    p.requires_grad_(False)

    ttt_params = [p for p in base_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)

    my_start = (total_seqs * rank) // world_size
    my_end = (total_seqs * (rank + 1)) // world_size

    base_model.train()
    for epoch in range(args.ttt_epochs):
        for batch_start in range(my_start, my_end, batch_seqs):
            batch_end = min(batch_start + batch_seqs, my_end)
            local = val_tokens[batch_start*seq_len : batch_end*seq_len+1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model(x, y)
            loss.backward()
            if world_size > 1:
                for p in ttt_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
            optimizer.step()

    for p in base_model.parameters():
        p.requires_grad_(True)
```

**TTT config**: `TTT_ENABLED=1`, `TTT_LR=0.002`, `TTT_EPOCHS=3`, `TTT_MOMENTUM=0.9`, `TTT_FREEZE_BLOCKS=2`, `TTT_BATCH_SEQS=32`.

**Pipeline**: TTT runs AFTER loading int6 roundtrip model, BEFORE torch.compile and sliding window eval.

---

## NEW: NTK-RoPE (from PR #265 and #254)

Auto-scales RoPE base frequency when eval seq_len > train seq_len. Enables seq2048 eval with a model trained at seq1024.

```python
class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        if seq_len > self.train_seq_len:
            scale = seq_len / self.train_seq_len
            new_base = self.base * (scale ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (new_base ** (torch.arange(0, self.dim, 2, device=device) / self.dim))
        else:
            inv_freq = self.inv_freq.to(device)
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        return cos.to(dtype), sin.to(dtype)
```

---

## Our plan to beat ≤1.1257 BPB

### Step 1: Match PR #198 + XSA + QAT (expected ~1.124–1.128 BPB)

Nobody has combined all three. Our stack:
- Full PR #198 base: 11L, SmearGate, BigramHash(2048), OrthoInit, WD=0.04, SWA, FA3, seq2048
- **+ XSA on last 3–4 layers** (from PR #265): ~0.001–0.002 BPB
- **+ Int6 STE QAT** (from PR #194): closes the 0.011 BPB post-quant gap by ~0.005–0.008
- **+ NTK-RoPE**: better long-context generalization at eval

Expected: PR #198 base (1.132) - QAT gain (0.005) - XSA gain (0.001) ≈ **~1.126 BPB**

### Step 2: Add TTT if legal (expected ~0.002 more)

If TTT is ruled legal, add it for another ~0.002 BPB → **~1.124 BPB**.

### Step 3: Exploration bets (each ~0.002–0.005 BPB)

**Bet A: Int5 MLP + 12L** (from PR #180)
- Int5 for MLP weights saves ~1.86MB → fund a 12th layer
- 12L + int5 MLP + full stack is unexplored

**Bet B: SWA every 50–120 steps** (from PR #194 / #265)
- PR #265 uses SWA every 120 (~13 checkpoints). PR #194 uses every 50 (~30 checkpoints)
- More frequent averaging → smoother weights → better post-quant

**Bet C: Stride 32 eval** (from PR #222)
- Cut stride from 64 to 32. ~2× slower eval but still under 10 min
- Small but free BPB improvement

**Bet D: XSA_LAST_N tuning**
- PR #265 uses 3 layers. Default in code is 4. Worth sweeping 2–5.

---

## Things that have been tried and FAILED

- **SwiGLU activation**: better per-step quality but 45% slower → fewer steps in 10 min → net negative (PR #163: 1.2091)
- **Depth recurrence at 10 min**: needs far more steps to converge; promising for unlimited compute track
- **lzma compression**: worse than zlib for int8 weight data (confirmed by official SOTA author)
- **Higher embed LR** (0.08 vs 0.05): hurts convergence
- **Late-stage QAT** (applied only in last N steps): not enough training signal; overhead not worth it
- **Larger vocab (8192)** so far: PR #78 (1.186), PR #92 (1.194) — competitive but not near frontier
- **8192 vocab + NorMuon + selective quant** (PR #78): 1.186 BPB — decent but behind int6 stack
- **SWA + doc-isolated eval** (PR #199 non-record): two negative findings at stride=64 — SWA alone didn't help without the other pieces
- **int8 QAT overhead** (PR #145 non-record): int8 QAT overhead exceeds quantization gap recovery — int6 STE is the right approach
- **SP4096 tokenizer** (PR #200: 1.2012, PR #217: 1.1753) — larger vocab + 4096 pieces underperforms the sp1024 stack significantly

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

## Architecture quick-reference (PR #198 frontier config)

```
11 layers × 512 dim
8 heads, 4 KV heads (GQA), head_dim=64
MLP: relu^2, hidden=1536 (MLP_MULT=3)
Vocab: 1024 (SentencePiece BPE)
Tied embeddings: yes (init_std=0.005)
Skip connections: encoder/decoder U-Net style
RoPE (base=10000), RMSNorm
Logit softcap: 30.0
SmearGate: per-dim (512 params)
BigramHash: 2048 buckets, dim=128→512, scale=0.05
Seq len: 2048 train, 2048 eval
Batch: 786K tokens
```

---

## File conventions

- All competitive work goes in `records/track_10min_16mb/<date>_<name>/`
- Each record folder must contain: `train_gpt.py`, `README.md`, `submission.json`, at least one `train.log`
- The `train_gpt.py` in the record folder must run self-contained from that directory
- Root `train_gpt.py` stays ≤1500 lines and is not the competition config
