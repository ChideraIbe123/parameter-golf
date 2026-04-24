# Research Notes

## Goal

Track hypotheses, papers, runs, and conclusions while chasing a better `track_10min_16mb` result.

Current public merged record referenced during this work:

- `1.0810 BPB`
- record: `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`

## Working Baseline

Best stable baseline found in this round of testing:

- SP8192 data/tokenizer
- EMA disabled: `EMA_DECAY=0`
- baseline quantization: `MATRIX_BITS=6`, `EMBED_BITS=8`, `QK_BITS=6`, `QK_CLIP_SIGMAS=12.85`
- score-first TTT still enabled

Representative baseline result:

- raw pre-quant `val_bpb`: about `1.1125`
- quantized `val_bpb`: about `1.1320`
- TTT `val_bpb`: about `1.10745691`

Main takeaway:

- raw model quality is decent
- most damage happens after quantization

## Infrastructure Findings

### SP8192 data access

The default downloader path was not enough for SP8192 on the pod. The working source was:

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
```

Relevant repo changes already made:

- `0d14850` Add sp8192 fallback support to FineWeb downloader

### EMA

EMA was actively harmful for this branch.

Evidence:

- with EMA, post-EMA pre-quant and quantized quality collapsed badly
- removing EMA improved final TTT substantially

Conclusion:

- keep `EMA_DECAY=0` unless a future branch proves otherwise

## Novel Technique Experiments

### 1. OSP-lite

Motivation:

- inspired by outlier-safe pretraining / quantization-robust training
- aimed to make the model easier to quantize during training

Implementation attempts:

- embedding bottleneck / projection path
- earlier version also scalarized residual controls

Result:

- harsh OSP-lite clearly hurt training
- softened OSP-lite roughly tied raw pre-quant quality
- but quantized and TTT results got worse than baseline

Representative softened run:

- raw pre-quant `val_bpb`: `1.1126`
- quantized `val_bpb`: `1.1437`
- TTT `val_bpb`: `1.11221097`

Conclusion:

- not a winning direction for this stack
- remove from default path

Relevant commits:

- `eeb2978` Add OSP-lite embedding projection and scalar residuals
- `d71036b` Soften OSP-lite to embedding bottleneck only
- `84b3b18` Fix GPTQ Hessians for embedding projection
- `3a74ac2` Revert OSP-lite and protect QK GPTQ

### 2. Q/K mixed-precision GPTQ

Motivation:

- Q/K looked more sensitive than average matrices
- idea: protect them with higher precision

Implementation:

- special quantization category for `attn.c_q.weight` and `attn.c_k.weight`
- env knobs:
  - `QK_BITS`
  - `QK_CLIP_SIGMAS`

Result with `QK_BITS=8`:

- raw pre-quant `val_bpb`: `1.1124`
- quantized `val_bpb`: `1.1675`
- TTT `val_bpb`: `1.10761032`
- total size: `16747259`

Conclusion:

- too expensive in bytes
- did not beat baseline final score
- keep the hookable code path, but this exact setting is not viable

Relevant commit:

- `3a74ac2` Revert OSP-lite and protect QK GPTQ

### 3. LR-QAT-inspired QAT-lite

Paper inspiration:

- low-rank / lightweight quantization-aware training
- adapted into a selective fake-quant regularizer instead of full QAT

Implementation:

- late-stage penalty on selected attention weights
- targets:
  - `blocks.*.attn.c_q.weight`
  - `blocks.*.attn.c_k.weight`
  - optional `attn.proj`

Core knobs:

- `QAT_LITE_ENABLED`
- `QAT_LITE_START_FRAC`
- `QAT_LITE_EVERY`
- `QAT_LITE_LAMBDA`
- `QAT_LITE_BITS`
- `QAT_LITE_CLIP_SIGMAS`
- `QAT_LITE_LAYER_START`
- `QAT_LITE_INCLUDE_PROJ`

Original stronger config:

```bash
QAT_LITE_ENABLED=1
QAT_LITE_START_FRAC=0.55
QAT_LITE_EVERY=4
QAT_LITE_LAMBDA=0.02
QAT_LITE_BITS=6
QAT_LITE_CLIP_SIGMAS=12.85
QAT_LITE_LAYER_START=7
QAT_LITE_INCLUDE_PROJ=0
```

Most promising run seen:

- raw pre-quant `val_bpb`: `1.1123`
- quantized `val_bpb`: `1.1472`
- TTT `val_bpb`: `1.10731739`
- total size: `16751540`

Interpretation:

- final TTT was the best seen from a novel training idea in this round
- but the run was over the size cap and still not enough to challenge SOTA

Tuned weaker config tested:

- later start / weaker lambda / later layers

Representative tuned result:

- raw pre-quant `val_bpb`: `1.1125`
- quantized `val_bpb`: `1.1372`
- TTT `val_bpb`: `1.10776585`
- total size: `16014204`

Conclusion:

- QAT-lite is the most promising novel pretraining-side direction tested so far
- but current tuning is unstable / weak
- original stronger setup looked better than the softened tuned version

Relevant commit:

- `0510575` Add LR-QAT-inspired fake-quant regularizer

### 4. QACT-lite activation-tail regularization

Motivation:

- weight-only QAT-lite may miss the true problem if quantization damage is driven by activations
- idea: regularize heavy tails in late-layer Q/K activations

Implementation:

- forward hooks on late-layer `attn.c_q` / `attn.c_k`
- penalty on:

```text
relu(|activation| - tau * sigma)^2
```

Core knobs:

- `QACT_LITE_ENABLED`
- `QACT_LITE_START_FRAC`
- `QACT_LITE_EVERY`
- `QACT_LITE_LAMBDA`
- `QACT_LITE_LAYER_START`
- `QACT_LITE_TAU`

Representative run:

- raw pre-quant `val_bpb`: `1.1126`
- quantized `val_bpb`: `1.1425`
- total size: `16012257`

Conclusion:

- QACT-lite alone does not look better than baseline
- likely worse than the stronger QAT-lite attempt

Relevant commit:

- `fbc4c08` Add Q/K activation-tail regularizer

## Paper Notes

### Useful / relevant

- Outlier-safe pretraining papers:
  - relevant conceptually
  - did not transfer well to this exact stack via OSP-lite

- lightweight / low-rank QAT papers:
  - most directly relevant to the observed bottleneck
  - led to the QAT-lite experiment

- activation- or outlier-oriented quantization robustness papers:
  - motivated QACT-lite
  - so far not enough by themselves

### Probably not useful here

- tensor parallelism / reduced synchronization papers

Why not:

- this challenge bottleneck is not mainly multi-GPU communication
- measured step times were already stable
- quality after quantization is the dominant issue, not distributed throughput

## Summary of Conclusions

What seems true right now:

1. The branch is limited more by quantization damage than by raw training quality.
2. EMA should stay off.
3. OSP-lite was not helpful for this stack.
4. Q/K-specific higher-precision GPTQ was too expensive in bytes.
5. QAT-lite is the best novel pretraining-side direction tested so far, but not yet strong enough.
6. QACT-lite alone is not better than baseline.

## Recommended Next Steps

If continuing pretraining-side novelty:

1. return to the stronger original QAT-lite recipe
2. compare QAT-lite plus a very small QACT-lite term
3. keep baseline quantization settings fixed while tuning training-side novelty

If optimizing for highest practical win probability instead:

1. recover the exact public SP8192 record recipe cleanly
2. benchmark from there
3. only test one narrow delta at a time
