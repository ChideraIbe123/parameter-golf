# Non-Record: SP8192 + MuonEq-R + Loop@0.42 + RECUR_AB + QAT-lite

**Single-seed best local run:** `ttt val_bpb = 1.09664494` | **artifact = 16,067,345 bytes** | **8xH100 SXM**

## Summary

This folder freezes the strongest stable local branch from the April 2026 research cycle:

- `SP8192`
- `MuonEq-R`
- wallclock-aware depth recurrence with `ENABLE_LOOPING_AT=0.42`
- learned recurrent alpha/beta blending (`RECUR_AB`)
- late ramped `QAT-lite`
- legal score-first TTT

This is a **non-record** submission because the artifact is still **67,345 bytes over** the `16,000,000` byte cap, even though the train/eval semantics are rule-compliant and the quality is the best stable branch found before later experimental detours.

## Result

Single seed: `1337`

| Stage | BPB | Notes |
|---|---:|---|
| Raw pre-quant | `1.1018` | best validation before GPTQ |
| Quantized | `1.1243` | GPTQ + Brotli |
| **TTT** | **`1.09664494`** | best stable final score from this branch |

Artifact breakdown:

| Item | Bytes |
|---|---:|
| Quantized model + Brotli | `15,982,868` |
| Code | `84,477` |
| **Total** | **`16,067,345`** |

Timing:

| Phase | Time |
|---|---:|
| Train | `600.020s` |
| Quantized eval | `2.564s` |
| TTT eval | `544.233s` |

## Exact Config

```bash
SEED=1337 \
MUON_EQR=1 \
EMA_DECAY=0 \
ENABLE_LOOPING_AT=0.42 \
RECUR_ALPHA_ENABLED=0 \
RECUR_AB_ENABLED=1 \
RECUR_A_INIT=1.0 \
RECUR_B_INIT=0.0 \
QAT_LITE_ENABLED=1 \
QAT_LITE_START_FRAC=0.55 \
QAT_LITE_EVERY=4 \
QAT_LITE_LAMBDA=0.02 \
QAT_LITE_BITS=6 \
QAT_LITE_CLIP_SIGMAS=12.85 \
QAT_LITE_LAYER_START=7 \
QAT_LITE_TARGETS=qk \
QAT_LITE_PENALTY=mse \
QAT_LITE_DEPTH_POWER=0.0 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Why This Branch

This branch was chosen over later experiments because:

- it beat the plain `Loop@0.42` control
- it beat `RecurAlpha`
- it avoided the large size blowups from broad `HQClip`
- it stayed simpler and more stable than the later `RECUR_LORA` and AWQ-lite attempts

Later research found one higher-quality branch (`HQClip`) at `1.09509057`, but that branch expanded the artifact to about `16.91 MB` and was not considered a practical submission candidate.

## Compliance Notes

Semantics are intended to remain compliant with Issue `#1017`:

- causal left-to-right scoring
- full normalized softmax distribution
- score-before-update TTT ordering
- single left-to-right pass with no rescoring

Reason this is **non-record**:

- artifact exceeds the `16,000,000` byte cap

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
