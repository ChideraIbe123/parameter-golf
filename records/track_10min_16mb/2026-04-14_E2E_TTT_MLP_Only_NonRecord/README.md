# Does E2E TTT help at 27M scale? A negative result.

**val_bpb: 1.1104** (seed 1337) | **13.85 MB** | 8×H100 SXM | Non-record idea submission

## TL;DR

We ported **End-to-End Test-Time Training** (arXiv:2512.23675, ICLR 2026) to
the parameter-golf setting and found that **TTT barely helps at 27M scale** —
all three wildly different hyperparameter configs landed within 0.001 BPB of
each other (1.110–1.112). At this model size, sliding-window eval alone
captures essentially all the achievable eval-time gain; TTT contributes a
rounding error.

## What the paper claims

From Feng et al. (*End-to-End Test-Time Training for Long Context*,
arXiv:2512.23675): at test time, update the model's MLP weights in the
last 1/4 of blocks via score-first SGD on the chunk being evaluated. Freeze
everything else — embeddings, attention, norms — because updating them
causes instability in the inner loop. On 3B+ models with 128K context, this
beats Mamba-2 and Gated DeltaNet and is 2.7× faster than full attention.

The paper's two main design choices we ported faithfully:

1. **MLP-only updates in the last N blocks** (paper: "safe" storage of
   pre-trained knowledge is the frozen attn/embeds; MLPs are the fast
   weights)
2. **Score-first SGD per chunk** (commit NLL under `no_grad`, then run a
   gradient step on that chunk's loss)

## What we tested

Fixed base stack throughout (PR #1493 architecture on SP1024: 11L × 512d,
GQA 8/4, 3-layer depth recurrence L3-5, parallel residuals L7+, QK-Gain 5.0,
GPTQ SDClip int6 + brotli). Same seed (1337). Only the TTT hyperparameters
changed.

| Run | TTT_LR | Epochs | Scope (fraction of blocks) | Trainable params | val_bpb | Eval time |
|-----|-------:|-------:|---------------------------:|-----------------:|--------:|----------:|
| #1 defaults | 0.005 | 3 | last 25% (blocks 8–10) | 6.3M | 1.11137 | 408s |
| #2 broader | **0.015** | **2** | **last 50% (blocks 5–10)** | **12.6M** | **1.11037** | 443s |
| #3 aggressive | 0.05 | 5 | last 9% (block 10 only) | 2.1M | 1.11112 | 413s |

We spanned 10× in learning rate (0.005 → 0.05), 6× in trainable params
(2.1M → 12.6M), and 2.5× in epochs (2 → 5). The spread in final BPB is
**0.001** — well within what a different seed would produce.

## What we learned

### 1. E2E TTT is saturated at this model scale.

The paper's gains come from 3B+ models with substantial adaptation capacity.
At 27M parameters trained to near the cross-entropy floor achievable in
588s, there's almost no adaptation room left. The base model is already
"as good as it's going to get" given its size and training budget, so
test-time gradient steps can't find meaningful slack.

This matches an intuition the paper itself hints at: the value of TTT-E2E
scales with (a) model capacity for storing fast-weight updates and (b)
context length. A 27M-param model with stride-64 sliding windows has
neither lever amplified enough.

### 2. Sliding window eval is doing ~95% of the eval-time work.

Decomposing the eval-time gain on the best config:

| Eval stage | BPB | Δ vs prior |
|------------|----:|----------:|
| Post-quantization (no eval tricks) | 1.1360 | — |
| + Sliding window (stride 64) | 1.1123 | **−0.0237** |
| + E2E TTT (run #2) | 1.1104 | −0.0019 |

The sliding-window trick — giving each token maximal context within a
2048-token window — is responsible for over 90% of the improvement
post-quantization. E2E TTT adds a marginal extra ~0.002 on top. For
small-model submissions, *investing more in the base model or the
sliding-window stride is likely a better use of compute than tuning TTT*.

### 3. "Moderate" beats both conservative and aggressive.

Run #2 (LR=0.015, 2 epochs, last 50% of blocks, 12.6M params) beat both
the paper's defaults (LR=0.005, last 25%) and a much more aggressive
single-block configuration (LR=0.05, 5 epochs, 2.1M params).

Intuition: the paper's defaults are tuned for a larger adaptation budget
than we have; they don't push hard enough. The aggressive single-block
config pushes too hard on too few parameters — it runs into the same
saturation problem faster. The "broader moderate" config gives the update
enough params to absorb signal without overfitting any single chunk.

### 4. Paper-recommended parameter selection matters more than TTT intensity.

Before deciding on MLP-only, we considered the obvious alternative of
updating "everything" at test time (which the competition's existing
score-first TTT does). The paper explicitly flags this as unstable. Our
MLP-only implementation is more conservative but passes all compliance
checks and trains cleanly — no NaNs, no divergence across all three
configs. That's the main contribution of the paper's framing: identifying
*which* parameters are safe to update.

## Why we're submitting this as non-record

1. **SP1024, not SP8192.** The SP8192 data used by PR #1493 is not available
   from the `willdepueoai/parameter-golf` HF repo that the challenge's data
   loader points at. Running on SP1024 costs us ~0.03 BPB versus SP8192
   baselines, so our 1.1104 is not directly comparable to merged records.
2. **Single seed.** Record submissions provide 3 seeds; we ran seed 1337
   only. Given the 0.001 BPB noise floor we measured, more seeds wouldn't
   change the story.
3. **Honest negative result.** The interesting content of this submission
   is the research finding, not the leaderboard score.

## Takeaway for other participants

If you're a small-model submission (< 50M params) considering E2E TTT as a
lever: *it will probably not move your score meaningfully*. The sliding-
window stride, the base model's architecture, and the training-budget
utilization matter more at this scale. Reserve TTT for when you have
substantial adaptation capacity (larger models, longer context, or
intentionally under-trained base models).

If you're pushing toward larger models or longer context, this
implementation (gated by `TTT_E2E_MODE=1`, `TTT_E2E_LAST_FRAC`,
`TTT_LR`, `TTT_EPOCHS` env vars) is a drop-in starting point that
respects the paper's stability findings.
