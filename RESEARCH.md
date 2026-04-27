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

### MuonEq-R gap

Important discovery from comparing the current branch to the public SP8192 record:

- the public record explicitly uses **MuonEq-R**
- MuonEq-R is described in the record docs as **row-normalized Muon before Newton-Schulz**
- the current branch had drifted back to plain Muon, i.e. it was missing that row-normalization step

Why this matters:

- the public record treats MuonEq-R as a core stack component, not a minor extra
- earlier record notes attribute about `~0.001 BPB` improvement to MuonEq-R by itself
- because our pre-TTT gap to the public record is huge, recovering missing record ingredients is higher-value than piling on more speculative regularizers

Result after restoring MuonEq-R on the current QAT-lite branch:

- raw pre-quant `val_bpb`: `1.1085`
- quantized `val_bpb`: `1.1288`
- TTT `val_bpb`: `1.10276340`
- total size: `16056474`

Interpretation:

- by far the biggest improvement seen in this research round
- improved best final TTT by about `0.00426 BPB`
- confirms MuonEq-R was a materially missing record ingredient
- still over the size cap, but now the branch is much more competitive on quality

### Fraction-of-iterations scheduling bug

Important follow-up discovery:

- several training features were gated by `step / iterations`
- the branch runs with `iterations=20000`, but wallclock cap stops training at only about `4860` steps
- therefore any feature starting at fractions like `0.35` or `0.55` of `iterations` never actually activated under the 600s cap

Affected features:

- depth recurrence activation (`enable_looping_at=0.35`)
- late-start EMA (`EMA_START_FRAC`)
- QAT-lite start (`QAT_LITE_START_FRAC`)

Consequence:

- several earlier experiments were partially or completely placebo with respect to the intended late-stage behavior

Fix applied:

- when `MAX_WALLCLOCK_SECONDS` is active, fraction-gated features now use wallclock progress as well as nominal iteration progress

### Wallclock-aware late-stage run

First run after fixing wallclock-aware scheduling:

- `MUON_EQR=1`
- `EMA_DECAY=0.9965`
- `EMA_START_FRAC=0.35`
- ramped QAT-lite enabled

What actually happened:

- depth recurrence finally activated for real:
  - `layer_loop:enabled step:1703 frac:0.350`
- step time jumped from about `123 ms` to about `155 ms`
- training stopped much earlier, at only about `3880` steps, because recurrence now consumed real wallclock budget

Pre-EMA raw validation right before stopping:

- raw pre-EMA `val_bpb`: `1.1042`

That is notable because:

- it is much better than the earlier MuonEq-R + QAT-lite run without real loop activation (`1.1085`)
- it strongly suggests that the recurrence schedule was one of the major missing pieces

However, EMA still failed badly on top of that stronger branch:

- post-EMA pre-quant `val_bpb`: `1.2211`
- quantized `val_bpb`: `1.2320`
- TTT `val_bpb`: `1.11873231`

Interpretation:

- the scheduling fix was real and important
- the recurrence activation looks promising
- EMA remains actively harmful, even after restoring MuonEq-R
- the next experiment should isolate **wallclock-aware recurrence without EMA**

### Wallclock-aware recurrence without EMA

Follow-up run with the same wallclock-aware recurrence activation, but with `EMA_DECAY=0`:

- `MUON_EQR=1`
- `EMA_DECAY=0`
- ramped QAT-lite enabled

Observed behavior:

- `layer_loop:enabled step:1703 frac:0.350`
- step time again jumped from about `123 ms` to about `154 ms`
- training again ended much earlier, at about `3886` steps, because the recurrence now consumed real wallclock budget

Result:

- raw pre-quant `val_bpb`: `1.1045`
- quantized `val_bpb`: `1.1320`
- TTT `val_bpb`: `1.09968653`
- total size: `16061968`

Interpretation:

- this is the best final TTT result seen so far in this research round
- it beats the previous best completed branch (`1.10276340`) by about `0.00308 BPB`
- the key gain came from real recurrence activation, not from EMA
- quantized quality stayed roughly flat, while TTT improved sharply
- the branch is still over the size cap, but the modeling side improved substantially

New implication:

- recurrence scheduling is now a first-class tuning knob
- because loop activation slows steps so much, `enable_looping_at=0.35` may not be optimal under a strict 600s wallclock cap
- a slightly later recurrence start could plausibly preserve most of the quality gain while allowing more total training steps

### Recurrence schedule tuned later: `ENABLE_LOOPING_AT=0.40`

Follow-up run with the same core recipe, but delaying recurrence slightly:

- `MUON_EQR=1`
- `EMA_DECAY=0`
- `ENABLE_LOOPING_AT=0.40`
- ramped QAT-lite enabled

Observed behavior:

- `layer_loop:enabled step:1934 frac:0.400`
- pre-loop step time stayed around `124 ms`
- post-loop step time climbed more gently than the `0.35` run
- training reached about `3934` steps instead of about `3886`

Result:

- raw pre-quant `val_bpb`: `1.1042`
- quantized `val_bpb`: `1.1223`
- TTT `val_bpb`: `1.09874205`
- total size: `16061958`

Interpretation:

- this is the new best final TTT result seen so far in this research round
- it beats the earlier `ENABLE_LOOPING_AT=0.35` recurrence run (`1.09968653`) by about `0.00094 BPB`
- it also improves quantized quality materially (`1.1320 -> 1.1223`)
- delaying recurrence slightly appears to preserve the recurrence benefit while recovering some useful extra training budget

New implication:

- the recurrence start fraction is a real optimization lever, not just a binary on/off switch
- the branch now looks strong enough that a small local sweep around `0.40` is higher-value than most new architectural ideas

### Recurrence schedule tuned later again: `ENABLE_LOOPING_AT=0.42`

Follow-up run with recurrence delayed a bit further:

- `MUON_EQR=1`
- `EMA_DECAY=0`
- `ENABLE_LOOPING_AT=0.42`
- ramped QAT-lite enabled

Result:

- raw pre-quant `val_bpb`: `1.1034`
- quantized `val_bpb`: `1.1288`
- TTT `val_bpb`: `1.09845764`
- total size: `16063475`

Interpretation:

- this beat the earlier `ENABLE_LOOPING_AT=0.40` control (`1.09874205`) by about `0.00028 BPB`
- raw pre-quant improved slightly
- quantized BPB was a bit worse than the `0.40` run, but final TTT was better
- this became the best control branch before recurrent-alpha gating

### Recurrent-alpha gates on top of `ENABLE_LOOPING_AT=0.42`

First run with learned recurrent-alpha gates enabled:

- `MUON_EQR=1`
- `EMA_DECAY=0`
- `ENABLE_LOOPING_AT=0.42`
- `RECUR_ALPHA_ENABLED=1`
- `RECUR_ALPHA_INIT=1.0`
- ramped QAT-lite enabled

Result:

- raw pre-quant `val_bpb`: `1.1036`
- quantized `val_bpb`: `1.1222`
- TTT `val_bpb`: `1.09808318`
- total size: `16062189`

Interpretation:

- this did **not** produce a big jump, but it did improve the best control branch
- gain vs the `0.42` control: about `0.00037 BPB`
- quantized BPB improved materially (`1.1288 -> 1.1222`)
- raw pre-quant moved slightly in the wrong direction, but final TTT still improved
- the technique looks alive, just not transformative on its first shot

New implication:

- recurrent-alpha gating is likely additive with the recurrence schedule, but only weakly so far
- it is worth one or two narrow follow-up ablations, not a large blind tuning tree

### Recurrent-alpha init tuned down: `RECUR_ALPHA_INIT=0.85`

Follow-up run with a slightly weaker recurrent-alpha initialization:

- `MUON_EQR=1`
- `EMA_DECAY=0`
- `ENABLE_LOOPING_AT=0.42`
- `RECUR_ALPHA_ENABLED=1`
- `RECUR_ALPHA_INIT=0.85`
- ramped QAT-lite enabled

Result:

- raw pre-quant `val_bpb`: `1.1036`
- quantized `val_bpb`: `1.1240`
- TTT `val_bpb`: `1.09845381`
- total size: `16067358`

Interpretation:

- this was effectively a wash versus the `0.42` control and worse than `RECUR_ALPHA_INIT=1.0`
- the weaker init gave back most of the benefit from recurrent-alpha gating
- this suggests the simpler `RECUR_ALPHA_INIT=1.0` setting is the stronger alpha-only version

### Low-rank GPTQ residual correction (`QRES`) on top of `ENABLE_LOOPING_AT=0.42`

Follow-up run testing a post-quant low-rank residual correction on late-layer `q/k` weights:

- `QRES_ENABLED=1`
- `QRES_RANK=1`
- `QRES_LAYER_START=7`
- `QRES_TARGETS=qk`

Result:

- raw pre-quant `val_bpb`: `1.1045`
- quantized `val_bpb`: `1.1227`
- TTT `val_bpb`: `1.09896577`
- total size: `16080150`

Interpretation:

- this was a miss
- it regressed the best completed branch despite adding more bytes
- the remaining gap does not appear to be best attacked by storing low-rank post-quant residuals in this form

### Recurrent alpha/beta memory blend on top of `ENABLE_LOOPING_AT=0.42`

Follow-up run replacing simple recurrent-alpha gating with a learned alpha/beta blend against the cached output from the previous visit of the same repeated layer:

- `MUON_EQR=1`
- `EMA_DECAY=0`
- `ENABLE_LOOPING_AT=0.42`
- `RECUR_ALPHA_ENABLED=0`
- `RECUR_AB_ENABLED=1`
- `RECUR_A_INIT=1.0`
- `RECUR_B_INIT=0.0`
- ramped QAT-lite enabled

Result:

- raw pre-quant `val_bpb`: `1.1018`
- quantized `val_bpb`: `1.1243`
- TTT `val_bpb`: `1.09664494`
- total size: `16067345`

Interpretation:

- this is the best completed run seen so far in this research round
- it beats the previous best recurrent-alpha branch (`1.09808318`) by about `0.00144 BPB`
- it beats the best plain `ENABLE_LOOPING_AT=0.42` control (`1.09845764`) by about `0.00181 BPB`
- the gain came from a recurrence-native learned blend, not from post-quant correction tricks
- it is still over the size cap, but this is the first recurrence-side novelty after MuonEq-R that produced another clearly meaningful jump

### XSA on last 4 layers on top of the best `RECUR_AB` branch

Follow-up run adding XSA as an orthogonal attention-side change on top of the current best recurrence stack:

- `MUON_EQR=1`
- `EMA_DECAY=0`
- `ENABLE_LOOPING_AT=0.42`
- `RECUR_ALPHA_ENABLED=0`
- `RECUR_AB_ENABLED=1`
- `RECUR_A_INIT=1.0`
- `RECUR_B_INIT=0.0`
- ramped QAT-lite enabled
- `XSA_LAST_N=4`

Result:

- raw pre-quant `val_bpb`: `1.1032`
- quantized `val_bpb`: `1.1191`
- TTT `val_bpb`: `1.09783631`
- total size: `16066270`

Interpretation:

- this did not beat the current best `RECUR_AB` run (`1.09664494`)
- it was worse on final TTT by about `0.00119 BPB`
- however, it improved quantized BPB materially (`1.1243 -> 1.1191`)
- that makes XSA look like a real orthogonal signal, just not yet aligned with the best final TTT setting
- the strongest next XSA follow-up is likely `XSA_LAST_N=3`, not `11`, because the looped architecture already reuses the core layers and `XSA-all` was previously reported as slightly harmful on repeated-loop models

### XSA on last 2 layers on top of the best `RECUR_AB` branch

Follow-up run narrowing XSA further:

- `MUON_EQR=1`
- `EMA_DECAY=0`
- `ENABLE_LOOPING_AT=0.42`
- `RECUR_ALPHA_ENABLED=0`
- `RECUR_AB_ENABLED=1`
- `RECUR_A_INIT=1.0`
- `RECUR_B_INIT=0.0`
- ramped QAT-lite enabled
- `XSA_LAST_N=2`

Result:

- raw pre-quant `val_bpb`: `1.1036`
- quantized `val_bpb`: `1.1277`
- TTT `val_bpb`: `1.09865076`
- total size: `16067361`

Interpretation:

- this was worse than both the `XSA_LAST_N=4` branch (`1.09783631`) and the no-XSA best `RECUR_AB` branch (`1.09664494`)
- unlike `XSA_LAST_N=4`, it did not preserve a quantized-BPB advantage
- this makes `XSA_LAST_N=2` a miss
- if XSA is continued at all, the only remaining high-signal probe is `XSA_LAST_N=3`; otherwise the evidence now points back toward recurrence-side tuning

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
- `QAT_LITE_TARGETS`
- `QAT_LITE_PENALTY`
- `QAT_LITE_DEPTH_POWER`

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

Cleaned-up ramped QAT-lite result:

- raw pre-quant `val_bpb`: `1.1124`
- quantized `val_bpb`: `1.1317`
- TTT `val_bpb`: `1.10702468`
- total size: `16009986`

Interpretation of the cleaned-up ramped run:

- temporarily became the best final TTT seen so far in this research round
- slightly better quantized result than the plain EMA-off baseline
- still over the cap by `9986` bytes
- the overage is effectively all code-size pressure, not model-size pressure

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
- not worth carrying in the main working branch unless combined in a future experiment that clearly wins

Relevant commit:

- `fbc4c08` Add Q/K activation-tail regularizer

## Pass-Specific Low-Rank Recurrence Adapters

Novel legal idea tested:

- tiny pass-specific low-rank adapters on repeated loop visits only
- inspired by recent weight-sharing / recursive-transformer modulation papers
- purely training-time / artifact-time; no eval semantic changes

Representative run:

- raw pre-quant `val_bpb`: `1.1052`
- quantized `val_bpb`: `1.1217`
- TTT `val_bpb`: `1.09932544`
- total size: `16083187`

Conclusion:

- this was a miss
- it hurt raw model quality and final TTT relative to the best `RECUR_AB` branch
- even though quantized BPB stayed decent, the overall result was clearly worse than the current best local run
- not worth keeping in the main quality branch

Relevant commits:

- `38fabb9` Add pass-specific low-rank recurrence adapters
- `f50ce9f` Fix compiled recurrence adapter scale handling

## Hessian-Guided GPTQ Clip Search (HQClip)

Novel legal idea tested:

- keep the best `RECUR_AB` training stack fixed
- change only post-training GPTQ by searching a small set of clip values per matrix
- choose the clip that minimizes a Hessian-diagonal-weighted reconstruction score

Representative run:

- raw pre-quant `val_bpb`: `1.1036`
- quantized `val_bpb`: `1.1160`
- TTT `val_bpb`: `1.09509057`
- total size: `16906236`

Observed clip-search summary:

- `HQClip:chosen_clips count=67 mean=11.1571 min=10.9225 max=17.0000`

Interpretation:

- this is the **best quality result** seen so far in this research round
- it improves the previous best `RECUR_AB` branch (`1.09664494`) by about `0.00155 BPB`
- the gain comes almost entirely from better post-quant / post-TTT behavior, not better raw training quality
- however, the quantized artifact became much less compressible:
  - quantized blob jumped to about `16.82 MB`
  - total submission size jumped to about `16.91 MB`
- therefore the current broad HQClip formulation is a **good signal but bad packaging**

Conclusion:

- Hessian-guided clip search is promising and clearly real as a quality lever
- but it must be narrowed or targeted before it is usable under the artifact cap
- safest next direction: restrict HQClip to the most sensitive matrix groups rather than all 67 large matrices

Relevant commit:

- `88578ee` Add Hessian-guided GPTQ clip search

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
7. The code budget matters enough that dead experiment paths should be removed from `train_gpt.py` once they stop paying off.
8. The cleaned-up ramped QAT-lite branch was a strong stepping stone, but restoring MuonEq-R produced a much larger gain.
9. MuonEq-R is a real breakthrough, but after the scheduling fix there is strong evidence that the depth-recurrence activation was also previously missing from effective runs.
10. Late-start EMA still appears harmful even on the stronger MuonEq-R branch.
11. The strongest local quality branch is now: MuonEq-R + wallclock-aware recurrence + no EMA + ramped QAT-lite + recurrent alpha/beta memory blend + Hessian-guided GPTQ clip search, with `ttt val_bpb = 1.09509057`, though it is far over the size cap.
12. Recurrence scheduling is now a major optimization lever because it improves quality but sharply reduces the number of training steps that fit in 600s.
13. Recurrence-native learned blending looks more promising than post-quant residual correction for this stack, but pass-specific low-rank recurrence adapters were not helpful.
14. XSA on the deepest layers is mixed: `XSA_LAST_N=4` improved quantized BPB significantly, but did not improve final TTT; `XSA_LAST_N=2` regressed on both quantized and final TTT.
15. Hessian-guided clip search is the first clearly helpful post-quant novelty in this round, but its current broad form destroys compressibility and is not viable as-is.
16. The branch is still over the cap, but at this point the dominant challenge is closing the remaining quality gap to the public record while eventually recovering submission legality.

## Recommended Next Steps

If continuing pretraining-side novelty:

1. keep baseline quantization settings fixed while tuning training-side novelty
2. use QAT-lite as the main surviving novelty path
3. prefer ramped/late QAT-lite over carrying extra dead regularizers in the main file
4. only revive QACT-lite if a combined run clearly beats QAT-lite alone
5. treat the current `RECUR_AB` + `HQClip` branch as the main quality leader, but treat the plain `RECUR_AB` branch as the main size-aware control
6. if continuing XSA at all, test `XSA_LAST_N=3` next and stop if it misses
7. prefer targeted post-quant follow-ups over reviving `QRES`, `RECUR_LORA`, or broad XSA sweeps
8. treat MuonEq-R as part of the mainline stack, not an optional tweak

## Next High-Upside Hypothesis

### Recurrence schedule tuning under wallclock budget

Rationale:

- after the scheduling fix, recurrence actually turns on around wallclock fraction `0.35`
- real recurrence activation produced the best final TTT seen so far, and delaying it to `ENABLE_LOOPING_AT=0.42` plus adding recurrent alpha/beta blending improved that further to `1.09664494`
- however, it also reduced total training from about `4860` steps to about `3886` steps
- this tradeoff suggests the start fraction itself may now be suboptimal

Implementation added:

- wallclock-aware fraction scheduling for late-stage features
- `EMA_START_FRAC`

Idea:

- keep `MUON_EQR=1`
- keep wallclock-aware recurrence activation
- set `EMA_DECAY=0`
- use the current `ENABLE_LOOPING_AT=0.42` + `RECUR_AB` branch as the control
- test recurrence-native follow-ups before introducing tokenizer or eval-risk ideas:
  - small negative `RECUR_B_INIT`
  - slightly sub-unit `RECUR_A_INIT`
  - narrow recurrence-start sweep around `0.42`
- test narrow XSA follow-ups on top of that control:
  - `XSA_LAST_N=3`
- compare them against the current best `1.09664494` branch

If optimizing for highest practical win probability instead:

1. recover the exact public SP8192 record recipe cleanly
2. benchmark from there
3. only test one narrow delta at a time
