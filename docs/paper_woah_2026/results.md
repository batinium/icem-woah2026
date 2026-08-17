# Experiment Result Index

## Main HateXplain Result

[results/005_hatexplain_dehatebert_5000/](results/005_hatexplain_dehatebert_5000/)
is the main ablation run: 5,000 HateXplain rows, synthetic PII injection, six
variants, and frozen `Hate-speech-CNERG/dehatebert-mono-english`.

| Variant | F1+ | Flip% | Tok% | Char% | PII resid% | Target% | Stance% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.684 | 0.0 | 100.000 | 100.000 | 100.0 | 100.0 | 100.0 |
| pii_mask | 0.679 | 3.6 | 100.000 | 100.000 | 0.0 | 100.0 | 100.0 |
| pii_quasi_mask | 0.692 | 8.5 | 100.000 | 100.000 | 0.0 | 100.0 | 77.6 |
| importance_only | 0.704 | 26.7 | 11.684 | 11.296 | 0.0 | 54.5 | 4.5 |
| importance_window | 0.716 | 18.0 | 39.754 | 37.268 | 0.0 | 72.2 | 24.6 |
| icem_context | 0.697 | 13.4 | 49.738 | 46.247 | 0.0 | 95.3 | 61.9 |

Paper use: I-CEM preserves substantially more target and stance context than
importance-only while retaining far less source text than full-text masking.
Classifier F1 is not the sole objective; discuss utility, reduction, privacy
proxy, and context separately.

### Token-matched fixed-window comparison (M3)

The fixed `importance_window` baseline in the main run (005) keeps only 39.8% of
tokens, less than I-CEM's 49.7%, so a context comparison against it conflates
*how much* text is released with *which* text. Run
[results/012_hatexplain_dehatebert_radius3_5000/](results/012_hatexplain_dehatebert_radius3_5000/)
is a radius-3 fixed window at a nearly matched token budget. Use its
`importance_window` row (not its `icem_context` row) as the token-matched
baseline in Table 1:

| Variant (run) | Tok% | Flip% | Target% | Stance% | F1+ |
| --- | --- | --- | --- | --- | --- |
| Fixed window r=2 (005 `importance_window`) | 39.8 | 18.0 | 72.2 | 24.6 | .716 |
| Fixed window r=3 (012 `importance_window`) | 48.8 | 15.0 | 77.2 | 33.8 | .703 |
| I-CEM (005 `icem_context`) | 49.7 | 13.4 | 95.3 | 61.9 | .697 |

Paper use: at a matched ~49% token budget the fixed window still recovers far
less target/stance context than I-CEM, so I-CEM's advantage is which tokens
(relation-anchored spans) rather than more tokens.

## Multi-seed Variance (M1)

The main HateXplain configuration was rerun over four sampling seeds to show the
context gains are not seed artifacts:

- [results/005_hatexplain_dehatebert_5000/](results/005_hatexplain_dehatebert_5000/) (seed 17, main)
- [results/016_hatexplain_seed7_5000/](results/016_hatexplain_seed7_5000/) (seed 7)
- [results/017_hatexplain_seed23_5000/](results/017_hatexplain_seed23_5000/) (seed 23)
- [results/018_hatexplain_seed41_5000/](results/018_hatexplain_seed41_5000/) (seed 41)

Aggregate with `scripts/summarize_seed_variance.py`; the generated
[seed_variance_summary.md](seed_variance_summary.md) holds mean ± std. Headline
mean ± std across seeds:

| Variant | F1+ | Flip% | Target% | Stance% |
| --- | --- | --- | --- | --- |
| Importance only | .704 ± .002 | 25.8 ± .6 | 54.8 ± .3 | 4.0 ± .3 |
| Importance window (r=2) | .712 ± .005 | 17.5 ± .7 | 72.8 ± .8 | 24.4 ± .7 |
| I-CEM | .698 ± .002 | 13.2 ± .4 | 95.1 ± .4 | 61.7 ± 1.0 |

Paper use: the ~22pt target and ~37pt stance gaps far exceed the across-seed std
(<1pt); F1 differences are within ~0.015. This is seed variance on one sampled
dataset, not multi-dataset replicability.

## HateCheck Context Stress Result

[results/003_hatecheck_dehatebert_full/](results/003_hatecheck_dehatebert_full/)
is the full 3,728-row HateCheck run. Use `context_breakdown.csv` for grouped
negation, quotation, counterspeech, and target-change claims.
The generated [hatecheck_context_summary.md](hatecheck_context_summary.md)
turns that CSV into paper-ready functionality-level comparisons.

Compact aggregate table:

| Variant | F1+ | Flip% | Tok% | Char% | PII resid% | Target% | Stance% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.214 | 0.0 | 100.000 | 100.000 | 100.0 | 100.0 | 100.0 |
| pii_mask | 0.210 | 2.9 | 100.000 | 100.000 | 0.0 | 100.0 | 100.0 |
| pii_quasi_mask | 0.257 | 5.4 | 100.000 | 100.000 | 0.0 | 100.0 | 70.1 |
| importance_only | 0.257 | 12.9 | 19.431 | 18.199 | 0.0 | 73.0 | 7.7 |
| importance_window | 0.293 | 8.5 | 57.694 | 52.537 | 0.0 | 88.7 | 37.9 |
| icem_context | 0.247 | 5.7 | 66.550 | 60.647 | 0.0 | 99.7 | 68.2 |

Functionality-level summary: against fixed `importance_window`, I-CEM improves
target preservation in 23/26 HateCheck functionality groups, stance-harm
preservation in 16/22, negation preservation in 22/29, quotation preservation
in 29/29, and lowers raw-decision flips in 17/29. This should be reported with
the higher retained-token exposure shown in the aggregate table.

## Civil Comments Domain-Transfer Check

[results/006_civil_comments_dehatebert_1000/](results/006_civil_comments_dehatebert_1000/)
is a 1,000-row domain-transfer check on `google/civil_comments`. The loader uses
`toxicity >= 0.5` as the gold utility label. The Hugging Face default config
exposes toxicity and six subtype scores, but not full identity-mention columns;
therefore target groups are lexical, non-gold context cues.

| Variant | F1+ | Flip% | Tok% | Char% | PII resid% | Target% | Stance% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.125 | 0.0 | 100.000 | 100.000 | 100.0 | 100.0 | 100.0 |
| pii_mask | 0.112 | 1.1 | 100.000 | 100.000 | 0.0 | 100.0 | 100.0 |
| pii_quasi_mask | 0.153 | 2.2 | 100.000 | 100.000 | 0.0 | 100.0 | 96.8 |
| importance_only | 0.178 | 7.7 | 10.982 | 10.918 | 0.0 | 19.3 | 3.2 |
| importance_window | 0.170 | 5.1 | 37.608 | 35.856 | 0.0 | 43.9 | 17.9 |
| icem_context | 0.132 | 4.8 | 43.186 | 40.915 | 0.0 | 86.1 | 57.7 |

Paper use: this is not a strong utility result because DeHateBERT is a hate
classifier evaluated against out-of-domain toxicity labels. It is useful as a
sensitivity check: even with weak raw utility, I-CEM retains far less text than
full-text masking and preserves much more target/stance context than
importance-only.

## Cross-Classifier Sensitivity Check

[results/007_hatexplain_dehatebert_release_unbiased_eval/](results/007_hatexplain_dehatebert_release_unbiased_eval/)
scores the already-generated 5,000-row HateXplain release variants with
`unitary/unbiased-toxic-roberta`. This evaluator is not used to select
importance anchors or render release text. It probes whether the main utility
pattern is tied only to DeHateBERT, the selector model.

The evaluator uses its `toxicity` output at index 0 with sigmoid activation.

| Variant | F1+ | Flip% | Tok% | Char% | PII resid% | Target% | Stance% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.661 | 0.0 | 100.000 | 100.000 | 100.0 | 100.0 | 100.0 |
| pii_mask | 0.668 | 1.5 | 100.000 | 100.000 | 0.0 | 100.0 | 100.0 |
| pii_quasi_mask | 0.667 | 2.8 | 100.000 | 100.000 | 0.0 | 100.0 | 77.6 |
| importance_only | 0.674 | 21.3 | 11.684 | 11.296 | 0.0 | 54.5 | 4.5 |
| importance_window | 0.693 | 14.5 | 39.754 | 37.268 | 0.0 | 72.2 | 24.6 |
| icem_context | 0.694 | 12.9 | 49.738 | 46.247 | 0.0 | 95.3 | 61.9 |

Paper use: this reduces the circularity concern. I-CEM release text was selected
with DeHateBERT, but a separate toxicity model still gives I-CEM nearly the
same positive-class F1 as the fixed-window baseline while preserving much more
target and stance context.

## No-PII Robustness Check

[results/015_hatexplain_dehatebert_5000_no_pii/](results/015_hatexplain_dehatebert_5000_no_pii/)
uses the same 5,000-row HateXplain sample and DeHateBERT selector settings as
the main run, but disables synthetic PII injection. This checks whether the
context-recovery pattern is caused by injecting identifiers and then removing
them.

| Variant | F1+ | Flip% | Tok% | Char% | Target% | Stance% |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 0.719 | 0.0 | 100.000 | 100.000 | 100.0 | 100.0 |
| pii_mask | 0.719 | 0.0 | 100.000 | 100.000 | 100.0 | 100.0 |
| pii_quasi_mask | 0.719 | 0.1 | 100.000 | 100.000 | 100.0 | 99.6 |
| importance_only | 0.705 | 22.4 | 14.339 | 15.086 | 53.4 | 7.5 |
| importance_window | 0.726 | 12.2 | 49.733 | 49.923 | 71.9 | 37.4 |
| icem_context | 0.730 | 10.1 | 57.467 | 57.803 | 93.8 | 81.4 |

Paper use: the qualitative I-CEM pattern persists without injected PII. Use this
only as a robustness check, because the run does not provide controlled gold
PII residual measurement and I-CEM retains more text than in the main injected
run.

## Classifier-Transfer Selector Check

[results/014_hatexplain_unbiased_selector_5000/](results/014_hatexplain_unbiased_selector_5000/)
uses `unitary/unbiased-toxic-roberta` for token selection and evaluation on the
same 5,000-row HateXplain sample. This probes whether the context-recovery
pattern survives when evidence anchors are selected by a different classifier.

| Variant | F1+ | Flip% | Tok% | Char% | PII resid% | Target% | Stance% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.661 | 0.0 | 100.000 | 100.000 | 100.0 | 100.0 | 100.0 |
| pii_mask | 0.668 | 1.5 | 100.000 | 100.000 | 0.0 | 100.0 | 100.0 |
| pii_quasi_mask | 0.667 | 2.8 | 100.000 | 100.000 | 0.0 | 100.0 | 77.6 |
| importance_only | 0.720 | 23.3 | 14.053 | 13.255 | 0.0 | 66.2 | 6.3 |
| importance_window | 0.723 | 16.6 | 41.438 | 39.436 | 0.0 | 80.6 | 28.4 |
| icem_context | 0.716 | 13.5 | 50.860 | 47.767 | 0.0 | 97.0 | 63.9 |

Paper use: this is a robustness note. I-CEM again lowers flip rate and recovers
target/stance context relative to fixed windows, but the toxicity model is not a
hate-specific selector and I-CEM retains 50.9% of source tokens, slightly above
the main 50% threshold.

## Selector Sensitivity

[results/sensitivity.md](results/sensitivity.md) summarizes full 5,000-row
HateXplain sweeps over `top_k` and `window_radius`, plus an independent
evaluator check for the strongest compression alternative. The main setting is
kept at `top_k=5`, `window_radius=2`: `top_k=3` and `radius=1` release less text
but increase raw-decision flips or lose context, while larger values release
more text for modest additional gains.

## Gold Human-Rationale Overlap (proxy validation, M2)

The target/stance metrics are lexical proxies. To validate them against an
external human signal, use the `rationale_overlap` column (fraction of gold
human-marked tokens each release retains). **Use no-PII runs only:**
`with_replaced_text` does not remap `rationale_token_mask`, so PII injection
misaligns it — rationale numbers from injected runs (005/012/016–018) are invalid.

HateXplain four-seed mean ± std, no-PII (runs 015/019/020/021), via
[rationale_overlap_summary.md](rationale_overlap_summary.md):

| Variant | Gold rationale retained |
| --- | --- |
| importance_only | 36.9 ± 0.2 |
| importance_window (r=2) | 68.8 ± 0.5 |
| icem_context | 75.7 ± 0.5 |

Token-matched (seed 17): radius-3 fixed window 75.1% at 59.7% tokens
(run 022 `importance_window`) vs. I-CEM 75.3% at 57.5% tokens (run 015
`icem_context`) — **comparable at matched budget**, not an I-CEM win.

Paper use (honest framing): this replaces the single-author audit. I-CEM retains
far more human-flagged content than importance-only and as much as a
matched-budget window, so the proxies track genuine human-relevant content (not
circular). At matched budget I-CEM is on par with the window on raw rationale;
its distinctive advantage is stance (rationales do not annotate the stance
relation). Rationales mark label-relevance, not reviewer-judged sufficiency.

### Second-dataset replication: Toxic Spans (M2/generality)

[results/023_toxic_spans_dehatebert_5000/](results/023_toxic_spans_dehatebert_5000/)
(r=2) and [results/024_toxic_spans_radius3_5000/](results/024_toxic_spans_radius3_5000/)
(r=3, token-matched): SemEval-2021 Toxic Spans Detection (CC0), 5,000 rows, seed
17, no PII, DeHateBERT. Gold human toxic spans = a second rationale signal with a
different annotation scheme.

| Variant | F1+ | Flip% | Tok% | Stance% | Gold span % |
| --- | --- | --- | --- | --- | --- |
| importance_only | 0.419 | 24.5 | 13.3 | 1.6 | 65.7 |
| importance_window (r=2) | 0.347 | 11.7 | 45.5 | 22.5 | 77.6 |
| importance_window (r=3) | 0.340 | 9.8 | 54.3 | 30.1 | 81.1 |
| icem_context | 0.341 | 10.6 | 50.7 | 64.2 | 79.2 |

Paper use: the **stance advantage replicates** (I-CEM 64.2 vs window 22.5/30.1 vs
importance-only 1.6) and holds at matched budget → context result is not
HateXplain-specific. Gold-span overlap is comparable between I-CEM and matched
windows (same as HateXplain). F1+ low because DeHateBERT is out of domain on
toxicity — report as context replication, not utility.

The earlier 30-row author manual audit
(`data/audits/author_sanity_audit_reviewed.csv`,
[author_sanity_audit_summary.md](author_sanity_audit_summary.md)) is **no longer
used in the draft**. It remains a local artifact only; do not reintroduce the
21/17/7 counts into the paper.

## Independent LLM Judges (external grounding, runs 025–027)

The Target/Stance metrics are lexical proxies scored with the same cue lexicons
I-CEM expands on, so they measure rule firing, not human sufficiency. To break that
circularity, an independent, lexicon-free LLM judge reads release excerpts and
scores them. **Judge: `qwen/qwen3.5-9b` served locally via LM Studio on the LAN
a LAN-local OpenAI-compatible endpoint, temperature 0. No release text leaves the local
network — never send release text to a cloud LLM (data-use policy).**

### Stance grounding (run 025)

[results/025_stance_llm_judge/](results/025_stance_llm_judge/)
(`stance_judge_summary.md`, `summary.json`). 963 no-PII HateXplain rows where a
stance and harm cue co-occur (runs 015 + 022). The judge independently defines the
denominator: the **228** rows whose full raw text reads as a non-assertion stance
(negate/quote/oppose). Metric = fraction whose release excerpt still reads as a
non-assertion stance.

| Variant | Tok% | Stance preserved | 95% CI |
| --- | ---: | ---: | --- |
| Importance only | 14.3 | 1.3% | [0.0, 3.1] |
| Window (r=2) | 49.7 | 26.3% | [20.6, 32.0] |
| Window (r=3, matched) | 59.7 | 42.1% | [36.0, 48.7] |
| **I-CEM** | 57.5 | **49.1%** | [42.5, 55.7] |

McNemar exact: I-CEM vs importance-only **p < 1e-4**; I-CEM vs window r=2
**p < 1e-4**; **I-CEM vs token-matched window r=3: p = 0.052 — NOT significant**.

Paper use (honest): the grounded claim is "importance-only collapses stance
(1.3%), context recovery restores it (49.1%)" — large and highly significant. The
proxy's 1.8× matched-budget edge shrinks to ~1.17× and is **not** significant; the
matched-budget stance win is directional only. Consistent with the matched-budget
gold-rationale tie (C26).

### Label/target recoverability (run 026 — HEADLINE)

[results/026_label_recovery_judge/](results/026_label_recovery_judge/)
(`label_recovery_summary.md`, `summary.json`). Same judge/data. Per excerpt, two
independent questions vs the judge's read of the full raw text: (1) hateful toward a
group? (2) which group? A release corrupts the label if its excerpt flips the hate
decision or loses the target.

Hate-label fidelity (excerpt decision == full-text decision; n=963):

| Variant | Tok% | Fidelity | 95% CI |
| --- | ---: | ---: | --- |
| Importance only | 14.3 | 76.9% | [74.2, 79.6] |
| Window (r=2) | 49.7 | 84.9% | [82.6, 87.1] |
| Window (r=3, matched) | 59.7 | 87.7% | [85.7, 89.8] |
| **I-CEM** | 57.5 | **89.7%** | [87.7, 91.7] |

McNemar: I-CEM vs importance-only **p < 1e-4**; **I-CEM vs token-matched window
p = 0.040 (significant)**.

The harm — on the 200 rows the judge reads as **not hateful** in full text
(counterspeech/negation/quotation), fraction the excerpt flips to hateful:
importance-only **53.0%**, window r=2 34.5%, matched window 28.0%, **I-CEM 24.5%**.

Target recovery (excerpt target overlaps full-text target; n=670): importance-only
35.5%, window r=2 52.1%, matched window 58.2% [54.5, 61.9], **I-CEM 67.3%**
[63.7, 70.7]. McNemar: I-CEM vs importance-only **p < 1e-4**; **I-CEM vs matched
window p < 1e-4** (103 vs 42 discordant).

Paper use (HEADLINE): naive importance-only minimization corrupts the label —
53% of non-hateful content reads as hateful and only 35% of targets survive, so a
dataset minimized that way systematically relabels counterspeech/quotation as hate.
I-CEM **significantly** beats a token-matched window on both label fidelity (89.7 vs
87.7, p=0.04) and target recovery (67.3 vs 58.2, p<1e-4) — the matched-budget win
the stance proxy could not establish (025, p=0.052) shows up on the
label-corruption measures that actually matter for dataset validity.

### Exposure-vs-stance Pareto frontier (runs 025 + 027)

[results/025_stance_llm_judge/pareto.json](results/025_stance_llm_judge/pareto.json);
run [results/027_hatexplain_nopii_radius1_5000/](results/027_hatexplain_nopii_radius1_5000/)
supplies the no-PII r=1 window point. Judge-scored stance vs retained-token exposure:
the window family traces (tok%, stance%) = (14.3, 1.3) → (35.5, 11.0) →
(49.7, 26.3) → (59.7, 42.1); I-CEM sits at (57.5, **49.1**). Interpolating the
window frontier to I-CEM's 57.5% exposure gives 38.6% stance, so **I-CEM is 10.5
points above the window frontier** at equal exposure.

Paper use: I-CEM is not merely a point on the window's exposure/stance tradeoff —
it dominates the frontier by ~10 pts at matched exposure on the independent judge.

### Multi-judge robustness (run 026, by_model/)

To address single-judge dependence, the label/target-recovery test is replicated
across **three additional independent local judges** spanning two more vendors and a
range of scales: `qwen/qwen3.5-9b` re-run as a protocol-matched control,
`gemma-4-26b-a4b-it` (Google, 26B MoE) and `openai/gpt-oss-20b` (OpenAI, 20B). All
panel judges use a **tool-calling** protocol (`ICEM_JUDGE_TOOLS=1`): the judge is
forced to emit a structured tool call, which unlocks reasoning-tuned models whose
plain-completion answer otherwise lands in `reasoning_content` with empty `content`
(gemma-4-12b, qwen3.6-27b, gpt-oss-20b, phi-4 all fail plain short-answer). The
panel runs on a deterministic **400-row subsample** (seed 17, same rows for every
judge) for tractability — LM Studio serializes requests on one model slot, so the
full 963×N would be ~15h. Driver `scripts/overnight_judge_panel.sh`, aggregate
`scripts/aggregate_label_recovery_judges.py` →
[results/026_label_recovery_judge/multi_judge_summary.md](results/026_label_recovery_judge/multi_judge_summary.md).

I-CEM vs token-matched window r=3 (p = McNemar exact; importance-only column shown
for the collapse baseline):

| Judge (protocol, n) | imp-only fid | matched fid | **I-CEM fid** | fid p | imp-only tgt | matched tgt | **I-CEM tgt** | tgt p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3.5-9b (plain, n=963)¹ | 76.9 | 87.7 | **89.7** | 0.040* | 35.5 | 58.2 | **67.3** | <1e-4* |
| qwen3.5-9b (tool, n=400) | 61.2 | 77.8 | **83.0** | 0.010* | 50.3 | 67.5 | **73.2** | 0.022* |
| gemma-4-26b (tool, n=400) | 53.1 | 79.6 | **82.1** | 0.245 | 58.9 | 73.4 | **75.3** | 0.430 |
| gpt-oss-20b (tool, n=400) | 65.7 | 81.9 | **84.4** | 0.237 | 56.6 | 69.5 | **75.9** | 0.015* |

¹ committed headline run (full 963 rows, plain completion). * p < 0.05.

Paper use (honest framing): the **direction is unanimous** — every judge ranks
I-CEM ≥ token-matched window on both label fidelity and target recovery, and ranks
importance-only worst. **Target recovery** is the more robust win: I-CEM
significantly beats the matched window for 3 of 4 judges (only gemma-26b n.s.).
**Label fidelity** is significant for both Qwen variants and directional (n.s. at
n=400) for gemma-26b and gpt-oss — consistent with the borderline matched-budget
margin (committed p=0.04) that the stance proxy also could not separate (025).
The non-hate→hate misread (the corruption harm) is worst for importance-only under
all four judges, though its magnitude is judge-dependent (e.g. qwen 53% vs gemma
20%), and for gpt-oss the I-CEM-vs-matched gap on that sub-metric is not clean.
Bottom line: the **headline corruption claim and the importance-only→context
recovery ordering replicate across vendors**; I-CEM's edge over a *wide matched
window* is robust in direction but only sometimes significant — report it as such.

## Sensitivity Runs

- [results/008_hatexplain_dehatebert_topk3_5000/](results/008_hatexplain_dehatebert_topk3_5000/),
  [results/009_hatexplain_dehatebert_topk7_5000/](results/009_hatexplain_dehatebert_topk7_5000/),
  and [results/010_hatexplain_dehatebert_topk10_5000/](results/010_hatexplain_dehatebert_topk10_5000/):
  full top-k sensitivity runs.
- [results/011_hatexplain_dehatebert_radius1_5000/](results/011_hatexplain_dehatebert_radius1_5000/),
  [results/012_hatexplain_dehatebert_radius3_5000/](results/012_hatexplain_dehatebert_radius3_5000/),
  and [results/013_hatexplain_radius1_unbiased_eval/](results/013_hatexplain_radius1_unbiased_eval/):
  window-radius sensitivity runs and independent radius-1 evaluator check.
