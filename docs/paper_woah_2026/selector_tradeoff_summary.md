# Selector tradeoff summary

Generated from saved experiment folders with `scripts/summarize_selector_tradeoffs.py`.

Conservative selector thresholds used for the paper default:

- zero injected and detected direct/quasi PII residuals
- zero placeholder-only releases
- target cue preservation >= 95%
- stance-harm relation preservation >= 60%
- raw-decision flip rate <= 13.5%
- retained source tokens <= 50%

Recommended setting under these thresholds: `k=5, r=2`.

This supports the current default `top_k=5, radius=2`: it is the only full
HateXplain selector setting in the sweep that satisfies all conservative
thresholds while staying below 50% retained source tokens.

## Full HateXplain selector sweep

Variant summarized: `icem_context` over 5,000 HateXplain examples.

| Run | Setting | F1+ | Flip % | Tok % | Target % | Stance % | Zero PII | Recommended | Dominated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 011_hatexplain_dehatebert_radius1_5000 | k=5, r=1 | 0.708 | 15.4 | 43.2 | 95.3 | 60.9 | yes | no | no |
| 008_hatexplain_dehatebert_topk3_5000 | k=3, r=2 | 0.700 | 14.2 | 44.2 | 93.9 | 59.9 | yes | no | no |
| 005_hatexplain_dehatebert_5000 | k=5, r=2 | 0.697 | 13.4 | 49.7 | 95.3 | 61.9 | yes | yes | no |
| 009_hatexplain_dehatebert_topk7_5000 | k=7, r=2 | 0.696 | 13.1 | 52.3 | 95.5 | 62.8 | yes | no | no |
| 010_hatexplain_dehatebert_topk10_5000 | k=10, r=2 | 0.699 | 12.8 | 54.0 | 95.6 | 63.6 | yes | no | no |
| 012_hatexplain_dehatebert_radius3_5000 | k=5, r=3 | 0.692 | 12.4 | 55.8 | 95.3 | 62.5 | yes | no | no |

## Independent evaluator check

These rows rescore saved release texts with `unitary/unbiased-toxic-roberta`,
which was not used for token selection.

| Source run | Variant | Setting | F1+ | Flip % | Tok % | Target % | Stance % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 005_hatexplain_dehatebert_5000 | importance_only | k=5, r=2 | 0.674 | 21.3 | 11.7 | 54.5 | 4.5 |
| 005_hatexplain_dehatebert_5000 | importance_window | k=5, r=2 | 0.693 | 14.5 | 39.8 | 72.2 | 24.6 |
| 005_hatexplain_dehatebert_5000 | icem_context | k=5, r=2 | 0.694 | 12.9 | 49.7 | 95.3 | 61.9 |
| 011_hatexplain_dehatebert_radius1_5000 | importance_only | k=5, r=1 | 0.674 | 21.3 | 11.7 | 54.5 | 4.5 |
| 011_hatexplain_dehatebert_radius1_5000 | importance_window | k=5, r=1 | 0.691 | 16.5 | 28.2 | 65.7 | 14.3 |
| 011_hatexplain_dehatebert_radius1_5000 | icem_context | k=5, r=1 | 0.694 | 14.6 | 43.2 | 95.3 | 60.9 |

## Interpretation

The sweep does not show a single setting that is best on every metric. Radius 1
and lower `top_k` settings release less text, but they increase prediction
flips or miss context. Larger settings reduce flips slightly, but cross the
50% retained-token line for limited context gains. The default therefore has a
clear paper rationale: it is a conservative balance, not the numerically best
setting for one isolated score.
