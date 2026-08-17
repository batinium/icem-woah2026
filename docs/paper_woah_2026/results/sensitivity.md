# Selector Sensitivity

These runs use the revised selector that falls back to the strongest non-PII
token when no token passes the importance threshold. The main paper setting is
`top_k=5`, `window_radius=2`.

## Top-K Sweep

Full 5,000-row HateXplain runs with DeHateBERT. All other settings fixed.

| Setting | Variant | F1+ | Flip% | Tok% | Char% | Target% | Stance% | Empty |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k=3 | importance_only | 0.710 | 27.2 | 8.965 | 8.904 | 49.1 | 2.0 | 0 |
| k=3 | importance_window | 0.715 | 19.3 | 33.415 | 31.226 | 67.6 | 19.5 | 0 |
| k=3 | icem_context | 0.700 | 14.2 | 44.152 | 40.888 | 93.9 | 59.9 | 0 |
| k=5 | importance_only | 0.704 | 26.7 | 11.684 | 11.296 | 54.5 | 4.5 | 0 |
| k=5 | importance_window | 0.716 | 18.0 | 39.754 | 37.268 | 72.2 | 24.6 | 0 |
| k=5 | icem_context | 0.697 | 13.4 | 49.738 | 46.247 | 95.3 | 61.9 | 0 |
| k=7 | importance_only | 0.702 | 26.3 | 13.206 | 12.622 | 56.9 | 6.0 | 0 |
| k=7 | importance_window | 0.713 | 17.7 | 42.704 | 40.157 | 74.3 | 26.4 | 0 |
| k=7 | icem_context | 0.696 | 13.1 | 52.259 | 48.729 | 95.5 | 62.8 | 0 |
| k=10 | importance_only | 0.703 | 26.0 | 14.349 | 13.620 | 58.8 | 7.6 | 0 |
| k=10 | importance_window | 0.714 | 17.4 | 44.714 | 42.105 | 75.6 | 28.6 | 0 |
| k=10 | icem_context | 0.699 | 12.8 | 53.952 | 50.391 | 95.6 | 63.6 | 0 |

Interpretation: `k=3` is the most compressed but loses target and stance
context. `k=7` and `k=10` recover only small additional context while releasing
more text. `k=5` is a conservative middle setting.

## Window-Radius Sweep

Full 5,000-row HateXplain runs with DeHateBERT and `top_k=5`.

| Setting | Variant | F1+ | Flip% | Tok% | Char% | Target% | Stance% | Empty |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| radius=1 | importance_window | 0.729 | 22.6 | 28.243 | 26.555 | 65.7 | 14.3 | 0 |
| radius=1 | icem_context | 0.708 | 15.4 | 43.169 | 40.030 | 95.3 | 60.9 | 0 |
| radius=2 | importance_window | 0.716 | 18.0 | 39.754 | 37.268 | 72.2 | 24.6 | 0 |
| radius=2 | icem_context | 0.697 | 13.4 | 49.738 | 46.247 | 95.3 | 61.9 | 0 |
| radius=3 | importance_window | 0.703 | 15.0 | 48.790 | 46.816 | 77.2 | 33.8 | 0 |
| radius=3 | icem_context | 0.692 | 12.4 | 55.831 | 53.068 | 95.3 | 62.5 | 0 |

Interpretation: radius 1 improves compression and classifier F1 but increases
raw-decision flip rate. Radius 3 lowers flip rate slightly but releases much
more text. Radius 2 is the balanced main setting.

## Independent Evaluator Check

`unitary/unbiased-toxic-roberta` scores saved release rows. This evaluator is
not used for token selection.

| Setting | Variant | F1+ | Flip% | Tok% | Target% | Stance% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| radius=1 | importance_window | 0.691 | 16.5 | 28.243 | 65.7 | 14.3 |
| radius=1 | icem_context | 0.694 | 14.6 | 43.169 | 95.3 | 60.9 |
| radius=2 | importance_window | 0.693 | 14.5 | 39.754 | 72.2 | 24.6 |
| radius=2 | icem_context | 0.694 | 12.9 | 49.738 | 95.3 | 61.9 |

Interpretation: the independent evaluator does not favor switching the main
setting to radius 1. I-CEM F1 is the same for radius 1 and 2, while radius 2
has lower raw-decision flip and slightly stronger stance preservation.

## Selector Audit

The generated selector audit in
[`../selector_tradeoff_summary.md`](../selector_tradeoff_summary.md) formalizes
the recommendation as conservative thresholds rather than a post-hoc narrative.
Under those thresholds, `top_k=5`, `window_radius=2` is the only full
HateXplain setting that simultaneously keeps retained tokens under 50%, flip
rate at or below 13.5%, target preservation at or above 95%, stance-harm
preservation at or above 60%, and zero injected/detected PII residuals.
