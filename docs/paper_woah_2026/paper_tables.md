# Paper-Ready Tables

Use these compact tables in the short paper. Keep extra metrics in the appendix
or result repository if page space is tight.

## Main HateXplain Ablation

Suggested caption:

> Main HateXplain release results over 5,000 examples. F1+ is positive-class
> F1 against dataset labels. Flip is the percentage of examples whose released
> text changes the frozen classifier decision relative to raw text. Tok is
> retained source-token percentage. PII residual is measured against injected
> direct/quasi identifier spans. Target and Stance are automatic cue
> preservation proxies.

| Variant | F1+ | Flip% | Tok% | PII resid% | Target% | Stance% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pii_quasi_mask` | 0.692 | 8.5 | 100.0 | 0.0 | 100.0 | 77.6 |
| `importance_only` | 0.704 | 26.7 | 11.7 | 0.0 | 54.5 | 4.5 |
| `importance_window` | 0.716 | 18.0 | 39.8 | 0.0 | 72.2 | 24.6 |
| `icem_context` | 0.697 | 13.4 | 49.7 | 0.0 | 95.3 | 61.9 |

## Independent Evaluator

Suggested caption:

> Cross-evaluator check on the same HateXplain release texts, scored with
> `unitary/unbiased-toxic-roberta`. The evaluator is not used to select spans.

| Variant | F1+ | Flip% | Tok% | Target% | Stance% |
| --- | ---: | ---: | ---: | ---: | ---: |
| `importance_only` | 0.674 | 21.3 | 11.7 | 54.5 | 4.5 |
| `importance_window` | 0.693 | 14.5 | 39.8 | 72.2 | 24.6 |
| `icem_context` | 0.694 | 12.9 | 49.7 | 95.3 | 61.9 |

## Selector Sensitivity

Suggested caption:

> Selector sensitivity for the proposed I-CEM variant on full 5,000-row
> HateXplain runs. We use `top_k=5`, `radius=2` as the main balanced setting.

| Setting | F1+ | Flip% | Tok% | Target% | Stance% |
| --- | ---: | ---: | ---: | ---: | ---: |
| `k=3, r=2` | 0.700 | 14.2 | 44.2 | 93.9 | 59.9 |
| `k=5, r=2` | 0.697 | 13.4 | 49.7 | 95.3 | 61.9 |
| `k=7, r=2` | 0.696 | 13.1 | 52.3 | 95.5 | 62.8 |
| `k=10, r=2` | 0.699 | 12.8 | 54.0 | 95.6 | 63.6 |
| `k=5, r=1` | 0.708 | 15.4 | 43.2 | 95.3 | 60.9 |
| `k=5, r=3` | 0.692 | 12.4 | 55.8 | 95.3 | 62.5 |

## Optional Classifier-Transfer Selector

Use this as a one-sentence robustness note if page space is tight.

| Variant | F1+ | Flip% | Tok% | Target% | Stance% |
| --- | ---: | ---: | ---: | ---: | ---: |
| `importance_only` | 0.720 | 23.3 | 14.1 | 66.2 | 6.3 |
| `importance_window` | 0.723 | 16.6 | 41.4 | 80.6 | 28.4 |
| `icem_context` | 0.716 | 13.5 | 50.9 | 97.0 | 63.9 |

## Optional HateCheck Aggregate

Use this only if there is space after the main table and cross-evaluator check.
HateCheck is most useful through its functionality breakdown CSV, not this
aggregate table. See `hatecheck_context_summary.md` for the generated
functionality-level counts.

| Variant | F1+ | Flip% | Tok% | Target% | Stance% |
| --- | ---: | ---: | ---: | ---: | ---: |
| `importance_only` | 0.257 | 12.9 | 19.4 | 73.0 | 7.7 |
| `importance_window` | 0.293 | 8.5 | 57.7 | 88.7 | 37.9 |
| `icem_context` | 0.247 | 5.7 | 66.6 | 99.7 | 68.2 |

Compact text if no table fits: against fixed windows, I-CEM improves target
preservation in 23/26 HateCheck functionality groups and stance-harm
preservation in 16/22, while lowering raw-decision flips in 17/29 groups at the
cost of higher retained-token exposure.

## LaTeX Skeleton

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrrr}
\toprule
Variant & F1+ & Flip & Tok & PII & Target & Stance \\
\midrule
PII+quasi mask & .692 & 8.5 & 100.0 & 0.0 & 100.0 & 77.6 \\
Importance only & .704 & 26.7 & 11.7 & 0.0 & 54.5 & 4.5 \\
Importance window & .716 & 18.0 & 39.8 & 0.0 & 72.2 & 24.6 \\
I-CEM & .697 & 13.4 & 49.7 & 0.0 & 95.3 & 61.9 \\
\bottomrule
\end{tabular}
\caption{Main HateXplain release results.}
\label{tab:main}
\end{table}
```
