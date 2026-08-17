# No-PII Robustness Check

Run directory:

```text
results/015_hatexplain_dehatebert_5000_no_pii/
```

This run uses the same 5,000-row HateXplain sample, seed, DeHateBERT selector,
and I-CEM parameters as the main result, but disables synthetic PII injection.
It checks whether the context/utility pattern depends on planted identifiers.

## Aggregate Result

| Variant | F1+ | Flip % | Tok % | Target % | Stance % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `importance_only` | 0.705 | 22.4 | 14.3 | 53.4 | 7.5 |
| `importance_window` | 0.726 | 12.2 | 49.7 | 71.9 | 37.4 |
| `icem_context` | 0.730 | 10.1 | 57.5 | 93.8 | 81.4 |

## Interpretation

The qualitative pattern survives without injected PII: I-CEM has lower
raw-decision flip rate and substantially higher target/stance preservation than
importance-only and fixed windows. This directly addresses the criticism that
the main effect is created by injecting identifiers and then removing them.

Use this as a robustness check only. Without synthetic PII, the run does not
measure controlled gold PII residuals, and I-CEM retains more source text
than in the main injected-PII run.
