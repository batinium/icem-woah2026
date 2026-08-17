# Classifier-transfer selector summary

This check uses `unitary/unbiased-toxic-roberta` for both token selection and
evaluation on the same 5,000 HateXplain sample. It answers a different question
from `results/007_hatexplain_dehatebert_release_unbiased_eval/`: here, the
evidence tokens themselves are selected by a different classifier.

Run directory:

```text
results/014_hatexplain_unbiased_selector_5000/
```

## Aggregate result

| Variant | F1+ | Flip % | Tok % | PII resid % | Target % | Stance % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `importance_only` | 0.720 | 23.3 | 14.1 | 0.0 | 66.2 | 6.3 |
| `importance_window` | 0.723 | 16.6 | 41.4 | 0.0 | 80.6 | 28.4 |
| `icem_context` | 0.716 | 13.5 | 50.9 | 0.0 | 97.0 | 63.9 |

## Interpretation

The direction matches the DeHateBERT selector run: I-CEM gives up a small amount
of F1 relative to fixed windows, but lowers raw-decision flips and preserves far
more target/stance context while removing injected direct and quasi identifiers.

This should be used as a robustness note, not as the main result. The toxicity
model is not hate-speech-specific, and the I-CEM retained-token rate is just
above the 50% line used for the main selector tradeoff audit.

## Reproducibility note

The local macOS micromamba environment hit a duplicate OpenMP runtime crash for
this model unless `KMP_DUPLICATE_LIB_OK=TRUE` was set. The completed run used:

```bash
KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false \
micromamba run -n icem-research python scripts/run_icem_experiment.py \
  --dataset hatexplain \
  --sample-size 5000 \
  --seed 17 \
  --classifier unitary/unbiased-toxic-roberta \
  --classifier-batch-size 64 \
  --classifier-device auto \
  --importance-top-k 5 \
  --window-radius 2 \
  --manual-review-sample-size 0 \
  --output-dir docs/paper_woah_2026/results/014_hatexplain_unbiased_selector_5000 \
  --full-output-dir data/outputs \
  --overwrite
```
