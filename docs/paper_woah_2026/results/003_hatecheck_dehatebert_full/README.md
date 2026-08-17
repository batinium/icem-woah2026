# 003_hatecheck_dehatebert_full

Rows: `3728`
Dataset: `hatecheck`
Classifier: `Hate-speech-CNERG/dehatebert-mono-english`

Run command:

```bash
micromamba run -n icem-research python scripts/run_icem_experiment.py \
  --dataset hatecheck \
  --classifier Hate-speech-CNERG/dehatebert-mono-english \
  --classifier-batch-size 64 \
  --output-dir docs/paper_woah_2026/results/003_hatecheck_dehatebert_full \
  --overwrite
```

Contents:

- `main_table.md`: compact paper-style ablation table.
- `variant_metrics.csv`: flat aggregate metrics for analysis.
- `variant_metrics.json`: full aggregate metric dictionary.
- `context_breakdown.csv`: grouped utility/context metrics.
- `privacy_breakdown.csv`: PII and identifier residual metrics.
- `dataset_summary.json`: sampled dataset summary.
- `config.json`: run configuration, including classifier label mapping.
- `qualitative_examples.md`: synthetic example table.

Full transformed rows are stored outside the paper folder under:

```text
data/outputs/003_hatecheck_dehatebert_full/release_rows.jsonl
```
