# 015_hatexplain_dehatebert_5000_no_pii

Rows: `5000`
Dataset: `hatexplain`
Classifier: `Hate-speech-CNERG/dehatebert-mono-english`

Run command:

```bash
micromamba run -n icem-research python scripts/run_icem_experiment.py \
  --dataset hatexplain \
  --sample-size 5000 \
  --classifier Hate-speech-CNERG/dehatebert-mono-english \
  --classifier-batch-size 64 \
  --no-inject-synthetic-pii \
  --output-dir docs/paper_woah_2026/results/015_hatexplain_dehatebert_5000_no_pii \
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
data/outputs/015_hatexplain_dehatebert_5000_no_pii/release_rows.jsonl
```
