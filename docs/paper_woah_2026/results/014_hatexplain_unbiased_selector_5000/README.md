# 014_hatexplain_unbiased_selector_5000

Rows: `5000`
Dataset: `hatexplain`
Classifier: `unitary/unbiased-toxic-roberta`

Run command:

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
data/outputs/014_hatexplain_unbiased_selector_5000/release_rows.jsonl
```

Note: `config.json` preserves legacy run metadata with
`manual_review_sample_size: 50`, but no manual-review CSV is included in the
paper folder or supplement. The manual-review sheet contains transformed text
and is not used by the reported aggregate metrics.
