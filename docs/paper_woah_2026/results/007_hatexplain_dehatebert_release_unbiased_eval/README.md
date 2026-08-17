# 007_hatexplain_dehatebert_release_unbiased_eval

Cross-classifier evaluation of already-generated I-CEM release rows.

Release rows: `data/outputs/005_hatexplain_dehatebert_5000/release_rows.jsonl`
Rows: `5000`
Evaluator: `unitary/unbiased-toxic-roberta`

This run does not recompute token importance or release text. It scores
the same released variants with an independent frozen evaluator to probe
whether utility preservation is tied to the selector classifier.

Contents:

- `main_table.md`: compact evaluator table.
- `variant_metrics.csv`: flat aggregate metrics.
- `variant_metrics.json`: full metric dictionary.
- `config.json`: evaluator configuration and label mapping.
