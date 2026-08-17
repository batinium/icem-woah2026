# Data

This repository does **not** redistribute any dataset. It ships code, run
configs, run manifests, and aggregate metrics only. No raw posts, no transformed
release rows, and no per-row text of any kind are included.

Two categories of per-row artifact do appear, and both are text-free:

- `docs/paper_woah_2026/results/*/checkpoint.jsonl` — dataset row IDs, gold
  labels, gold target groups, and the judge's verdict per release variant. No
  source text.
- `docs/paper_woah_2026/results/*/qualitative_examples.md` — synthetic
  schematics with `[GROUP]`/`[SLUR]`/`[PERSON]` placeholders, written to
  illustrate method behavior. Not sampled dataset rows.

## Obtaining the datasets

| Dataset | Used for | Source | License (as checked) |
|---|---|---|---|
| HateXplain | Main benchmark; gold rationales and target groups | [hate-alert/HateXplain](https://github.com/hate-alert/HateXplain) | Inconsistent across sources: Hugging Face metadata lists CC-BY-4.0; the dataset card's licensing section and the GitHub repo list MIT |
| HateCheck | Functional context stress test | [paul-rottger/hatecheck-data](https://github.com/paul-rottger/hatecheck-data) | CC-BY-4.0 |
| Civil Comments | Out-of-domain sensitivity check | [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments) | CC0-1.0 |
| SemEval-2021 Task 5 Toxic Spans | Second-dataset rationale replication | [`heegyu/toxic-spans`](https://huggingface.co/datasets/heegyu/toxic-spans) mirror | CC0-1.0 |

Verify the license at the source before redistributing anything derived from
these corpora. The HateXplain discrepancy above is unresolved as of publication
and is noted in the paper's Ethics section.

## Models

Both classifiers are frozen and used for inference only; no parameters are
updated.

- `Hate-speech-CNERG/dehatebert-mono-english` (BERT-base scale, ~110M)
- `unitary/unbiased-toxic-roberta` via Detoxify (RoBERTa-base scale, ~125M)

## Synthetic PII

The residual-identifier experiments inject deterministic synthetic identifiers
(fake names, handles, dates, locations, schools, workplaces, contact strings)
and record their exact character offsets. Those recorded offsets are what the
masker consumes, which is why the residual is zero by construction. See
`scripts/icem/synthetic_pii.py`. No real personal data is injected, detected, or
released.

## Expected local layout

Experiment scripts read datasets from a local `data/` directory, which is
gitignored. Point the scripts at your own copies; see
`scripts/run_icem_experiment.py --help`.
