# I-CEM: Context-Aware Evidence Minimization for Harmful-Speech Dataset Release

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21974841.svg)](https://doi.org/10.5281/zenodo.21974841)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code, experiment configs, and aggregate results for:

> **Context-Aware Evidence Minimization for Privacy-Conscious Harmful-Speech
> Dataset Release.** Batın Örene. Proceedings of the 10th Workshop on Online
> Abuse and Harms (WOAH 2026).

The paper itself is not hosted here. It will be available in the ACL Anthology
once the WOAH 2026 proceedings are published; this repository holds the code and
the experimental record behind it.

## What this is

Curators who want to share harmful-speech corpora face a bind: releasing raw
text exposes more of the source than model comparison and error analysis
require, but releasing only the tokens a classifier finds important destroys the
context that makes a label interpretable.

**I-CEM** is a deterministic release baseline for that bind. Given an example it

1. masks direct and quasi identifier spans,
2. scores every eligible token by leave-one-out erasure attribution against a
   frozen harmful-speech classifier,
3. keeps the highest-scoring tokens plus a small symmetric window, and
4. **expands those spans only when deterministic rules detect a target--harm or
   stance--harm relation** (negation, quotation/reporting, or counter-speech).

Every step is rule-based and auditable. No paraphrasing, no generation: outputs
are verbatim spans of the source, so each release stays traceable.

## Headline result

Releasing only classifier-important tokens corrupts the data. Judged by an
independent, lexicon-free LLM, such excerpts recover the targeted group only
**36%** of the time and read **53%** of non-hateful excerpts (counter-speech,
negation, quotation) as hateful.

The paper's primary, multiply-grounded result is **target recovery**: against a
token-matched fixed window, I-CEM improves it from 58% to 67% (McNemar
*p*<.001), and the gain replicates against HateXplain's gold target labels
(50% to 56%, *p*=.0003) and under three of four judges in a multi-vendor panel.

Secondary gains are reported as **directional** and should be read that way:
hate-label fidelity improves modestly (90% vs 88%, *p*=.04) but does not hold
against gold labels, and the stance advantage at a matched token budget is not
significant (*p*=.052).

Recovered context is not free: when I-CEM's rules fire it releases more text
than a fixed window would. The method lets a curator make that
context-for-exposure trade deliberately; it is not a strictly better point on
the compression curve.

## What this is *not*

- **Not a de-identification tool.** The identifier stage in these experiments
  receives the recorded gold spans planted by a synthetic PII injector. The 0.0%
  residual in the tables confirms the masking pipeline executes end to end. It
  is not evidence that a detector would catch real-world identifiers, and it
  must not be read as anonymization. A real release needs a production PII
  recognizer and human review.
- **Not a hate-speech classifier.** Both classifiers are frozen, inference-only,
  and off the shelf.
- **Not multilingual.** All datasets and cue inventories are English. The stance
  rules in particular are lexical and would not port by translation alone; the
  paper's Limitations section discusses what a real port would require.

## Layout

```text
scripts/icem/                  I-CEM implementation (spans, importance, context rules, render, metrics)
scripts/run_icem_experiment.py Main experiment driver
scripts/run_label_recovery_judge.py   Independent LLM judge: hate label + target recovery
scripts/run_stance_llm_judge.py       Independent LLM judge: stance preservation
scripts/analyze_label_recovery_vs_gold.py   Re-anchors judge results to gold dataset labels
scripts/aggregate_label_recovery_judges.py  Multi-judge panel aggregation
tests/                         Smoke tests
docs/paper_woah_2026/results/  Per-run configs, manifests, and aggregate metrics (runs 003-027)
docs/paper_woah_2026/claims_audit.md   Claim-to-source ledger for every number in the paper
docs/paper_woah_2026/results.md        Consolidated results narrative
```

Each `results/<run>/` directory carries `config.json` and `run_manifest.json`,
so any table row in the paper can be traced back to the exact run that produced
it. `claims_audit.md` maps individual paper claims to those runs.

## Reproducing

Requires [micromamba](https://mamba.readthedocs.io/) (or conda; adjust the
commands) and the datasets, which are **not redistributed here** — see
[DATA.md](DATA.md).

```bash
micromamba env update -n icem-research -f environment.yml -y
micromamba run -n icem-research python -m pytest -q
micromamba run -n icem-research python scripts/run_icem_experiment.py --help
```

The LLM-judge experiments expect an OpenAI-compatible endpoint on the local
network (we used [LM Studio](https://lmstudio.ai/)). **Release text is real
dataset text and must never be sent to a hosted API** — the judges are run
locally for exactly this reason. Set the endpoint via the environment variables
documented in `scripts/run_label_recovery_judge.py`.

## Content warning

This repository concerns hateful content. The cue inventory in
`scripts/icem/context_rules.py` contains a verbatim slur list, used for lexical
matching. Following WOAH policy the paper describes and counts these terms
rather than reprinting them; they appear here only because the code needs them.
All examples in the paper and in `results/*/qualitative_examples.md` are
synthetic schematics with placeholders, not real dataset posts.

## License

The code and documentation in this repository are [MIT](LICENSE) licensed. The
paper is published separately by the ACL and carries its own license; it is not
distributed here. The datasets have their own licenses and are not included;
see [DATA.md](DATA.md).

## Citation

```bibtex
@inproceedings{orene-2026-context,
  title     = {Context-Aware Evidence Minimization for Privacy-Conscious Harmful-Speech Dataset Release},
  author    = {{\"O}rene, Bat{\i}n},
  booktitle = {Proceedings of the 10th Workshop on Online Abuse and Harms (WOAH 2026)},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff). Both will be
updated with pages and the ACL Anthology URL once the proceedings are live.

To cite this software specifically, use the archived snapshot:

> Örene, B. (2026). *I-CEM: Context-Aware Evidence Minimization for
> Privacy-Conscious Harmful-Speech Dataset Release* (v1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.21974841

That is the concept DOI, which always resolves to the latest version.

## Contact

Batın Örene, Department of Computer Engineering, Bahçeşehir University
([ORCID 0000-0002-3342-0808](https://orcid.org/0000-0002-3342-0808)).

Questions about the method or the results are best raised as a GitHub issue so
the answers stay public. For anything else: `batinorene+research@gmail.com`.
