# Claims Audit

Last updated: 2026-06-24 (reviewer-pass precision edits: abstract matched-budget
stance multiple corrected to ~1.8x; unified Detoxify/unbiased-toxic-roberta
naming; added explicit no-external-validation note for the stance metric).

Use this file as the first stop before editing the WOAH draft. Every substantive
claim in `draft.tex` should be traceable to one of: a primary source, a saved
experiment file, a verification command, or an explicit limitation. If a claim
cannot be traced, soften it or remove it.

## Agent Rules

- Prefer primary sources over search snippets or Consensus links.
- Treat Consensus links in `literature.md` as discovery links only until the
  primary paper page is read.
- The author manual audit has been removed from the draft (superseded by gold
  human-rationale overlap, C26). Do not reintroduce it; if ever cited again,
  report it only as a small single-author check, never as formal annotation or
  inter-annotator agreement.
- Report rationale overlap as automatic span intersection with gold human
  rationales (HateXplain) or gold human toxic spans (SemEval Toxic Spans) --- an
  external human signal corroborating the lexical proxies, not a human usability
  study.
- **Rationale overlap is only valid on no-PII runs.** `with_replaced_text`
  (`scripts/icem/datasets.py`) does not shift `rationale_token_mask`, so synthetic
  PII injection misaligns it. Cite rationale numbers from no-PII runs
  (HateXplain 015/019--022; Toxic Spans 023/024) only. All other metrics
  (F1/Flip/Tok/Target/Stance) are computed on the live source text and are
  unaffected by injection.
- Do not call injected-span residuals a real-world anonymization guarantee.
- Do not describe automatic target/stance preservation as human judgment.
- The model `unitary/unbiased-toxic-roberta` is the Detoxify "unbiased" model.
  Name it consistently as "Detoxify (unbiased-toxic-roberta)" whether it acts as
  the independent evaluator (run 007) or the transfer selector (run 014); do not
  call it a generic "RoBERTa toxicity model" in one place and "Detoxify" in
  another.
- Gold rationales validate the **target** proxy only. The **stance** metric is
  externally grounded by the independent LLM judge (C28), not by rationales/toxic
  spans (which do not annotate the stance relation). The judge confirms the large
  importance-only stance collapse but finds I-CEM's matched-window edge small and
  non-significant; report the matched-budget stance advantage as directional, not
  as ~1.8x or significant. A human study is still future work.
- Do not call the selector sweep exhaustive; it is a bounded grid over the
  settings in `selector_tradeoff_summary.csv`.
- When updating result claims, rerun the verification commands below and update
  the row in this file.

## Verification Commands

Primary automated checks run in this audit:

```bash
micromamba run -n icem-research python -m pytest -q
micromamba run -n icem-research python scripts/summarize_selector_tradeoffs.py
micromamba run -n icem-research python scripts/summarize_hatecheck_context.py
micromamba run -n icem-research python scripts/summarize_seed_variance.py \
  --runs docs/paper_woah_2026/results/005_hatexplain_dehatebert_5000 \
         docs/paper_woah_2026/results/016_hatexplain_seed7_5000 \
         docs/paper_woah_2026/results/017_hatexplain_seed23_5000 \
         docs/paper_woah_2026/results/018_hatexplain_seed41_5000 \
  --seeds 17 7 23 41 --out docs/paper_woah_2026/seed_variance_summary.md
TEXINPUTS="$PWD/acl-style-files-master//:" \
BIBINPUTS="$PWD//:" \
BSTINPUTS="$PWD/acl-style-files-master//:" \
/Library/TeX/texbin/latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/icem-draft-build docs/paper_woah_2026/draft.tex
```

No-PII robustness command:

```bash
KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
micromamba run -n icem-research python scripts/run_icem_experiment.py \
  --dataset hatexplain \
  --sample-size 5000 \
  --seed 17 \
  --classifier Hate-speech-CNERG/dehatebert-mono-english \
  --classifier-batch-size 64 \
  --classifier-device auto \
  --importance-top-k 5 \
  --window-radius 2 \
  --manual-review-sample-size 0 \
  --no-inject-synthetic-pii \
  --output-dir docs/paper_woah_2026/results/015_hatexplain_dehatebert_5000_no_pii \
  --full-output-dir data/outputs \
  --overwrite
```

The 2026-06-22 residual recomputation read `release_rows.jsonl` for:

- `data/outputs/005_hatexplain_dehatebert_5000/release_rows.jsonl`
- `data/outputs/014_hatexplain_unbiased_selector_5000/release_rows.jsonl`
- `data/outputs/003_hatecheck_dehatebert_full/release_rows.jsonl`

It matched the saved `variant_metrics.csv` row counts, injected direct/quasi
residual counts, deterministic detector counts, and placeholder-only counts for
all variants in those runs.

## Zotero Source Paper Audit

For the single de-duplicated claim--citation audit with direct paper links and
Zotero PDF/full-text recheck results, see `citation_claim_audit.md`.

Local Zotero collection checked: `All Articles > PhD > NLP` (`WQW2X9VN`).
Fresh recheck on 2026-06-23: the current `draft.tex` cite commands resolve to
the BibTeX keys in `references.bib`. The canonical audit now keeps the Borkan
paper claim separate from the `GoogleCivil_commentsDatasets2024` dataset-card
claim, and the Aluru background citation separate from the DeHateBERT model-card
citation, with Zotero item/PDF/snapshot links in one place.

## Claim Ledger

| ID | Claim | Evidence | Status | Boundary |
| --- | --- | --- | --- | --- |
| C01 | HateXplain provides class labels, target communities, and human rationales. | Zotero item `VDLX42F5`, PDF `2QUG5XCE`; primary source: https://ojs.aaai.org/index.php/AAAI/article/view/17745; local `dataset_summary.json` for the sampled rows. | Supported. | Do not claim our reduced texts preserve human rationales as judged by humans; our rationale overlap is automatic. |
| C02 | HateCheck is a suite of 29 functionality tests for hate-speech detection. | Zotero item `IGTGWYTC`, PDF `3J2VGF37`; primary source: https://aclanthology.org/2021.acl-long.4/. | Supported. | Our use is a context stress test; DeHateBERT raw F1 is low on HateCheck, so do not present it as strong classifier utility. |
| C03 | Borkan et al. support an online-comment toxicity and identity-reference/unintended-bias framing; the `GoogleCivil_commentsDatasets2024` dataset card supports the exact `google/civil_comments` Hugging Face release and its relationship to the Jigsaw Unintended Bias/Civil Comments Kaggle data. | Zotero item `YZJTP8Q3`, PDF `SY7G822U`; Borkan et al., WWW '19 Companion, DOI `10.1145/3308560.3317593`; Zotero item `VALGPLFT`, snapshot `S3AP8CKV`; dataset card https://huggingface.co/datasets/google/civil_comments; cached card `data/hf_cache/hub/datasets--google--civil_comments/blobs/92511dd19fd36e335c9869acb33a35f1449d5a73`. | Supported with precise wording. | Cite Borkan for peer-reviewed unintended-bias metrics and identity-reference framing. Cite the dataset card for the exact Hugging Face/Kaggle/Civil Comments dataset identity. Our local Civil Comments run is out-of-domain for DeHateBERT and uses lexical target proxies, not gold identity labels from the full challenge schema. |
| C04 | Text anonymization/privacy-preserving NLP should pair utility measurement with explicit privacy-risk measures. | Zotero PDFs for Lison `WMFJFFZ7`, TAB `GXFL4C6Q`, Coavoux `5GMX8ED2`, Ren `SQJNI532`, and Sousa/Kern `KNNWP9WW`; primary pages: https://aclanthology.org/2021.acl-long.323/, https://aclanthology.org/2022.cl-4.19/, https://aclanthology.org/D18-1001/, https://aclanthology.org/2025.findings-ijcnlp.94/, https://link.springer.com/article/10.1007/s10462-022-10204-6. | Supported. | Our privacy metric is a residual-span proxy, not formal disclosure-risk measurement. |
| C05 | Pseudonymization/anonymization choices can alter downstream NLP behavior. | Zotero item `AT4AMEZJ`, PDF `TNRK8YZP`; primary source: https://aclanthology.org/2023.trustnlp-1.20/. | Supported. | Use as motivation only; our method is not pseudonymization. |
| C06 | The main selector/evaluator is DeHateBERT. | Saved run configs identify `Hate-speech-CNERG/dehatebert-mono-english`; Zotero model-card item `2SF473WY`, snapshot `ZH6F7FUP`; model card: https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english; Aluru et al. Zotero item `RXLUDZRP`, PDF `X2RVLU8V`, source https://link.springer.com/chapter/10.1007/978-3-030-67670-4_26. | Supported with revised wording. | Use saved configs/model card for the exact model identifier and Aluru for classifier-family background. Do not claim DeHateBERT is trained for this release task. |
| C07 | The independent evaluator is a Detoxify toxicity model, not used for DeHateBERT token selection. | Zotero item `HYG26S62`; Zenodo DOI `10.5281/zenodo.7925667`; model card: https://huggingface.co/unitary/unbiased-toxic-roberta; project: https://github.com/unitaryai/detoxify; run `results/007_hatexplain_dehatebert_release_unbiased_eval/config.json`. | Supported. | It is a toxicity model, not a hate-specific gold evaluator. |
| C08 | Main run uses 5,000 HateXplain examples with synthetic PII injection and DeHateBERT. | `results/005_hatexplain_dehatebert_5000/config.json`, `dataset_summary.json`, `run_manifest.json`. | Supported. | The sampled split is recorded as `train`; do not frame this as held-out model generalization or SOTA classification. |
| C09 | Main I-CEM result: F1 .697, flip 13.4%, retained tokens 49.7%, PII residual 0.0%, target 95.3%, stance 61.9%. | `results/005_hatexplain_dehatebert_5000/variant_metrics.csv`, `variant_metrics.json`; residual recomputation from `data/outputs/005_hatexplain_dehatebert_5000/release_rows.jsonl`. | Supported. | PII residual is for recorded injected spans plus deterministic detector counts. |
| C10 | Importance-only preserves little context: target 54.5%, stance 4.5%, retained tokens 11.7%, flip 26.7%. | `results/005_hatexplain_dehatebert_5000/variant_metrics.csv`. | Supported. | Automatic cue preservation only; do not call this human context sufficiency. |
| C11 | Fixed windows have higher main F1 than I-CEM but worse flip/context tradeoff. | `results/005_hatexplain_dehatebert_5000/variant_metrics.csv`: window F1 .716, flip 18.0%, target 72.2%, stance 24.6%; I-CEM F1 .697, flip 13.4%, target 95.3%, stance 61.9%. | Supported. | Do not imply I-CEM wins every metric. |
| C12 | Under the independent toxicity evaluator, I-CEM roughly matches fixed-window F1 while preserving more target/stance context. | `results/007_hatexplain_dehatebert_release_unbiased_eval/variant_metrics.csv`: I-CEM F1 .694, window F1 .693; target/stance 95.3/61.9 vs. 72.2/24.6. | Supported. | This reduces same-model evaluation concern; it does not prove model-independent generality. |
| C13 | HateCheck context comparison: against fixed windows, I-CEM improves target in 23/26 groups, stance-harm in 16/22, negation in 22/29, quotation in 29/29, and lowers flips in 17/29. | `results/003_hatecheck_dehatebert_full/context_breakdown.csv`; regenerated `hatecheck_context_summary.md`. | Supported. | Higher retained-token exposure is the cost; report it. |
| C14 | Toxicity-selector transfer run preserves the same qualitative pattern. | `results/014_hatexplain_unbiased_selector_5000/variant_metrics.csv`; `model_transfer_summary.md`. I-CEM: F1 .716, flip 13.5%, tokens 50.9%, target 97.0%, stance 63.9%; window: F1 .723, flip 16.6%, tokens 41.4%, target 80.6%, stance 28.4%. | Supported. | Robustness note only; retained tokens exceed the main 50% threshold and the selector is not hate-specific. |
| C15 | `top_k=5`, `window_radius=2` is selected as a balanced operating point on the compression--context curve traced by the tested grid (runs 008--012), where context preservation has largely saturated while token exposure is still in the lower half of the tested range; it is not claimed as the unique optimum of a fixed acceptance rule. | `selector_tradeoff_summary.csv`; regenerated `selector_tradeoff_summary.md`; runs 008--012 `main_table.md`. | Supported. | Say "tested grid", not "all possible settings". Do not reintroduce the six-threshold "only setting that passes" framing; present it as a curator-tunable operating point. |
| C24 | At a matched token budget, a fixed importance window still recovers far less target/stance context than I-CEM. The radius-3 fixed window (run 012) releases 48.8% of tokens, nearly matching I-CEM's 49.7%, yet reaches only 77.2% target / 33.8% stance vs. I-CEM's 95.3% / 61.9%, with a higher flip rate (15.0% vs. 13.4%). | `results/012_hatexplain_dehatebert_radius3_5000/main_table.md` (`importance_window` row); `results/005_hatexplain_dehatebert_5000/main_table.md` (`icem_context` row). | Supported. | The matched comparison uses run 012's `importance_window` variant (r=3), not its `icem_context` variant. State that the advantage is which tokens (relation-anchored spans), not more tokens. |
| C16 | I-CEM masks all recorded injected direct/quasi identifiers in the main controlled run. | `variant_metrics.csv` and residual recomputation from `release_rows.jsonl`: direct 0/5009 residual, quasi 0/5002 residual, detected direct/quasi 0/0 for I-CEM. | Supported. | Do not say all real PII is removed. |
| C17 | The system is deterministic and auditable. | Local implementation in `scripts/icem/experiment.py`, `scripts/icem/metrics.py`, `scripts/icem/release_policy.py`, and saved configs/manifests. | Supported for this implementation. | Classifier scores still depend on model/runtime; do not imply legal auditability. |
| C18 | A 30-row author manual audit is consistent with the context trend (I-CEM context sufficient in 21/30, fixed windows 17/30, importance-only 7/30). | `data/audits/author_sanity_audit_reviewed.csv`; `author_sanity_audit_summary.md`; `author_sanity_audit_summary.json`. | **Removed from draft.** | The author audit is no longer cited in `draft.tex`; it is superseded by the gold human-rationale overlap (C26), which is an external human signal without single-author circularity. The audit files remain as local artifacts only. Do not reintroduce the 21/17/7 counts into the paper. |
| C26 | I-CEM retains far more of HateXplain's gold human rationale tokens than importance-only, and on par with a token-matched fixed window. | Four-seed mean$\pm$std of `rationale_overlap` on **no-PII** runs `015`/`019`/`020`/`021`: importance-only $36.9\pm0.2$\%, fixed window (r=2) $68.8\pm0.5$\%, I-CEM $75.7\pm0.5$\%. Matched-budget (seed 17): run `022` `importance_window` (r=3) 75.1\% at 59.7\% tokens vs. run `015` `icem_context` 75.3\% at 57.5\% tokens. `scripts/summarize_seed_variance.py` -> `rationale_overlap_summary.md`. | Supported. | **Use no-PII runs only.** `with_replaced_text` does not remap `rationale_token_mask`, so PII injection (prefix templates) misaligns the mask; rationale overlap on injected runs (005/012/016--018) is invalid and must not be cited. At matched budget I-CEM is comparable to, not ahead of, the window on raw human-flagged tokens; do not claim it beats a matched window on rationale. The distinctive I-CEM advantage is stance (C24/proxies), which rationales do not annotate. |
| C27 | The context-preservation result replicates on a second dataset (SemEval-2021 Toxic Spans Detection, CC0) with a different human-annotation scheme. | Runs `023` (r=2) / `024` (r=3, token-matched), 5,000 rows, seed 17, no PII, DeHateBERT. I-CEM stance--harm 64.2\% vs. fixed window (r=2) 22.5\% / (r=3) 30.1\% / importance-only 1.6\%; gold-toxic-span overlap I-CEM 79.2\% vs. window(r=3) 81.1\% / importance-only 65.7\%. `results/023_toxic_spans_dehatebert_5000/`, `results/024_toxic_spans_radius3_5000/`. | Supported. | F1+ is low (DeHateBERT out of domain on toxicity); report as a context/rationale replication, not a utility result. The stance advantage transfers and holds at matched budget; raw toxic-span overlap is comparable between I-CEM and matched windows. Toxic spans = gold human toxic tokens, not PII spans. |
| C19 | Token importance is occlusion-based: `Delta_i = s(x)-s(x_-i)`, with non-identifier eligible anchors, `min_delta=.02`, `top_k=5`, a 30% anchor cap, and fallback to the strongest eligible token. | `scripts/icem/importance.py`; `results/005_hatexplain_dehatebert_5000/config.json`. | Supported. | Importance is classifier evidence, not a human rationale. |
| C20 | I-CEM context expansion uses radius-2 evidence windows and target/harm and stance/harm joins within eight tokens in the main setting. | `scripts/icem/context_rules.py`, `scripts/icem/release_policy.py`; `results/005_hatexplain_dehatebert_5000/config.json`. | Supported. | These are deterministic lexical rules, not learned discourse parsing. |
| C21 | The I-CEM context-recovery pattern persists without synthetic PII injection. | `results/015_hatexplain_dehatebert_5000_no_pii/variant_metrics.csv`; `no_pii_robustness_summary.md`. I-CEM: F1 .730, flip 10.1%, tokens 57.5%, target 93.8%, stance 81.4%; window: F1 .726, flip 12.2%, tokens 49.7%, target 71.9%, stance 37.4%. | Supported. | Robustness check only; it does not provide controlled gold PII residual measurement. |
| C22 | The paper reports descriptive fixed-run comparisons, not statistical replicability analysis across multiple independently sampled datasets. | Draft setup/results describe fixed saved runs; Zotero item `MA4A3LQH`, PDF `FKYLW9BX`, and primary source https://aclanthology.org/Q17-1033/ support replicability analysis as a distinct NLP evaluation concern. | Supported as a limitation. | Do not claim statistical significance, multiple-dataset replicability, or generalization beyond the saved runs. |
| C23 | HateXplain, HateCheck, and the `google/civil_comments` Civil Comments mirror do not provide gold PII spans for residual-span measurement in our release setting. | Loader schemas in `scripts/icem/datasets.py`; local HateXplain raw cache fields are `post_id`, `annotators`, `post_tokens`, and `rationales`; Hugging Face builder features for `Paul/hatecheck` are functionality/test-case/label/target metadata; Zotero item `VALGPLFT` plus cached `google/civil_comments` card and dataset info expose `text` plus toxicity/subtype score columns only. | Supported. | HateXplain provides rationales and target labels; HateCheck provides functionality/target labels; Civil Comments/Jigsaw includes toxicity and, in fuller releases, identity-reference labels. These are not PII span annotations. |

| C25 | I-CEM's target/stance context gains over the fixed window are stable across sampling seeds and far exceed the across-seed standard deviation, while positive-class F1 differences are small. | Runs `005` (seed 17), `016` (seed 7), `017` (seed 23), `018` (seed 41) `variant_metrics.json`; aggregated by `scripts/summarize_seed_variance.py` into `seed_variance_summary.md`. Mean$\pm$std: I-CEM target $95.1\pm0.4$, stance $61.7\pm1.0$; window(r=2) target $72.8\pm0.8$, stance $24.4\pm0.7$; F1 within ${\sim}0.015$. | Supported. | This is seed variance on one sampled dataset, not multi-dataset statistical replicability (see C22). Report mean$\pm$std, not significance tests. The window's small consistent F1 edge under DeHateBERT disappears under the independent evaluator. |

| C28 | An independent, lexicon-free LLM stance judge corroborates the stance proxy's direction but not its matched-budget magnitude. On 228 no-PII HateXplain rows whose full text reads as a non-assertion stance, stance is preserved in 1.3% (importance-only), 26.3% (window r=2), 42.1% (window r=3 matched), 49.1% (I-CEM). I-CEM beats importance-only and the r=2 window at p<1e-4; I-CEM vs the token-matched r=3 window is 49.1 vs 42.1, McNemar p=0.052 (not significant). | `results/025_stance_llm_judge/summary.json`, `stance_judge_summary.md`; `scripts/run_stance_llm_judge.py`; judge `qwen/qwen3.5-9b` via LM Studio on LAN (no external disclosure); releases from runs `015` (importance-only/window r2/I-CEM) and `022` (window r3). | Supported. | The judge grounds the **large** importance-only collapse, not the matched-window edge. Do not state the matched-budget stance advantage as significant or as ~1.8x; the grounded gap is ~1.17x and n.s. The lexical proxy overstates it. Judge is not a human study. |

| C29 | An independent, lexicon-free LLM judge shows evidence-only release corrupts the hate label and target, and I-CEM significantly reduces this vs a token-matched window. On 963 context-sensitive no-PII HateXplain rows: hate-label fidelity (excerpt vs full-text judge) 76.9% (importance-only) / 84.9% (window r2) / 87.7% (window r3 matched) / 89.7% (I-CEM); of the 200 full-text-non-hateful rows, the fraction the excerpt flips to hateful is 53.0 / 34.5 / 28.0 / 24.5%; target recovery 35.5 / 52.1 / 58.2 / 67.3% (n=670). I-CEM vs matched window: HateFid McNemar p=0.04, TgtRec p<1e-4; both vs importance-only p<1e-4. | `results/026_label_recovery_judge/summary.json`, `label_recovery_summary.md`; `scripts/run_label_recovery_judge.py`, `scripts/analyze_label_recovery.py`; judge `qwen/qwen3.5-9b` via LM Studio on LAN; releases from runs `015`+`022`. | Supported. | This is the headline grounded result: the harm (counterspeech read as hate, target lost) and I-CEM's significant edge over a matched window on label fidelity + target. Do not overstate hate-fidelity gap (90 vs 88 is significant but small); the large, robust gaps are vs importance-only and on CS->Hate / target. |
| C30 | On the exposure-vs-stance tradeoff, I-CEM is Pareto-dominant over the fixed-window family. Independent stance judge across window radii (228 rows): r0/importance-only 14.3% tok/1.3% stance, r1 35.5%/11.0%, r2 49.7%/26.3%, r3 59.7%/42.1%; I-CEM 57.5%/49.1% = 10.5 pts above the window frontier interpolated at 57.5% tokens (38.6%). | `results/025_stance_llm_judge/pareto.json`; `scripts/build_pareto_stance.py`; window r1 releases from run `027_hatexplain_nopii_radius1_5000`. | Supported. | Frontier claim rescues the stance story the single matched point (C28, p=0.05) could not. Report as Pareto dominance over the tested radii, not all settings. |
| C31 | The label/target-recovery result (C29) replicates across independent judges from multiple vendors. Three further judges under a tool-calling protocol on a deterministic 400-row subsample (same rows each judge): qwen3.5-9b (tool control), gemma-4-26b-a4b-it (Google), gpt-oss-20b (OpenAI). Hate-label fidelity ImpOnly/Matched/I-CEM = 61.2/77.8/83.0 (qwen tool), 53.1/79.6/82.1 (gemma), 65.7/81.9/84.4 (gpt-oss); target recovery 50.3/67.5/73.2, 58.9/73.4/75.3, 56.6/69.5/75.9. Direction is unanimous (I-CEM ≥ matched window, importance-only worst, every judge, both metrics). I-CEM vs matched window: target recovery significant for 3/4 judges (qwen plain p<1e-4, qwen tool p=0.02, gpt-oss p=0.02; gemma p=0.43 n.s.); label fidelity significant for both qwen judges (p=0.04, p=0.01), directional/n.s. for gemma (0.25) and gpt-oss (0.24) at n=400. | `results/026_label_recovery_judge/by_model/{qwen__qwen3.5-9b,gemma-4-26b-a4b-it,openai__gpt-oss-20b}/summary.json`, `multi_judge_summary.md`; `scripts/run_label_recovery_judge.py` (ICEM_JUDGE_TOOLS=1), `scripts/aggregate_label_recovery_judges.py`, `scripts/overnight_judge_panel.sh`; all judges LM Studio on LAN. | Supported. | Robustness check answering single-judge dependence. State exactly: the corruption claim and the importance-only->context-recovery ordering replicate across vendors; I-CEM's edge over a *wide* matched window is robust in direction but only sometimes significant (target > fidelity). 400-row subsample (LM Studio serialises requests); the committed full-963 C29 carries the headline significance. Reasoning-tuned judges require the tool-calling protocol (plain short-answer lands in empty `content`). |

## Critique And Risks

- The strongest privacy claim is limited: I-CEM removes recorded synthetic
  direct/quasi identifiers and passes the deterministic residual detector. It
  does not establish anonymization, k-anonymity, differential privacy, or
  protection against real re-identification.
- Target and stance preservation are lexical/proxy metrics over cue inventories.
  They are useful for comparing release variants but are not human judgments.
- The lexical target/stance proxies are now corroborated by overlap with
  HateXplain's gold human rationales (C26), an external human signal that
  replaces the earlier single-author audit. Rationales mark label-relevance,
  not reviewer-judged context sufficiency, so this is corroboration, not a
  formal human usability study.
- The main result uses a 5,000-row HateXplain sample recorded as `train`; this
  is a release-protocol evaluation, not a new classifier benchmark.
- The independent evaluator and transfer-selector runs reduce circularity
  concerns, but both use a toxicity model whose task does not exactly match
  hate-speech labels.
- I-CEM intentionally releases more source tokens than fixed windows. The paper
  should present this as a tradeoff, not a free improvement.
- Civil Comments should remain a sensitivity check only because utility is weak
  with the DeHateBERT setup and the target/stance cues are lexical proxies.

## Safe Summary

The current evidence supports a conservative WOAH short-paper claim: for a
controlled harmful-speech release setting with synthetic identifier spans,
I-CEM gives a reproducible middle point between full-text masking and
importance-only release. It masks the recorded injected identifiers, retains
about half the source tokens, and preserves much more automatic target/stance
context than importance-only or fixed-window reductions --- corroborated by
higher overlap with HateXplain's gold human rationales --- with classifier
utility remaining in the same range. The headline grounded result (C29): an
independent lexicon-free LLM judge shows evidence-only release corrupts the label
(53% of non-hateful/counterspeech content misread as hateful, target recovered 35%),
and I-CEM significantly reduces this versus a token-matched fixed window (hate-label
fidelity p=0.04, target recovery p<1e-4), while being Pareto-dominant over the window
family on stance (C30). The corruption result and the importance-only->context
ordering replicate across independent judges from three vendors (C31); I-CEM's edge
over a wide matched window is robust in direction but only sometimes significant
(target recovery in 3/4 judges, label fidelity in the Qwen judges). The matched-budget
stance proxy multiple (C28) overstates the effect and is not significant at the single point. The evidence does not support claims of complete
anonymization, production PII detection, human usability, or state-of-the-art
hate-speech classification.
