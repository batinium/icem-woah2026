# Independent LLM Label/Target Recoverability Judge (run 026)

Tests the **data-validity harm** of evidence-minimized release: does the reduced
excerpt make an independent reader misjudge whether content is hateful and who is
targeted? Same lexicon-free local judge as run 025 (`qwen/qwen3.5-9b`, local endpoint).

## Setup

- 963 no-PII HateXplain rows (seed 17) where a stance and harm cue co-occur (the
  context-sensitive subset). Releases from run `015` (importance-only, window r=2,
  I-CEM) and run `022` (token-matched window r=3).
- Per excerpt, two independent questions: (1) is this hateful/offensive toward a
  group? YES/NO/UNCLEAR; (2) which group is targeted? The judge never sees I-CEM's
  cue lexicon.
- Reference = the judge's reading of the **full raw text**. A release corrupts the
  label if its excerpt flips the hate decision or loses the target the full text
  conveys.
- Reproduce: `scripts/run_label_recovery_judge.py`, `scripts/analyze_label_recovery.py`.

## Hate-label fidelity (excerpt decision == full-text decision; n=963)

| Variant | Tokens | Fidelity | 95% CI |
| --- | ---: | ---: | --- |
| Importance only | 14.3% | 76.9% | [74.2, 79.6] |
| Window (r=2) | 49.7% | 84.9% | [82.6, 87.1] |
| Window (r=3, matched) | 59.7% | 87.7% | [85.7, 89.8] |
| **I-CEM** | 57.5% | **89.7%** | [87.7, 91.7] |

McNemar: I-CEM vs importance-only **p < 1e-4**; **I-CEM vs token-matched window
p = 0.040 (significant)**.

## The harm: counterspeech / quoted content misread as hate

On the 200 rows the judge reads as **not hateful** in full text (counterspeech,
negation, quotation), the fraction the *excerpt* flips to **hateful**:

| Variant | NO -> YES (non-hate read as hate) |
| --- | ---: |
| Importance only | **53.0%** |
| Window (r=2) | 34.5% |
| Window (r=3, matched) | 28.0% |
| **I-CEM** | **24.5%** |

Naive importance-only release makes an independent reader misclassify **over half**
of non-hateful content as hateful. I-CEM less than halves that rate.

## Target recovery (excerpt target overlaps full-text target; n=670)

| Variant | Recovered |
| --- | ---: |
| Importance only | 35.5% [31.9, 39.1] |
| Window (r=2) | 52.1% |
| Window (r=3, matched) | 58.2% [54.5, 61.9] |
| **I-CEM** | **67.3%** [63.7, 70.7] |

McNemar: I-CEM vs importance-only **p < 1e-4**; **I-CEM vs token-matched window
p < 1e-4** (103 vs 42 discordant).

## Reading

1. **Concrete harm, independently grounded.** Importance-only release (keep only
   classifier-evidence tokens) corrupts the label: 53% of non-hateful content reads
   as hateful and only 35% of targets survive. A dataset minimized that way would
   systematically relabel counterspeech and quotation as hate.
2. **I-CEM significantly beats a token-matched window** on both label fidelity
   (89.7 vs 87.7, p=0.04) and target recovery (67.3 vs 58.2, p<1e-4) -- the
   advantage that the stance proxy could not establish on its own (run 025,
   p=0.052) shows up clearly on the label-corruption and target measures that
   actually matter for dataset validity.
