# Independent LLM Stance Judge (run 025)

External, lexicon-free grounding for the Stance proxy, addressing the circularity
that Stance is otherwise scored with the same cue lexicons I-CEM expands on.

## Setup

- Source releases: no-PII HateXplain, seed 17 (run `015` for importance-only,
  fixed window r=2, and I-CEM; run `022` for the token-matched fixed window r=3).
  Same 5,000-row draw, so rows align across variants by `row_id`.
- Judge: `qwen/qwen3.5-9b`, local OpenAI-compatible endpoint, temperature 0.
- The judge never sees I-CEM's cue lexicon. It reads each text and labels the
  author's stance toward the harmful content as one of
  `ASSERT / NEGATE / QUOTE / OPPOSE / UNCLEAR`.
- Candidates: the 963 rows where the lexical proxy fires (a stance cue and a harm
  cue both present in the source). The judge then defines the denominator
  independently: the **228** rows whose *full raw text* reads as a non-assertion
  stance (NEGATE/QUOTE/OPPOSE) — i.e. cases where context genuinely changes the
  reading.
- Metric: fraction of those 228 rows whose release excerpt *still* reads as a
  non-assertion stance to the judge (stance preserved, not collapsed to a bare
  assertion or to UNCLEAR).
- Reproduce: `scripts/run_stance_llm_judge.py` (writes `checkpoint.jsonl`,
  `summary.json`).

## Result (228 rows, paired)

| Variant | Retained tokens | Stance preserved | 95% CI |
| --- | ---: | ---: | --- |
| Importance only | 14.3% | 1.3% | [0.0, 3.1] |
| Fixed window (r=2) | 49.7% | 26.3% | [20.6, 32.0] |
| Fixed window (r=3, token-matched) | 59.7% | 42.1% | [36.0, 48.7] |
| I-CEM | 57.5% | 49.1% | [42.5, 55.7] |

McNemar exact, two-sided:

- I-CEM vs importance-only: 109 discordant, all favor I-CEM, **p < 1e-4**.
- I-CEM vs fixed window r=2: 58 vs 6 discordant, **p < 1e-4** (but window r=2 uses fewer tokens).
- Token-matched window r=3 vs window r=2: 43 vs 7, **p < 1e-4**.
- **I-CEM vs token-matched window r=3: 38 vs 22 discordant, p = 0.052 — not significant at .05.**

## Reading

1. **The circularity is broken for the headline message.** An independent,
   lexicon-free judge confirms that importance-only release destroys stance
   (1.3% preserved) and that adding context recovers it. This is large and highly
   significant; it no longer rests on the self-referential proxy.
2. **The proxy overstates I-CEM's edge over a token-matched window.** The lexical
   proxy reports 61.9% vs 33.8% at matched budget (1.8x). The independent judge
   reports 49.1% vs 42.1% (~1.17x), and the difference is **not statistically
   significant** (p = 0.052). I-CEM retains its advantage in direction and at a
   slightly lower token budget, but the matched-budget stance win is not a grounded
   claim. This is consistent with the gold-rationale tie at matched budget (C26).
3. Even I-CEM loses about half of stance relations (49% preserved). "Preserves
   stance context" is a relative claim against importance-only, not an absolute
   guarantee.
