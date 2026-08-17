# HateCheck context summary

Generated from `results/003_hatecheck_dehatebert_full/context_breakdown.csv`
with `scripts/summarize_hatecheck_context.py`.

## Dataset-level results

| Variant | F1+ | Flip % | Tok % | Target % | Negation % | Quote % | Counter % | Target-harm % | Stance-harm % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| importance_only | 0.257 | 12.9 | 19.4 | 73.0 | 33.4 | 11.2 | 18.2 | 35.4 | 7.7 |
| importance_window | 0.293 | 8.5 | 57.7 | 88.7 | 72.0 | 33.9 | 81.8 | 76.2 | 37.9 |
| icem_context | 0.247 | 5.7 | 66.5 | 99.7 | 98.7 | 54.6 | 100.0 | 99.4 | 68.2 |

## Functionality-level comparison

Against `importance_only`, I-CEM improves target preservation in 26/26 functionality groups, stance-harm preservation in 22/22, negation preservation in 27/29, quotation preservation in 29/29, and lowers raw-decision flips in 26/29 groups.
Against `importance_window`, I-CEM improves target preservation in 23/26 functionality groups, stance-harm preservation in 16/22, negation preservation in 22/29, quotation preservation in 29/29, and lowers raw-decision flips in 17/29 groups.

## Largest stance gains over fixed windows

| Functionality | n | Window stance % | I-CEM stance % | Delta | Token cost | Flip delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| derog_impl_h | 140 | 0.0 | 75.0 | +75.0 | +8.8 | +0.7 |
| target_obj_nh | 65 | 25.0 | 100.0 | +75.0 | +6.1 | -1.5 |
| derog_neg_attrib_h | 140 | 27.3 | 81.8 | +54.5 | +12.6 | -1.4 |
| derog_dehum_h | 140 | 0.0 | 50.0 | +50.0 | +9.8 | -3.6 |
| target_group_nh | 62 | 50.0 | 100.0 | +50.0 | +6.9 | +1.6 |
| ref_subs_clause_h | 140 | 18.2 | 63.6 | +45.5 | +13.4 | -12.1 |
| phrase_opinion_h | 133 | 0.0 | 40.0 | +40.0 | +9.0 | -0.8 |
| profanity_h | 140 | 40.0 | 80.0 | +40.0 | +11.1 | -2.1 |

## Largest flip reductions over fixed windows

| Functionality | n | Window stance % | I-CEM stance % | Delta | Token cost | Flip delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ref_subs_clause_h | 140 | 18.2 | 63.6 | +45.5 | +13.4 | -12.1 |
| counter_quote_nh | 173 | 71.1 | 97.4 | +26.3 | +17.7 | -10.4 |
| counter_ref_nh | 141 | 44.4 | 74.1 | +29.6 | +11.2 | -9.2 |
| ref_subs_sent_h | 133 | 8.3 | 41.7 | +33.3 | +10.4 | -8.3 |
| derog_neg_emote_h | 140 | 25.0 | 25.0 | +0.0 | +7.1 | -7.1 |
| slur_homonym_nh | 30 | 8.3 | 41.7 | +33.3 | +9.0 | -6.7 |
| threat_norm_h | 140 | 50.0 | 50.0 | +0.0 | +9.4 | -4.3 |
| slur_h | 144 | 32.4 | 56.8 | +24.3 | +10.5 | -4.2 |

## Paper interpretation

HateCheck is a stress test for context rather than a strong classifier-utility
benchmark here: DeHateBERT has low positive F1 on raw HateCheck. The useful
claim is narrower and stronger: I-CEM recovers target, negation, quotation,
counter-speech, and stance context that importance-only and fixed-window
releases frequently drop. The cost is higher retained-token exposure than
fixed windows, which should be reported as the privacy-utility tradeoff.
