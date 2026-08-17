# Rationale overlap and multi-seed variance summary

Aggregated over 4 HateXplain runs (17, 7, 23, 41), 5,000 rows each, identical config except the sampling seed. Values are mean +/- sample std across seeds. F1+ is the positive-class F1; Flip/Tok/Target/Stance are percentages.

| Variant | F1+ | Flip% | Tok% | Target% | Stance% | Rationale% |
| --- | --- | --- | --- | --- | --- | --- |
| Importance only | 0.703 +/- 0.001 | 22.0 +/- 0.6 | 14.3 +/- 0.0 | 53.2 +/- 0.9 | 6.8 +/- 1.0 | 36.9 +/- 0.2 |
| Importance window (r=2) | 0.722 +/- 0.003 | 12.1 +/- 0.3 | 49.7 +/- 0.2 | 71.4 +/- 0.6 | 37.0 +/- 0.4 | 68.8 +/- 0.5 |
| I-CEM | 0.726 +/- 0.004 | 10.3 +/- 0.3 | 57.5 +/- 0.2 | 93.6 +/- 0.5 | 82.0 +/- 0.5 | 75.7 +/- 0.5 |

## Per-seed raw values

### Importance only

| Seed | F1+ | Flip% | Tok% | Target% | Stance% | Rationale% |
| --- | --- | --- | --- | --- | --- | --- |
| 17 | 0.705 | 22.4 | 14.3 | 53.4 | 7.5 | 37.0 |
| 7 | 0.703 | 22.3 | 14.3 | 53.9 | 7.4 | 36.8 |
| 23 | 0.704 | 22.1 | 14.3 | 53.7 | 5.4 | 36.6 |
| 41 | 0.702 | 21.1 | 14.3 | 52.0 | 6.9 | 37.2 |

### Importance window (r=2)

| Seed | F1+ | Flip% | Tok% | Target% | Stance% | Rationale% |
| --- | --- | --- | --- | --- | --- | --- |
| 17 | 0.726 | 12.2 | 49.7 | 71.9 | 37.4 | 68.3 |
| 7 | 0.721 | 12.0 | 49.6 | 71.8 | 37.4 | 68.9 |
| 23 | 0.719 | 12.5 | 49.9 | 71.4 | 36.5 | 69.4 |
| 41 | 0.723 | 11.9 | 49.6 | 70.5 | 36.9 | 68.5 |

### I-CEM

| Seed | F1+ | Flip% | Tok% | Target% | Stance% | Rationale% |
| --- | --- | --- | --- | --- | --- | --- |
| 17 | 0.730 | 10.1 | 57.5 | 93.8 | 81.4 | 75.3 |
| 7 | 0.725 | 10.2 | 57.4 | 94.1 | 81.7 | 75.6 |
| 23 | 0.721 | 10.7 | 57.9 | 93.5 | 82.4 | 76.4 |
| 41 | 0.728 | 10.3 | 57.4 | 93.0 | 82.5 | 75.4 |

