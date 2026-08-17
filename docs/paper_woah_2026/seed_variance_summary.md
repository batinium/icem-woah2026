# Multi-seed variance summary

Aggregated over 4 HateXplain runs (17, 7, 23, 41), 5,000 rows each, identical config except the sampling seed. Values are mean +/- sample std across seeds. F1+ is the positive-class F1; Flip/Tok/Target/Stance are percentages.

| Variant | F1+ | Flip% | Tok% | Target% | Stance% |
| --- | --- | --- | --- | --- | --- |
| Importance only | 0.704 +/- 0.002 | 25.8 +/- 0.6 | 11.8 +/- 0.1 | 54.8 +/- 0.3 | 4.0 +/- 0.3 |
| Importance window (r=2) | 0.712 +/- 0.005 | 17.5 +/- 0.7 | 39.8 +/- 0.1 | 72.8 +/- 0.8 | 24.4 +/- 0.7 |
| I-CEM | 0.698 +/- 0.002 | 13.2 +/- 0.4 | 49.9 +/- 0.1 | 95.1 +/- 0.4 | 61.7 +/- 1.0 |

## Per-seed raw values

### Importance only

| Seed | F1+ | Flip% | Tok% | Target% | Stance% |
| --- | --- | --- | --- | --- | --- |
| 17 | 0.704 | 26.7 | 11.7 | 54.5 | 4.5 |
| 7 | 0.705 | 25.6 | 11.8 | 55.2 | 4.0 |
| 23 | 0.701 | 25.4 | 11.7 | 54.7 | 3.9 |
| 41 | 0.705 | 25.4 | 11.8 | 54.8 | 3.8 |

### Importance window (r=2)

| Seed | F1+ | Flip% | Tok% | Target% | Stance% |
| --- | --- | --- | --- | --- | --- |
| 17 | 0.716 | 18.0 | 39.8 | 72.2 | 24.6 |
| 7 | 0.715 | 17.0 | 39.9 | 73.8 | 25.1 |
| 23 | 0.714 | 18.1 | 39.6 | 73.0 | 24.5 |
| 41 | 0.705 | 16.8 | 39.9 | 72.2 | 23.6 |

### I-CEM

| Seed | F1+ | Flip% | Tok% | Target% | Stance% |
| --- | --- | --- | --- | --- | --- |
| 17 | 0.697 | 13.4 | 49.7 | 95.3 | 61.9 |
| 7 | 0.698 | 13.0 | 50.0 | 95.3 | 60.3 |
| 23 | 0.696 | 13.6 | 49.9 | 95.1 | 61.7 |
| 41 | 0.701 | 12.9 | 49.8 | 94.5 | 62.7 |

