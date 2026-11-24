# Calibration: restaurant.wants_human

Model `qwen3.8-27b`, prompt `a01cbde79d06`, 32 labelled cases, fitted temperature **T = 1.0** (2026-11-24).

| | accuracy | ECE | Brier |
|---|---|---|---|
| raw (T = 1) | 0.969 | 0.057 | 0.036 |
| calibrated, 5-fold cross-validated | 0.969 | 0.057 | 0.036 |

Risk-coverage (calibrated): the share of cases at or above each threshold and their accuracy.

| threshold | coverage | accuracy | n |
|---|---|---|---|
| 0.5 | 1.00 | 0.969 | 32 |
| 0.6 | 0.97 | 1.000 | 31 |
| 0.7 | 0.94 | 1.000 | 30 |
| 0.8 | 0.91 | 1.000 | 29 |
| 0.85 | 0.88 | 1.000 | 28 |
| 0.9 | 0.88 | 1.000 | 28 |
| 0.95 | 0.78 | 1.000 | 25 |
| 0.99 | 0.50 | 1.000 | 16 |

![Reliability diagram](restaurant.wants_human.svg)
