# Calibration: restaurant.confirm_booking

Model `qwen3.8-27b`, prompt `3c6207610a11`, 32 labelled cases, fitted temperature **T = 1.0** (2026-11-25).

| | accuracy | ECE | Brier |
|---|---|---|---|
| raw (T = 1) | 1.000 | 0.013 | 0.004 |
| calibrated, 5-fold cross-validated | 1.000 | 0.013 | 0.004 |

Risk-coverage (calibrated): the share of cases at or above each threshold and their accuracy.

| threshold | coverage | accuracy | n |
|---|---|---|---|
| 0.5 | 1.00 | 1.000 | 32 |
| 0.6 | 1.00 | 1.000 | 32 |
| 0.7 | 1.00 | 1.000 | 32 |
| 0.8 | 0.97 | 1.000 | 31 |
| 0.85 | 0.97 | 1.000 | 31 |
| 0.9 | 0.97 | 1.000 | 31 |
| 0.95 | 0.94 | 1.000 | 30 |
| 0.99 | 0.84 | 1.000 | 27 |

Cases labelled *unclear* (graded by action, not label): 6.

![Reliability diagram](restaurant.confirm_booking.svg)
