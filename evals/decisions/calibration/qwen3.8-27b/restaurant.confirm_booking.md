# Calibration: restaurant.confirm_booking

Model `qwen3.8-27b`, prompt `08de156eb30a`, 26 labelled cases, fitted temperature **T = 1.0** (2026-11-23).

| | accuracy | ECE | Brier |
|---|---|---|---|
| raw (T = 1) | 1.000 | 0.004 | 0.000 |
| calibrated, 5-fold cross-validated | 1.000 | 0.004 | 0.000 |

Risk-coverage (calibrated): the share of cases at or above each threshold and their accuracy.

| threshold | coverage | accuracy | n |
|---|---|---|---|
| 0.5 | 1.00 | 1.000 | 26 |
| 0.6 | 1.00 | 1.000 | 26 |
| 0.7 | 1.00 | 1.000 | 26 |
| 0.8 | 1.00 | 1.000 | 26 |
| 0.85 | 1.00 | 1.000 | 26 |
| 0.9 | 1.00 | 1.000 | 26 |
| 0.95 | 0.96 | 1.000 | 25 |
| 0.99 | 0.88 | 1.000 | 23 |

Cases labelled *unclear* (graded by action, not label): 6.

![Reliability diagram](restaurant.confirm_booking.svg)
