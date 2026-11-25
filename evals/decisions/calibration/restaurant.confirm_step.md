# Calibration: restaurant.confirm_step

Model `gemma-4-26b-a4b-it`, prompt `18af14a9b38b`, 26 labelled cases, fitted temperature **T = 1.0** (2026-11-25).

| | accuracy | ECE | Brier |
|---|---|---|---|
| raw (T = 1) | 1.000 | 0.015 | 0.012 |
| calibrated, 5-fold cross-validated | 1.000 | 0.015 | 0.012 |

Risk-coverage (calibrated): the share of cases at or above each threshold and their accuracy.

| threshold | coverage | accuracy | n |
|---|---|---|---|
| 0.5 | 1.00 | 1.000 | 26 |
| 0.6 | 0.96 | 1.000 | 25 |
| 0.7 | 0.96 | 1.000 | 25 |
| 0.8 | 0.96 | 1.000 | 25 |
| 0.85 | 0.96 | 1.000 | 25 |
| 0.9 | 0.96 | 1.000 | 25 |
| 0.95 | 0.96 | 1.000 | 25 |
| 0.99 | 0.96 | 1.000 | 25 |

Cases labelled *unclear* (graded by action, not label): 4.

![Reliability diagram](restaurant.confirm_step.svg)
