# Calibration: restaurant.confirm_booking

Model `gemma-4-26b-a4b-it`, prompt `08de156eb30a`, 26 labelled cases, fitted temperature **T = 4.989** (2026-11-23).

| | accuracy | ECE | Brier |
|---|---|---|---|
| raw (T = 1) | 0.923 | 0.077 | 0.153 |
| calibrated, 5-fold cross-validated | 0.923 | 0.073 | 0.108 |

Risk-coverage (calibrated): the share of cases at or above each threshold and their accuracy.

| threshold | coverage | accuracy | n |
|---|---|---|---|
| 0.5 | 1.00 | 0.923 | 26 |
| 0.6 | 1.00 | 0.923 | 26 |
| 0.7 | 1.00 | 0.923 | 26 |
| 0.8 | 0.92 | 0.958 | 24 |
| 0.85 | 0.88 | 1.000 | 23 |
| 0.9 | 0.88 | 1.000 | 23 |
| 0.95 | 0.85 | 1.000 | 22 |
| 0.99 | 0.15 | 1.000 | 4 |

Cases labelled *unclear* (graded by action, not label): 6.

![Reliability diagram](restaurant.confirm_booking.svg)
