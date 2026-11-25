# Calibration: restaurant.instruction_in_reply

Model `qwen3.8-27b`, prompt `96494bbd5e56`, 68 labelled cases, fitted temperature **T = 1.0** (2026-11-25).

| | accuracy | ECE | Brier |
|---|---|---|---|
| raw (T = 1) | 1.000 | 0.018 | 0.008 |
| calibrated, 5-fold cross-validated | 1.000 | 0.018 | 0.008 |

Risk-coverage (calibrated): the share of cases at or above each threshold and their accuracy.

| threshold | coverage | accuracy | n |
|---|---|---|---|
| 0.5 | 1.00 | 1.000 | 68 |
| 0.6 | 1.00 | 1.000 | 68 |
| 0.7 | 0.99 | 1.000 | 67 |
| 0.8 | 0.97 | 1.000 | 66 |
| 0.85 | 0.96 | 1.000 | 65 |
| 0.9 | 0.94 | 1.000 | 64 |
| 0.95 | 0.94 | 1.000 | 64 |
| 0.99 | 0.85 | 1.000 | 58 |

![Reliability diagram](restaurant.instruction_in_reply.svg)
