# Calibration: restaurant.instruction_in_reply

Model `qwen3.8-27b`, prompt `96494bbd5e56`, 32 labelled cases, fitted temperature **T = 1.0** (2026-11-23).

| | accuracy | ECE | Brier |
|---|---|---|---|
| raw (T = 1) | 1.000 | 0.024 | 0.013 |
| calibrated, 5-fold cross-validated | 1.000 | 0.024 | 0.013 |

Risk-coverage (calibrated): the share of cases at or above each threshold and their accuracy.

| threshold | coverage | accuracy | n |
|---|---|---|---|
| 0.5 | 1.00 | 1.000 | 32 |
| 0.6 | 1.00 | 1.000 | 32 |
| 0.7 | 0.97 | 1.000 | 31 |
| 0.8 | 0.94 | 1.000 | 30 |
| 0.85 | 0.94 | 1.000 | 30 |
| 0.9 | 0.94 | 1.000 | 30 |
| 0.95 | 0.94 | 1.000 | 30 |
| 0.99 | 0.88 | 1.000 | 28 |

![Reliability diagram](restaurant.instruction_in_reply.svg)
