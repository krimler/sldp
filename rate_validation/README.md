# SLDP rate validation

Reproduces the empirical structure-MSE rate validation for Structural Local
Differential Privacy (SLDP). The simulation checks that the debiased
randomized-response estimator for SLDP-Report attains the theoretical rate
`Theta(m / (n * eps_s^2))` for structure estimation — i.e. slope `-1` on a
log-log plot of mean squared error against the number of users `n`.

## Files

- `sldp_rate_validation.py` — simulation driver (NumPy + Matplotlib).
- `rate_validation.pdf` — log-log MSE vs `n`, one curve per `eps_s`.
- `rate_validation.csv` — raw per-cell results.

## Usage

```
python sldp_rate_validation.py
```

Requires Python 3.9+, NumPy, and Matplotlib.

## Output

For each privacy budget `eps_s` in `{0.5, 1.0, 2.0}`, the fitted log-log slope is
within tolerance of the theoretical `-1`, confirming the `Theta(m/(n eps_s^2))`
rate. Results are written to `rate_validation.pdf` and `rate_validation.csv`.
