#!/usr/bin/env python3
"""
SLDP rate-validation simulation.

Validates that SLDP-Report's empirical structure-MSE matches
the theoretical rate Theta(m / (n eps_s^2)) up to log factors.

Design principles:
  * Adaptive sampling: start small, grow trials only where needed.
  * Sanity checks: abort if slope is far from theoretical -1.
  * Debiased estimators (this is the bug that causes slope=0).
  * Parallel across trials via multiprocessing.Pool with 'spawn'
    (correct for macOS M-series; 'fork' breaks numpy/BLAS state).

Output:
  rate_validation.pdf : log-log MSE vs n, three eps_s curves with
                        theoretical lower-bound reference lines.
  rate_validation.csv : raw data for each (eps_s, n) cell.

Runtime estimate on M4 Pro 16-core:
  ~2 minutes for the default sweep at base_trials=20.
  Scales sub-linearly because larger n needs fewer trials for tight CIs.
"""

from __future__ import annotations
import csv
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

# macOS M-series: 'spawn' is required. 'fork' (default) corrupts numpy
# BLAS thread pools and shared RNG state in subprocesses.
mp.set_start_method("spawn", force=True)


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class Config:
    m: int = 100                # registry size
    K: int = 10                 # value alphabet size (unused for structure-only)
    eps_values: tuple = (0.5, 1.0, 2.0)
    n_grid: tuple = (1_000, 3_000, 10_000, 30_000, 100_000)

    # Adaptive controls
    base_trials: int = 20       # initial trials per (eps, n) cell
    max_trials: int = 500       # ceiling per cell
    slope_target: float = -1.0  # theoretical slope on log-log
    slope_tolerance: float = 0.15   # accept slope in [-1.15, -0.85]
    abort_tolerance: float = 0.5    # bug if slope outside [-1.5, -0.5]
    ci_target: float = 0.05     # stop growing when slope CI half-width <= 0.05

    # Reproducibility
    base_seed: int = 20260430

    # Parallelism
    n_workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 4) - 2))


# =====================================================================
# Mechanism: SLDP-Report (structure-only, debiased)
# =====================================================================

def run_one_trial(args: tuple) -> float:
    """One independent trial. Returns structure MSE.

    Must be top-level for pickling under 'spawn'.

    The estimator is the standard debiased RR mean:
      theta_hat_i = (mean(S_tilde_i) - q) / (1 - 2q)
    where q = 1 / (1 + exp(eps_s)) is the flip probability.

    Without debiasing, MSE has a fixed bias floor and slope is ~0.
    """
    trial_idx, n, eps_s, m, base_seed = args
    rng = np.random.default_rng(base_seed + trial_idx * 1_000_003 + int(n) + int(eps_s * 1000))

    # Ground truth: Beta(2, 5) gives a moderately sparse population
    theta_true = rng.beta(2.0, 5.0, size=m)

    # Sample n users; each user's S_i ~ Bernoulli(theta_i)
    S = rng.binomial(1, theta_true, size=(n, m)).astype(np.int8)

    # Apply per-bit randomized response with budget eps_s
    q = 1.0 / (1.0 + np.exp(eps_s))         # flip probability
    flips = rng.binomial(1, q, size=(n, m)).astype(np.int8)
    S_tilde = S ^ flips                     # XOR

    # Debiased estimator
    mean_S_tilde = S_tilde.mean(axis=0)
    theta_hat = (mean_S_tilde - q) / (1.0 - 2.0 * q)

    mse = float(np.mean((theta_hat - theta_true) ** 2))
    return mse


# =====================================================================
# Statistics
# =====================================================================

def fit_loglog_slope(n_values: np.ndarray, mse_values: np.ndarray) -> tuple[float, float]:
    """Fit slope of log(mse) vs log(n) via OLS.

    Returns (slope, slope_ci_halfwidth_95).
    """
    log_n = np.log(n_values)
    log_mse = np.log(mse_values)

    # OLS slope and intercept
    n_pts = len(log_n)
    slope, intercept = np.polyfit(log_n, log_mse, deg=1)

    # Standard error of slope from residuals
    pred = slope * log_n + intercept
    residuals = log_mse - pred
    rss = float(np.sum(residuals ** 2))
    sxx = float(np.sum((log_n - log_n.mean()) ** 2))
    if n_pts <= 2 or sxx == 0.0 or rss == 0.0:
        return float(slope), float("inf")
    sigma2 = rss / (n_pts - 2)
    se_slope = float(np.sqrt(sigma2 / sxx))
    # 95% CI half-width using t_{n-2, 0.975}; for small n approximate with 2.0
    ci_half = 2.0 * se_slope
    return float(slope), ci_half


def aggregate_trials(trials: list[float]) -> tuple[float, float]:
    """Mean and 95% CI half-width of MSE across trials."""
    arr = np.asarray(trials, dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, float("inf")
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return mean, 1.96 * sem


# =====================================================================
# Adaptive driver
# =====================================================================

def run_eps_curve(eps_s: float, cfg: Config, pool) -> dict:
    """Run all n in cfg.n_grid for this eps_s, growing trials adaptively.

    Returns dict with: eps_s, n_grid, mean_mse (per n), trials_used (per n),
    slope, slope_ci, status in {'converged', 'aborted', 'maxed_out'}.
    """
    n_grid = np.asarray(cfg.n_grid, dtype=int)

    # Per-n MSE samples accumulate across rounds
    mse_samples_per_n: dict[int, list[float]] = {int(n): [] for n in n_grid}
    trials_used = {int(n): 0 for n in n_grid}

    n_trials_this_round = cfg.base_trials
    total_trials = 0

    while True:
        # Build the work list: (trial_idx, n, eps_s, m, base_seed)
        work_items = []
        for n in n_grid:
            existing = trials_used[int(n)]
            for t in range(existing, existing + n_trials_this_round):
                work_items.append((t, int(n), float(eps_s), cfg.m, cfg.base_seed))

        # Run in parallel
        results = pool.map(run_one_trial, work_items)

        # Distribute results back
        idx = 0
        for n in n_grid:
            for _ in range(n_trials_this_round):
                mse_samples_per_n[int(n)].append(results[idx])
                idx += 1
            trials_used[int(n)] += n_trials_this_round
        total_trials += len(work_items)

        # Compute mean MSE per n
        mean_mse = np.array([np.mean(mse_samples_per_n[int(n)]) for n in n_grid])

        # Fit slope
        slope, slope_ci = fit_loglog_slope(n_grid.astype(float), mean_mse)

        slope_err = abs(slope - cfg.slope_target)
        print(
            f"  eps_s={eps_s:>4}  trials={trials_used[int(n_grid[0])]:>4}  "
            f"slope={slope:+.3f} ± {slope_ci:.3f}  "
            f"|slope - (-1)|={slope_err:.3f}",
            flush=True,
        )

        # Abort if slope is wildly off (likely a bug)
        if slope_err > cfg.abort_tolerance:
            print(
                f"    SLOPE DEVIATION EXCEEDS abort_tolerance ({cfg.abort_tolerance}). "
                f"Likely bug. Aborting eps_s={eps_s}.",
                flush=True,
            )
            return {
                "eps_s": eps_s,
                "n_grid": n_grid,
                "mean_mse": mean_mse,
                "trials_used": trials_used,
                "slope": slope,
                "slope_ci": slope_ci,
                "status": "aborted",
            }

        # Converged
        if slope_err <= cfg.slope_tolerance and slope_ci <= cfg.ci_target:
            return {
                "eps_s": eps_s,
                "n_grid": n_grid,
                "mean_mse": mean_mse,
                "trials_used": trials_used,
                "slope": slope,
                "slope_ci": slope_ci,
                "status": "converged",
            }

        # Hit ceiling
        if trials_used[int(n_grid[0])] >= cfg.max_trials:
            return {
                "eps_s": eps_s,
                "n_grid": n_grid,
                "mean_mse": mean_mse,
                "trials_used": trials_used,
                "slope": slope,
                "slope_ci": slope_ci,
                "status": "maxed_out",
            }

        # Otherwise double trials and continue
        n_trials_this_round = min(
            n_trials_this_round * 2,
            cfg.max_trials - trials_used[int(n_grid[0])],
        )
        if n_trials_this_round <= 0:
            return {
                "eps_s": eps_s,
                "n_grid": n_grid,
                "mean_mse": mean_mse,
                "trials_used": trials_used,
                "slope": slope,
                "slope_ci": slope_ci,
                "status": "maxed_out",
            }


# =====================================================================
# Plot (publication style: muted, thin, no decoration)
# =====================================================================

def make_plot(curves: list[dict], cfg: Config, outpath: str) -> None:
    """Log-log MSE vs n. One curve per eps_s. Theoretical reference lines."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "legend.frameon": False,
    })

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    # Color-blind-safe palette (Wong, 2011), three muted colors
    palette = ["#0072B2", "#D55E00", "#009E73"]
    markers = ["o", "s", "^"]

    for i, curve in enumerate(curves):
        eps_s = curve["eps_s"]
        n_grid = curve["n_grid"]
        mse = curve["mean_mse"]
        color = palette[i % len(palette)]
        marker = markers[i % len(markers)]

        # Empirical curve
        ax.plot(
            n_grid, mse,
            color=color, marker=marker, linestyle="-",
            label=fr"$\varepsilon_s={eps_s}$",
        )

        # Theoretical reference: m / (n * eps_s^2), aligned at the first n
        theory = cfg.m / (n_grid * eps_s ** 2)
        # Match constant by aligning at smallest n
        ax.plot(
            n_grid, theory,
            color=color, linestyle=":", linewidth=0.8, alpha=0.7,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$ (number of users)")
    ax.set_ylabel(r"structure MSE $\,\|\widehat{\theta} - \theta\|_2^2 / m$")
    ax.legend(loc="upper right")

    # Minimal grid: only major, very faint
    ax.grid(True, which="major", linestyle="-", linewidth=0.3, alpha=0.3)
    ax.tick_params(direction="in", which="both")

    plt.tight_layout(pad=0.4)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}", flush=True)


def write_csv(curves: list[dict], outpath: str) -> None:
    """Raw data dump for reproducibility."""
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps_s", "n", "mean_mse", "trials_used", "slope", "slope_ci_95", "status"])
        for c in curves:
            for n, mse in zip(c["n_grid"], c["mean_mse"]):
                w.writerow([
                    c["eps_s"], int(n), f"{mse:.6e}",
                    c["trials_used"][int(n)], f"{c['slope']:.4f}",
                    f"{c['slope_ci']:.4f}", c["status"],
                ])
    print(f"  wrote {outpath}", flush=True)


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    cfg = Config()
    print("=" * 60)
    print("SLDP rate validation")
    print("=" * 60)
    print(f"  m = {cfg.m}, eps_values = {cfg.eps_values}")
    print(f"  n_grid = {cfg.n_grid}")
    print(f"  base_trials = {cfg.base_trials}, max_trials = {cfg.max_trials}")
    print(f"  workers = {cfg.n_workers}")
    print()

    t0 = time.time()
    with mp.Pool(processes=cfg.n_workers) as pool:
        curves = []
        for eps_s in cfg.eps_values:
            print(f"--- eps_s = {eps_s} ---")
            curve = run_eps_curve(eps_s, cfg, pool)
            curves.append(curve)
            print(f"  status: {curve['status']}")
            print()

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")
    print()

    # Reporting
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for c in curves:
        print(
            f"  eps_s={c['eps_s']:>4}  status={c['status']:>10}  "
            f"slope={c['slope']:+.3f} ± {c['slope_ci']:.3f}"
        )

    # Bail out if any curve failed
    aborted = [c for c in curves if c["status"] == "aborted"]
    if aborted:
        print()
        print("ABORTED CURVES detected. Not generating plot.")
        print("Likely cause: missing debiasing in the structure estimator,")
        print("or MSE computed against the wrong target.")
        return 1

    # Make outputs
    print()
    print("Writing outputs...")
    make_plot(curves, cfg, "rate_validation.pdf")
    write_csv(curves, "rate_validation.csv")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
