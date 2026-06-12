"""Result 8: m x sparsity sweep + cube-root rule (prop:budget) check."""

import json
import math
import pathlib
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = pathlib.Path(__file__).parent
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

K = 10
N = 50_000
N_TRIALS = 5
EPS_TOTAL = 2.0
EPS_S_GRID = np.linspace(0.2, EPS_TOTAL - 0.2, 19)


def cube_root_optimum(m, K, eps_total):
    a = m ** (1.0 / 3.0)
    b = K ** (1.0 / 3.0)
    return eps_total * a / (a + b)


def make_population(m, sparsity, n, rng):
    """Sample a population with controlled sparsity. sparsity in (0, 1]
    is the expected fraction of bits set to 1 per record."""
    # Per-class profiles: 5 classes, each with a class-specific
    # signature in 1/4 of the registry.
    C = 5
    PREVALENCE = np.array([0.40, 0.30, 0.20, 0.07, 0.03])
    profiles = np.full((C, m), 0.5 * sparsity)
    for c in range(C):
        # Each class "owns" a contiguous block of the registry.
        block = slice(int(c * m / C), int((c + 1) * m / C))
        profiles[c, block] = min(0.95, 5 * sparsity)
        # Plus a few cross-class shared signals.
        if m >= 16:
            profiles[c, ::max(1, m // 8)] += 0.1 * sparsity
    profiles = np.clip(profiles, 0.01, 0.99)
    classes = rng.choice(C, size=n, p=PREVALENCE)
    S = (rng.random((n, m)) < profiles[classes]).astype(np.int8)

    # Tracer value field (for value MSE).
    f_true = rng.dirichlet(np.ones(K))
    X = rng.choice(K, size=n, p=f_true)
    return classes, S, profiles, PREVALENCE, X, f_true


def bayes_optimal_acc(S_obs, classes, profiles, prior, eps_s):
    """Closed-form Bayes posterior given known channel and profiles."""
    a = math.exp(eps_s) / (math.exp(eps_s) + 1.0)
    q = a * profiles + (1 - a) * (1 - profiles)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    log_q = np.log(q)
    log_1mq = np.log(1 - q)
    Y = S_obs.astype(np.int8)
    log_lik = Y @ log_q.T + (1 - Y) @ log_1mq.T
    log_post = log_lik + np.log(prior)[None, :]
    pred = log_post.argmax(1)
    return float((pred == classes).mean())


def structure_mse(S_obs, S_true, eps_s):
    """Per-coordinate population-frequency MSE under unbiased RR
    estimator."""
    a = math.exp(eps_s) / (math.exp(eps_s) + 1.0)
    Y_bar = S_obs.mean(axis=0)
    p_hat = (Y_bar - (1 - a)) / (2 * a - 1)
    p_true = S_true.mean(axis=0)
    return float(np.sum((p_hat - p_true) ** 2))


def value_mse(X_obs, f_true, K, eps_v):
    a = math.exp(eps_v) / (math.exp(eps_v) + K - 1)
    b = 1.0 / (math.exp(eps_v) + K - 1)
    f_obs = np.bincount(X_obs, minlength=K) / len(X_obs)
    f_hat = (f_obs - b) / (a - b)
    return float(np.sum((f_hat - f_true) ** 2))


def randomize_value_krr(X, K, eps_v, rng):
    a = math.exp(eps_v) / (math.exp(eps_v) + K - 1)
    keep = rng.random(len(X)) < a
    flip_to = rng.integers(0, K - 1, size=len(X))
    flip_to = np.where(flip_to >= X, flip_to + 1, flip_to)
    return np.where(keep, X, flip_to)


def run_cell(m, sparsity, seed):
    rng = np.random.default_rng(seed)
    classes, S, profiles, prior, X, f_true = make_population(
        m, sparsity, N, rng)

    rows = []
    for eps_s in EPS_S_GRID:
        eps_v = EPS_TOTAL - eps_s
        # Structure side: bitwise RR on S.
        a = math.exp(eps_s) / (math.exp(eps_s) + 1.0)
        keep = rng.random(S.shape) < a
        S_obs = np.where(keep, S, 1 - S).astype(np.int8)

        # Value side: k-RR on X.
        X_obs = randomize_value_krr(X, K, eps_v, rng)

        attack_acc = bayes_optimal_acc(S_obs, classes, profiles, prior,
                                        eps_s)
        s_mse = structure_mse(S_obs, S, eps_s)
        v_mse = value_mse(X_obs, f_true, K, eps_v)
        rows.append({"eps_s": float(eps_s), "eps_v": float(eps_v),
                     "attack_acc": attack_acc,
                     "structure_mse": s_mse,
                     "value_mse": v_mse,
                     "total_mse": s_mse + v_mse})
    return rows


def aggregate_cell(m, sparsity):
    trials = [run_cell(m, sparsity, seed) for seed in range(N_TRIALS)]
    out = []
    for i in range(len(EPS_S_GRID)):
        eps_s = trials[0][i]["eps_s"]
        attack = np.mean([t[i]["attack_acc"] for t in trials])
        s_mse = np.mean([t[i]["structure_mse"] for t in trials])
        v_mse = np.mean([t[i]["value_mse"] for t in trials])
        tot = np.mean([t[i]["total_mse"] for t in trials])
        out.append({
            "eps_s": float(eps_s),
            "eps_v": float(EPS_TOTAL - eps_s),
            "attack_acc": float(attack),
            "structure_mse": float(s_mse),
            "value_mse": float(v_mse),
            "total_mse": float(tot),
        })
    return out


def main():
    cells = []
    sparsities = [("dense", 1.0), ("10%", 0.10), ("1%", 0.01)]
    ms = [16, 64, 256]

    print("=" * 100)
    print(f"Scale x sparsity sweep: K={K}, n={N}, eps_total={EPS_TOTAL}, "
          f"trials={N_TRIALS}")
    print(f"Grid points along eps_s: {len(EPS_S_GRID)}")
    print("=" * 100)
    print(f"{'m':>6} | {'sparsity':>9} | {'eps_s* (theory)':>16} | "
          f"{'eps_s* (empirical)':>20} | min total MSE")
    print("-" * 100)

    for m in ms:
        for s_label, s_val in sparsities:
            data = aggregate_cell(m, s_val)
            theo = cube_root_optimum(m, K, EPS_TOTAL)
            tot = [r["total_mse"] for r in data]
            i_min = int(np.argmin(tot))
            emp = data[i_min]["eps_s"]
            cells.append({"m": m, "sparsity": s_label,
                          "sparsity_val": s_val,
                          "eps_s_theory": float(theo),
                          "eps_s_empirical": float(emp),
                          "min_total_mse": float(tot[i_min]),
                          "data": data})
            print(f"{m:>6} | {s_label:>9} | {theo:16.3f} | "
                  f"{emp:20.3f} | {tot[i_min]:.3e}")

    out = {"K": K, "N": N, "N_TRIALS": N_TRIALS,
           "EPS_TOTAL": EPS_TOTAL,
           "cells": cells}
    with open(ROOT / "results_scale.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure 1: total-MSE U-curves with cube-root rule overlay ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for ax, m in zip(axes, ms):
        for s_label, s_val in sparsities:
            cell = next(c for c in cells
                        if c["m"] == m and c["sparsity"] == s_label)
            x = [r["eps_s"] for r in cell["data"]]
            y = [r["total_mse"] for r in cell["data"]]
            ax.plot(x, y, "-o", label=f"sparsity={s_label}", markersize=4)
        theo = cube_root_optimum(m, K, EPS_TOTAL)
        ax.axvline(theo, linestyle="--", color="black", alpha=0.6,
                   label=f"cube-root $\\varepsilon_s^*$ = {theo:.2f}")
        ax.set_xlabel(r"$\varepsilon_s$ (structure budget)")
        ax.set_yscale("log")
        ax.set_title(f"m = {m}")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("total MSE = struct + value")
    fig.suptitle(f"Empirical vs cube-root optimum"
                 f" ($\\varepsilon_\\mathrm{{tot}}={EPS_TOTAL}$, K={K})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "scale_sparsity_budget.pdf")
    fig.savefig(FIGS / "scale_sparsity_budget.png", dpi=160)
    plt.close(fig)

    # ---- figure 2: cube-root validation across (m, sparsity) ---------
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for s_label, s_val in sparsities:
        xs = [c["m"] for c in cells if c["sparsity"] == s_label]
        ys = [c["eps_s_empirical"] for c in cells
              if c["sparsity"] == s_label]
        ax.plot(xs, ys, "o-", label=f"empirical ({s_label})", markersize=8)
    # Theoretical curve.
    m_curve = np.array(ms, dtype=float)
    theo_curve = np.array([cube_root_optimum(m, K, EPS_TOTAL) for m in m_curve])
    ax.plot(m_curve, theo_curve, "--", color="black",
            label="cube-root rule (theory)")
    ax.set_xscale("log")
    ax.set_xlabel("registry size $m$")
    ax.set_ylabel(r"empirical optimum $\varepsilon_s^*$")
    ax.set_title("Cube-root rule (Prop. budget) vs. empirical optimum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "cube_root_validation.pdf")
    fig.savefig(FIGS / "cube_root_validation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
