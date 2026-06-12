"""Result 5: end-to-end DAP vs SLDP vs Hybrid privacy/utility frontier."""

import json
import math
import pathlib
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = pathlib.Path(__file__).parent
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

M = 16
N = 10_000
N_TRIALS = 5

DEVICE_PROFILES = np.array([
    [0.95, 0.90, 0.85, 0.80, 0.10, 0.05, 0.05, 0.10, 0.50, 0.50, 0.05, 0.05, 0.95, 0.90, 0.05, 0.05],
    [0.10, 0.10, 0.85, 0.80, 0.95, 0.90, 0.05, 0.10, 0.50, 0.50, 0.05, 0.05, 0.10, 0.10, 0.95, 0.90],
    [0.95, 0.90, 0.10, 0.10, 0.10, 0.05, 0.95, 0.90, 0.50, 0.50, 0.05, 0.05, 0.10, 0.90, 0.05, 0.05],
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05],
    [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.95, 0.05, 0.95, 0.05, 0.50, 0.50, 0.50, 0.50],
])
PREVALENCE = np.array([0.40, 0.30, 0.20, 0.07, 0.03])
C = len(PREVALENCE)

# ---- value-side parameters (frequency-estimation task) ---------------
K = 10  # categorical value alphabet per active field

# We attribute one categorical value per ACTIVE field. To keep the
# experiment tractable and scalar-comparable, we evaluate value MSE on
# a single tracer field that every device class fills with a
# class-dependent Dirichlet draw.


def rr_keep(eps):
    return math.exp(eps) / (math.exp(eps) + 1.0)


def krr_keep(eps, K):
    return math.exp(eps) / (math.exp(eps) + K - 1)


def krr_other(eps, K):
    return 1.0 / (math.exp(eps) + K - 1)


def sample_population(n, rng):
    classes = rng.choice(C, size=n, p=PREVALENCE)
    profiles = DEVICE_PROFILES[classes]
    S = (rng.random((n, M)) < profiles).astype(np.int8)
    # Tracer values: each client emits one categorical X in [0, K).
    f_true = rng.dirichlet(np.ones(K))
    X = rng.choice(K, size=n, p=f_true)
    return classes, S, X, f_true


def randomize_structure(S, eps_s, rng):
    a = rr_keep(eps_s)
    keep = rng.random(S.shape) < a
    return np.where(keep, S, 1 - S).astype(np.int8)


def randomize_value_krr(X, K, eps_v, rng):
    a = krr_keep(eps_v, K)
    keep = rng.random(len(X)) < a
    flip_to = rng.integers(0, K - 1, size=len(X))
    flip_to = np.where(flip_to >= X, flip_to + 1, flip_to)
    return np.where(keep, X, flip_to)


def estimate_freq_dap(X, K):
    counts = np.bincount(X, minlength=K)
    return counts / len(X)


def estimate_freq_sldp(X_obs, K, eps_v):
    a = krr_keep(eps_v, K)
    b = krr_other(eps_v, K)
    f_obs = np.bincount(X_obs, minlength=K) / len(X_obs)
    return (f_obs - b) / (a - b)


def attack_structure(S_obs, classes):
    rng = np.random.default_rng(0)
    n = len(classes)
    perm = rng.permutation(n)
    split = int(0.7 * n)
    tr, te = perm[:split], perm[split:]
    sc = StandardScaler()
    Xtr = sc.fit_transform(S_obs[tr].astype(float))
    Xte = sc.transform(S_obs[te].astype(float))
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(Xtr, classes[tr])
    return clf.score(Xte, classes[te])


def run_pipeline(name, eps_s, eps_v, rng):
    classes, S, X, f_true = sample_population(N, rng)

    if name == "DAP":
        # Plaintext under collusion. Values aggregated exactly.
        S_obs = S
        f_hat = estimate_freq_dap(X, K)
    elif name == "SLDP":
        S_obs = randomize_structure(S, eps_s, rng)
        X_obs = randomize_value_krr(X, K, eps_v, rng)
        f_hat = estimate_freq_sldp(X_obs, K, eps_v)
    elif name == "Hybrid":
        # SLDP on S, DAP on X.
        S_obs = randomize_structure(S, eps_s, rng)
        # Values go through DAP -> exact aggregation under non-collusion.
        f_hat = estimate_freq_dap(X, K)
    else:
        raise ValueError(name)

    attack_acc = attack_structure(S_obs, classes)
    value_mse = float(np.sum((f_hat - f_true) ** 2))
    return attack_acc, value_mse


def main():
    # Sweep eps_s for SLDP and Hybrid; eps_v fixed at 1.0 for SLDP-only.
    eps_s_grid = [0.25, 0.5, 1.0, 2.0, 4.0]
    eps_v_for_sldp = 1.0

    rows = []
    # DAP baseline (no privacy parameters).
    accs, mses = [], []
    for seed in range(N_TRIALS):
        a, m = run_pipeline("DAP", None, None, np.random.default_rng(seed))
        accs.append(a); mses.append(m)
    rows.append({"pipeline": "DAP",
                 "eps_s": float("inf"), "eps_v": float("inf"),
                 "attack_acc_mean": float(np.mean(accs)),
                 "attack_acc_ci": float(1.96*np.std(accs, ddof=1)/math.sqrt(N_TRIALS)),
                 "value_mse_mean": float(np.mean(mses)),
                 "value_mse_ci": float(1.96*np.std(mses, ddof=1)/math.sqrt(N_TRIALS))})

    for eps_s in eps_s_grid:
        # SLDP-only.
        accs, mses = [], []
        for seed in range(N_TRIALS):
            a, m = run_pipeline("SLDP", eps_s, eps_v_for_sldp,
                                np.random.default_rng(1000 + seed))
            accs.append(a); mses.append(m)
        rows.append({"pipeline": "SLDP",
                     "eps_s": eps_s, "eps_v": eps_v_for_sldp,
                     "attack_acc_mean": float(np.mean(accs)),
                     "attack_acc_ci": float(1.96*np.std(accs, ddof=1)/math.sqrt(N_TRIALS)),
                     "value_mse_mean": float(np.mean(mses)),
                     "value_mse_ci": float(1.96*np.std(mses, ddof=1)/math.sqrt(N_TRIALS))})
        # Hybrid.
        accs, mses = [], []
        for seed in range(N_TRIALS):
            a, m = run_pipeline("Hybrid", eps_s, None,
                                np.random.default_rng(2000 + seed))
            accs.append(a); mses.append(m)
        rows.append({"pipeline": "Hybrid",
                     "eps_s": eps_s, "eps_v": float("inf"),
                     "attack_acc_mean": float(np.mean(accs)),
                     "attack_acc_ci": float(1.96*np.std(accs, ddof=1)/math.sqrt(N_TRIALS)),
                     "value_mse_mean": float(np.mean(mses)),
                     "value_mse_ci": float(1.96*np.std(mses, ddof=1)/math.sqrt(N_TRIALS))})

    print("=" * 80)
    print(f"{'pipeline':>9} | {'eps_s':>6} | {'eps_v':>6} | "
          f"{'attack_acc':>12} | {'value_mse':>12}")
    print("-" * 80)
    for r in rows:
        eps_s = "inf" if math.isinf(r["eps_s"]) else f"{r['eps_s']:.2f}"
        eps_v = "inf" if math.isinf(r["eps_v"]) else f"{r['eps_v']:.2f}"
        print(f"{r['pipeline']:>9} | {eps_s:>6} | {eps_v:>6} | "
              f"{r['attack_acc_mean']:.3f} +/- {r['attack_acc_ci']:.3f} | "
              f"{r['value_mse_mean']:.2e}")

    with open(ROOT / "results_hybrid.json", "w") as f:
        json.dump({"rows": rows, "K": K, "M": M, "N": N,
                   "N_TRIALS": N_TRIALS,
                   "eps_v_for_sldp": eps_v_for_sldp}, f, indent=2)

    # ---- figure: privacy-utility scatter -------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    # DAP point.
    dap = rows[0]
    ax.scatter([dap["attack_acc_mean"]], [dap["value_mse_mean"]],
               marker="X", s=140, color="C3", label="DAP-only (collusion)")
    ax.annotate("DAP", (dap["attack_acc_mean"], dap["value_mse_mean"]),
                xytext=(-30, -12), textcoords="offset points")

    # SLDP curve.
    sldp_rows = [r for r in rows if r["pipeline"] == "SLDP"]
    sldp_x = [r["attack_acc_mean"] for r in sldp_rows]
    sldp_y = [r["value_mse_mean"] for r in sldp_rows]
    ax.plot(sldp_x, sldp_y, "o-", color="C0",
            label=fr"SLDP-only ($\varepsilon_v={eps_v_for_sldp}$)")
    for r in sldp_rows:
        ax.annotate(fr"$\varepsilon_s={r['eps_s']}$",
                    (r["attack_acc_mean"], r["value_mse_mean"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=7)

    # Hybrid curve.
    hyb_rows = [r for r in rows if r["pipeline"] == "Hybrid"]
    hyb_x = [r["attack_acc_mean"] for r in hyb_rows]
    hyb_y = [r["value_mse_mean"] for r in hyb_rows]
    ax.plot(hyb_x, hyb_y, "s-", color="C2", label="Hybrid (SLDP on S, DAP on X)")
    for r in hyb_rows:
        ax.annotate(fr"$\varepsilon_s={r['eps_s']}$",
                    (r["attack_acc_mean"], r["value_mse_mean"]),
                    xytext=(6, -10), textcoords="offset points", fontsize=7)

    ax.set_yscale("log")
    ax.set_xlabel("structural attack accuracy (lower is better)")
    ax.set_ylabel("value-frequency MSE (lower is better)")
    ax.set_title("Privacy/utility frontier: DAP vs SLDP vs Hybrid")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "hybrid_frontier.pdf")
    fig.savefig(FIGS / "hybrid_frontier.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
