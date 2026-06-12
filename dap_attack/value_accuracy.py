"""Result 3: DAP O(1/n) vs SLDP O(K/(n eps_v^2)) value-MSE comparison."""

import json
import pathlib
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = pathlib.Path(__file__).parent
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)


def krr_estimate(samples, K, eps, rng):
    # k-RR with debias f_hat = (f_obs - b)/(a - b).
    n = len(samples)
    a = np.exp(eps) / (np.exp(eps) + K - 1)
    b = 1.0 / (np.exp(eps) + K - 1)
    keep = rng.random(n) < a
    flip_to = rng.integers(0, K - 1, size=n)
    flip_to = np.where(flip_to >= samples, flip_to + 1, flip_to)
    out = np.where(keep, samples, flip_to)
    f_obs = np.bincount(out, minlength=K) / n
    return (f_obs - b) / (a - b)


def true_mse(f_hat, f_true):
    return float(np.sum((f_hat - f_true) ** 2))


def dap_estimate(samples, K):
    return np.bincount(samples, minlength=K) / len(samples)


def main():
    rng = np.random.default_rng(0)
    K = 10
    f_true = rng.dirichlet(alpha=np.ones(K))
    n_grid = [int(x) for x in np.logspace(3, 6, 7)]
    eps_grid = [0.5, 1.0, 2.0, 4.0]
    N_TRIALS = 30

    results = {"K": K, "f_true": f_true.tolist(),
               "n_grid": n_grid, "eps_grid": eps_grid,
               "N_TRIALS": N_TRIALS}

    dap_curve = []
    sldp_curves = {eps: [] for eps in eps_grid}

    for n in n_grid:
        dap_mses, sldp_mses = [], {eps: [] for eps in eps_grid}
        for t in range(N_TRIALS):
            r = np.random.default_rng(1000 + t * 13 + n)
            samples = r.choice(K, size=n, p=f_true)
            f_hat_dap = dap_estimate(samples, K)
            dap_mses.append(true_mse(f_hat_dap, f_true))
            for eps in eps_grid:
                f_hat = krr_estimate(samples, K, eps, r)
                sldp_mses[eps].append(true_mse(f_hat, f_true))
        dap_curve.append({
            "n": n,
            "mse_mean": float(np.mean(dap_mses)),
            "mse_ci": float(1.96 * np.std(dap_mses, ddof=1) / np.sqrt(N_TRIALS)),
        })
        for eps in eps_grid:
            sldp_curves[eps].append({
                "n": n,
                "mse_mean": float(np.mean(sldp_mses[eps])),
                "mse_ci": float(1.96 * np.std(sldp_mses[eps], ddof=1) / np.sqrt(N_TRIALS)),
            })

    results["dap_curve"] = dap_curve
    results["sldp_curves"] = {str(eps): sldp_curves[eps] for eps in eps_grid}

    # Crossover: at fixed eps, find n where DAP MSE first beats SLDP by 10x.
    crossovers = {}
    for eps in eps_grid:
        ratios = []
        for i, n in enumerate(n_grid):
            r = sldp_curves[eps][i]["mse_mean"] / dap_curve[i]["mse_mean"]
            ratios.append(r)
        crossovers[str(eps)] = {"ratios_per_n": ratios}
    results["sldp_over_dap_ratios"] = crossovers

    # ---- figure -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    n_arr = np.array(n_grid, dtype=float)
    dap_y = np.array([row["mse_mean"] for row in dap_curve])
    ax.loglog(n_arr, dap_y, marker="o", color="C3",
              label="DAP (secure aggregation)")
    for i, eps in enumerate(eps_grid):
        ys = np.array([row["mse_mean"] for row in sldp_curves[eps]])
        ax.loglog(n_arr, ys, marker="s",
                  label=fr"SLDP $\varepsilon_v={eps}$")

    # Reference slopes.
    ax.loglog(n_arr, 1.0 / n_arr, linestyle=":", color="gray", alpha=0.6,
              label=r"$1/n$ slope")
    ax.set_xlabel("clients $n$")
    ax.set_ylabel("value-frequency MSE")
    ax.set_title(f"DAP vs SLDP value accuracy (K={K} categories)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "dap_vs_sldp_value_mse.pdf")
    fig.savefig(FIGS / "dap_vs_sldp_value_mse.png", dpi=160)
    plt.close(fig)

    with open(ROOT / "results_value.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== Value-accuracy comparison ===")
    print(f"K = {K}, f_true = Dirichlet(1)")
    print()
    print("        n |   DAP MSE  | SLDP eps=0.5 | eps=1.0   | eps=2.0   | eps=4.0")
    for i, n in enumerate(n_grid):
        d = dap_curve[i]["mse_mean"]
        s = [sldp_curves[eps][i]["mse_mean"] for eps in eps_grid]
        print(f"{n:9d} | {d:.2e} | {s[0]:.2e}     | {s[1]:.2e}  | {s[2]:.2e}  | {s[3]:.2e}")
    print()
    print("SLDP-over-DAP MSE ratios (how many times worse SLDP is):")
    print("        n | eps=0.5 | eps=1.0 | eps=2.0 | eps=4.0")
    for i, n in enumerate(n_grid):
        rs = [crossovers[str(eps)]["ratios_per_n"][i] for eps in eps_grid]
        print(f"{n:9d} | {rs[0]:7.1f} | {rs[1]:7.1f} | {rs[2]:7.1f} | {rs[3]:7.1f}")


if __name__ == "__main__":
    main()
