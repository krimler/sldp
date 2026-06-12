"""Result 7: SLDP vs pad / k-anonymity / random-suppression on the
(attack-acc, population-MSE) plane."""

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

# Same population as attack_sim.py.
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


def sample_population(rng):
    classes = rng.choice(C, size=N, p=PREVALENCE)
    profiles = DEVICE_PROFILES[classes]
    S = (rng.random((N, M)) < profiles).astype(np.int8)
    theta_true = (PREVALENCE[:, None] * DEVICE_PROFILES).sum(axis=0)  # (M,)
    return classes, S, theta_true


def attack_lr(S_obs, classes, mask=None):
    if mask is not None:
        idx = np.where(mask)[0]
        if len(idx) < 100:
            return float("nan"), float("nan")
        S_obs = S_obs[idx]; classes = classes[idx]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(classes))
    split = int(0.7 * len(classes))
    tr, te = perm[:split], perm[split:]
    sc = StandardScaler(with_mean=False)
    Xtr = sc.fit_transform(S_obs[tr].astype(float))
    Xte = sc.transform(S_obs[te].astype(float))
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    clf.fit(Xtr, classes[tr])
    acc = float(clf.score(Xte, classes[te]))
    return acc, len(idx) / len(mask) if mask is not None else 1.0


def population_mse(theta_hat, theta_true):
    return float(np.sum((theta_hat - theta_true) ** 2))


# ---- defenses ---------------------------------------------------------
def D_no_defense(S, rng):
    return S, np.ones(len(S), bool), S.mean(axis=0)


def D_pad_ones(S, rng):
    S_obs = np.ones_like(S)
    # Population estimate: useless -- we report the all-ones vector.
    theta_hat = np.ones(S.shape[1])
    return S_obs, np.ones(len(S), bool), theta_hat


def D_pad_pattern(S, rng):
    pattern = np.array([1, 0] * (S.shape[1] // 2))
    S_obs = np.broadcast_to(pattern, S.shape).copy()
    theta_hat = pattern.astype(float)
    return S_obs, np.ones(len(S), bool), theta_hat


def D_k_anonymity(S, rng, k=10):
    keys = [tuple(row) for row in S]
    counts = {}
    for kk in keys:
        counts[kk] = counts.get(kk, 0) + 1
    keep_mask = np.array([counts[kk] >= k for kk in keys])
    if keep_mask.sum() == 0:
        # Fall back to "drop everything" -> theta_hat == 0.5 (uninformative)
        theta_hat = np.full(S.shape[1], 0.5)
    else:
        theta_hat = S[keep_mask].mean(axis=0)
    return S, keep_mask, theta_hat


def D_singleton_suppression(S, rng):
    return D_k_anonymity(S, rng, k=2)


def D_random_drop(S, rng, delta=0.5):
    # Sentinel = -1 so LR distinguishes suppressed from 0/1.
    drop = rng.random(S.shape) < delta
    S_obs = np.where(drop, -1, S).astype(np.int8)
    # Population estimate: per coord, drop the suppressed entries.
    theta_hat = np.zeros(S.shape[1])
    for i in range(S.shape[1]):
        kept = S[:, i][~drop[:, i]]
        theta_hat[i] = kept.mean() if len(kept) else 0.5
    return S_obs, np.ones(len(S), bool), theta_hat


def D_sldp(S, rng, eps_s):
    a = math.exp(eps_s) / (math.exp(eps_s) + 1.0)
    keep = rng.random(S.shape) < a
    S_obs = np.where(keep, S, 1 - S).astype(np.int8)
    # Unbiased population estimate: theta_hat = (Y_bar - (1-a)) / (2a-1).
    Y_bar = S_obs.mean(axis=0)
    theta_hat = (Y_bar - (1 - a)) / (2 * a - 1)
    theta_hat = np.clip(theta_hat, 0, 1)
    return S_obs, np.ones(len(S), bool), theta_hat


DEFENSES = [
    ("D0 No defense (DAP plaintext, collusion)",
        lambda S, rng: D_no_defense(S, rng)),
    ("D1 Pad-to-ones",
        lambda S, rng: D_pad_ones(S, rng)),
    ("D2 Pad-to-fixed-pattern",
        lambda S, rng: D_pad_pattern(S, rng)),
    ("D3 k-anonymity (k=10)",
        lambda S, rng: D_k_anonymity(S, rng, k=10)),
    ("D4 Singleton suppression (k=2)",
        lambda S, rng: D_singleton_suppression(S, rng)),
    ("D5 Random bit suppression (delta=0.5)",
        lambda S, rng: D_random_drop(S, rng, delta=0.5)),
    ("D6 SLDP RR eps_s=1.0",
        lambda S, rng: D_sldp(S, rng, eps_s=1.0)),
    ("D6 SLDP RR eps_s=0.5",
        lambda S, rng: D_sldp(S, rng, eps_s=0.5)),
    ("D6 SLDP RR eps_s=0.25",
        lambda S, rng: D_sldp(S, rng, eps_s=0.25)),
]


def main():
    rows = []
    for label, defense in DEFENSES:
        accs, mses, retentions = [], [], []
        for seed in range(N_TRIALS):
            rng = np.random.default_rng(seed)
            classes, S, theta_true = sample_population(rng)
            S_obs, mask, theta_hat = defense(S, rng)
            mses.append(population_mse(theta_hat, theta_true))
            retentions.append(float(mask.mean()))
            acc, _ = attack_lr(S_obs, classes, mask=mask)
            accs.append(acc)
        rows.append({
            "label": label,
            "attack_acc_mean": float(np.nanmean(accs)),
            "attack_acc_ci": float(
                1.96 * np.nanstd(accs, ddof=1) / math.sqrt(N_TRIALS)),
            "pop_mse_mean": float(np.mean(mses)),
            "pop_mse_ci": float(
                1.96 * np.std(mses, ddof=1) / math.sqrt(N_TRIALS)),
            "retention_mean": float(np.mean(retentions)),
        })

    print("=" * 100)
    print(f"{'Defense':>40} | {'attack_acc':>14} | {'pop_MSE':>10} | retention")
    print("-" * 100)
    for r in rows:
        print(f"{r['label']:>40} | "
              f"{r['attack_acc_mean']:.3f} +/- {r['attack_acc_ci']:.3f} | "
              f"{r['pop_mse_mean']:.2e} | "
              f"{r['retention_mean']:.2f}")

    with open(ROOT / "results_defenses.json", "w") as f:
        json.dump({"rows": rows, "M": M, "N": N, "N_TRIALS": N_TRIALS,
                   "random_guess": float(PREVALENCE.max())}, f, indent=2)

    # ---- figure: Pareto frontier --------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    palette = {
        "D0": "C3",
        "D1": "C7",
        "D2": "C8",
        "D3": "C5",
        "D4": "C4",
        "D5": "C9",
        "D6": "C0",
    }
    used_keys = set()
    for r in rows:
        key = r["label"][:2]
        x = r["attack_acc_mean"]
        y = max(r["pop_mse_mean"], 1e-6)
        color = palette[key]
        marker = "X" if key == "D0" else ("s" if key == "D6" else "o")
        ax.errorbar(x, y, xerr=r["attack_acc_ci"], yerr=r["pop_mse_ci"],
                    marker=marker, markersize=10, color=color, capsize=3)
        ax.annotate(r["label"], (x, y),
                    xytext=(8, 5), textcoords="offset points", fontsize=7)
        used_keys.add(key)

    ax.axvline(PREVALENCE.max(), linestyle=":", color="gray", alpha=0.7,
               label=f"random guess = {PREVALENCE.max():.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("attack accuracy (lower is better)")
    ax.set_ylabel("population-frequency MSE (lower is better)")
    ax.set_title("Privacy/utility frontier: SLDP vs simpler structural defenses")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "defense_frontier.pdf")
    fig.savefig(FIGS / "defense_frontier.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
