"""Result 1+2: DAP structural-leakage attack, SLDP defense.
Collusion adversary; m=16 fields; see CONFORMANCE.md for threat model."""

import json
import pathlib
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = pathlib.Path(__file__).parent
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

RNG = np.random.default_rng(0)

# ---- population ---------------------------------------------------------
M = 16            # registry size
N = 10_000        # clients per simulation
N_TRIALS = 5      # Monte-Carlo trials for confidence intervals

# Five device classes with distinct structural profiles.
# Each row gives Pr[S_i = 1 | class].
DEVICE_PROFILES = np.array([
    # 16 fields:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    [0.95, 0.90, 0.85, 0.80, 0.10, 0.05, 0.05, 0.10, 0.50, 0.50, 0.05, 0.05, 0.95, 0.90, 0.05, 0.05],  # desktop
    [0.10, 0.10, 0.85, 0.80, 0.95, 0.90, 0.05, 0.10, 0.50, 0.50, 0.05, 0.05, 0.10, 0.10, 0.95, 0.90],  # mobile
    [0.95, 0.90, 0.10, 0.10, 0.10, 0.05, 0.95, 0.90, 0.50, 0.50, 0.05, 0.05, 0.10, 0.90, 0.05, 0.05],  # tablet
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05],  # iot
    [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.95, 0.05, 0.95, 0.05, 0.50, 0.50, 0.50, 0.50],  # researcher (rare)
])
PREVALENCE = np.array([0.40, 0.30, 0.20, 0.07, 0.03])  # 3% rare class

assert DEVICE_PROFILES.shape == (5, M)
assert np.isclose(PREVALENCE.sum(), 1.0)


def sample_population(n, rng):
    """Sample (class_label, structure_vector) for n clients."""
    classes = rng.choice(len(PREVALENCE), size=n, p=PREVALENCE)
    profiles = DEVICE_PROFILES[classes]      # (n, M)
    S = (rng.random((n, M)) < profiles).astype(np.int8)
    return classes, S


def rr_flip_prob(eps):
    """Standard binary RR keep-probability for eps-LDP."""
    return np.exp(eps) / (np.exp(eps) + 1.0)


def sldp_structure(S, eps_s, rng):
    """Bitwise eps_s-LDP randomised response on the structure vector."""
    p_keep = rr_flip_prob(eps_s)
    keep = rng.random(S.shape) < p_keep
    return np.where(keep, S, 1 - S).astype(np.int8)


def attack_classifier(S_train, y_train, S_test, y_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(S_train.astype(float))
    Xte = scaler.transform(S_test.astype(float))
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    clf.fit(Xtr, y_train)
    acc = clf.score(Xte, y_test)
    proba = clf.predict_proba(Xte)
    auc_rare = roc_auc_score((y_test == 4).astype(int), proba[:, 4])
    return acc, auc_rare


def k_anonymity(S):
    keys = [tuple(row) for row in S]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    sizes = np.array(list(counts.values()))
    n_unique = len(counts)
    singleton_frac = (sizes == 1).sum() / len(keys)
    median_class_size = float(np.median([counts[k] for k in keys]))
    return n_unique, singleton_frac, median_class_size


def run_one_trial(eps_s_grid, rng):
    classes, S_true = sample_population(N, rng)
    split = int(0.7 * N)
    perm = rng.permutation(N)
    train_idx, test_idx = perm[:split], perm[split:]

    results = {}

    # DAP baseline (no SLDP).
    acc, auc = attack_classifier(
        S_true[train_idx], classes[train_idx],
        S_true[test_idx], classes[test_idx],
    )
    n_uniq, singleton, med = k_anonymity(S_true)
    results["dap_baseline"] = {
        "eps_s": float("inf"),
        "attack_accuracy": acc,
        "rare_class_auc": auc,
        "n_unique_patterns": int(n_uniq),
        "singleton_fraction": float(singleton),
        "median_anon_set": med,
    }

    # SLDP defense across eps_s.
    sldp_curve = []
    for eps_s in eps_s_grid:
        S_obs = sldp_structure(S_true, eps_s, rng)
        acc, auc = attack_classifier(
            S_obs[train_idx], classes[train_idx],
            S_obs[test_idx], classes[test_idx],
        )
        n_uniq, singleton, med = k_anonymity(S_obs)
        sldp_curve.append({
            "eps_s": eps_s,
            "attack_accuracy": acc,
            "rare_class_auc": auc,
            "n_unique_patterns": int(n_uniq),
            "singleton_fraction": float(singleton),
            "median_anon_set": med,
        })
    results["sldp_curve"] = sldp_curve
    results["random_guess_accuracy"] = float(PREVALENCE.max())
    return results


def aggregate(trials, key_path):
    vals = []
    for t in trials:
        v = t
        for k in key_path:
            v = v[k]
        vals.append(v)
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))


def main():
    eps_s_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    trials = [run_one_trial(eps_s_grid, np.random.default_rng(seed))
              for seed in range(N_TRIALS)]

    summary = {"N": N, "M": M, "N_TRIALS": N_TRIALS,
               "device_classes": int(len(PREVALENCE)),
               "rare_class_prevalence": float(PREVALENCE[-1])}

    # DAP baseline aggregated.
    summary["dap_baseline"] = {
        "attack_accuracy_mean": aggregate(trials, ["dap_baseline", "attack_accuracy"])[0],
        "attack_accuracy_ci": aggregate(trials, ["dap_baseline", "attack_accuracy"])[1],
        "rare_class_auc_mean": aggregate(trials, ["dap_baseline", "rare_class_auc"])[0],
        "n_unique_patterns_mean": aggregate(trials, ["dap_baseline", "n_unique_patterns"])[0],
        "singleton_fraction_mean": aggregate(trials, ["dap_baseline", "singleton_fraction"])[0],
        "median_anon_set_mean": aggregate(trials, ["dap_baseline", "median_anon_set"])[0],
    }

    # SLDP curve aggregated.
    sldp = []
    for i, eps_s in enumerate(eps_s_grid):
        acc_m, acc_ci = aggregate(trials, ["sldp_curve", i, "attack_accuracy"])
        auc_m, auc_ci = aggregate(trials, ["sldp_curve", i, "rare_class_auc"])
        uniq_m, _ = aggregate(trials, ["sldp_curve", i, "n_unique_patterns"])
        sing_m, _ = aggregate(trials, ["sldp_curve", i, "singleton_fraction"])
        med_m, _ = aggregate(trials, ["sldp_curve", i, "median_anon_set"])
        sldp.append({
            "eps_s": eps_s,
            "attack_accuracy_mean": acc_m, "attack_accuracy_ci": acc_ci,
            "rare_class_auc_mean": auc_m, "rare_class_auc_ci": auc_ci,
            "n_unique_patterns_mean": uniq_m,
            "singleton_fraction_mean": sing_m,
            "median_anon_set_mean": med_m,
        })
    summary["sldp_curve"] = sldp
    summary["random_guess_accuracy"] = float(PREVALENCE.max())

    # ---- figure: attack accuracy + rare-class AUC vs eps_s -------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    eps = np.array([row["eps_s"] for row in sldp])
    acc = np.array([row["attack_accuracy_mean"] for row in sldp])
    acc_ci = np.array([row["attack_accuracy_ci"] for row in sldp])
    auc = np.array([row["rare_class_auc_mean"] for row in sldp])
    auc_ci = np.array([row["rare_class_auc_ci"] for row in sldp])

    ax.errorbar(eps, acc, yerr=acc_ci, marker="o", capsize=3,
                label="device-class accuracy")
    ax.errorbar(eps, auc, yerr=auc_ci, marker="s", capsize=3,
                label="rare-class AUC (3% prevalence)")
    ax.axhline(summary["dap_baseline"]["attack_accuracy_mean"],
               linestyle="--", color="C3",
               label=f"DAP baseline accuracy = {summary['dap_baseline']['attack_accuracy_mean']:.2f}")
    ax.axhline(summary["random_guess_accuracy"], linestyle=":",
               color="gray",
               label=f"random guess = {summary['random_guess_accuracy']:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"structural privacy budget $\varepsilon_s$")
    ax.set_ylabel("adversary success")
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Structural fingerprinting: DAP baseline vs SLDP defense")
    fig.tight_layout()
    fig.savefig(FIGS / "dap_attack_accuracy.pdf")
    fig.savefig(FIGS / "dap_attack_accuracy.png", dpi=160)
    plt.close(fig)

    # ---- figure: anonymity-set collapse --------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    sing = np.array([row["singleton_fraction_mean"] for row in sldp])
    uniq = np.array([row["n_unique_patterns_mean"] for row in sldp])
    ax.plot(eps, sing, marker="o", label="fraction of clients in unique pattern")
    ax2 = ax.twinx()
    ax2.plot(eps, uniq, marker="s", color="C1", label="distinct patterns observed")
    ax.set_xscale("log")
    ax.set_xlabel(r"structural privacy budget $\varepsilon_s$")
    ax.set_ylabel("singleton fraction")
    ax2.set_ylabel("distinct patterns")
    dap = summary["dap_baseline"]
    ax.axhline(dap["singleton_fraction_mean"], linestyle="--", color="C3",
               label=f"DAP baseline singletons = {dap['singleton_fraction_mean']:.2f}")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    ax.set_title("Anonymity-set collapse under DAP, restored by SLDP")
    fig.tight_layout()
    fig.savefig(FIGS / "dap_anonymity.pdf")
    fig.savefig(FIGS / "dap_anonymity.png", dpi=160)
    plt.close(fig)

    with open(ROOT / "results_attack.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print a compact summary so the user sees the headline numbers.
    print("=== DAP structural-leakage attack ===")
    print(f"DAP baseline attack accuracy: "
          f"{summary['dap_baseline']['attack_accuracy_mean']:.3f} "
          f"+/- {summary['dap_baseline']['attack_accuracy_ci']:.3f}")
    print(f"DAP baseline rare-class AUC: "
          f"{summary['dap_baseline']['rare_class_auc_mean']:.3f}")
    print(f"DAP baseline distinct patterns / {N}: "
          f"{summary['dap_baseline']['n_unique_patterns_mean']:.0f}")
    print(f"DAP baseline singleton fraction: "
          f"{summary['dap_baseline']['singleton_fraction_mean']:.3f}")
    print()
    print("eps_s | attack_acc | rare_auc | distinct_patterns | singletons")
    for row in sldp:
        print(f"{row['eps_s']:5.2f} | "
              f"{row['attack_accuracy_mean']:.3f}      | "
              f"{row['rare_class_auc_mean']:.3f}    | "
              f"{row['n_unique_patterns_mean']:.0f}              | "
              f"{row['singleton_fraction_mean']:.3f}")


if __name__ == "__main__":
    main()
