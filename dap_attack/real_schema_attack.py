"""Result 9: attack on real Firefox schema (66 probes from gecko-dev)."""

import json
import math
import pathlib
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import firefox_schema as fs

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = pathlib.Path(__file__).parent
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

PROFILES = fs.build_profile_matrix()      # (5, 66)
PREVALENCE = np.asarray(fs.PREVALENCE)
M = PROFILES.shape[1]
C = PROFILES.shape[0]
N = 10_000
N_TRIALS = 5


def sample_population(rng):
    classes = rng.choice(C, size=N, p=PREVALENCE)
    profiles = PROFILES[classes]
    S = (rng.random((N, M)) < profiles).astype(np.int8)
    return classes, S


def rr_keep(eps):
    return math.exp(eps) / (math.exp(eps) + 1.0)


def sldp_structure(S, eps_s, rng):
    a = rr_keep(eps_s)
    keep = rng.random(S.shape) < a
    return np.where(keep, S, 1 - S).astype(np.int8)


def attacker_lr(S_train, y_train, S_test, y_test):
    sc = StandardScaler()
    Xtr = sc.fit_transform(S_train.astype(float))
    Xte = sc.transform(S_test.astype(float))
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    clf.fit(Xtr, y_train)
    proba = clf.predict_proba(Xte)
    pred = proba.argmax(1)
    acc = float((pred == y_test).mean())
    auc = float(roc_auc_score((y_test == 4).astype(int), proba[:, 4]))
    return acc, auc


def attacker_bayes(S_test, y_test, eps_s):
    a = rr_keep(eps_s)
    q = a * PROFILES + (1 - a) * (1 - PROFILES)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    log_q = np.log(q); log_1mq = np.log(1 - q)
    log_lik = S_test @ log_q.T + (1 - S_test) @ log_1mq.T
    log_post = log_lik + np.log(PREVALENCE)[None, :]
    log_post -= log_post.max(axis=1, keepdims=True)
    p = np.exp(log_post); p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(1)
    acc = float((pred == y_test).mean())
    auc = float(roc_auc_score((y_test == 4).astype(int), p[:, 4]))
    return acc, auc


def trial(eps_s, seed):
    rng = np.random.default_rng(seed)
    classes, S_true = sample_population(rng)
    perm = rng.permutation(N); split = int(0.7 * N)
    tr, te = perm[:split], perm[split:]
    if math.isinf(eps_s):
        S_obs = S_true; eps_for_bayes = math.log(0.999 / 0.001)
    else:
        S_obs = sldp_structure(S_true, eps_s, rng)
        eps_for_bayes = eps_s
    lr_acc, lr_auc = attacker_lr(S_obs[tr], classes[tr],
                                 S_obs[te], classes[te])
    by_acc, by_auc = attacker_bayes(S_obs[te], classes[te], eps_for_bayes)
    return {"lr_acc": lr_acc, "lr_auc": lr_auc,
            "bayes_acc": by_acc, "bayes_auc": by_auc}


def main():
    eps_grid = [0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []

    # DAP plaintext baseline.
    runs = [trial(math.inf, seed) for seed in range(N_TRIALS)]
    rows.append({
        "label": "DAP (no SLDP)", "eps_s": float("inf"),
        "lr_acc_mean": float(np.mean([r["lr_acc"] for r in runs])),
        "lr_acc_ci": float(1.96*np.std([r["lr_acc"] for r in runs], ddof=1)/math.sqrt(N_TRIALS)),
        "bayes_acc_mean": float(np.mean([r["bayes_acc"] for r in runs])),
        "bayes_acc_ci": float(1.96*np.std([r["bayes_acc"] for r in runs], ddof=1)/math.sqrt(N_TRIALS)),
        "lr_auc_mean": float(np.mean([r["lr_auc"] for r in runs])),
        "bayes_auc_mean": float(np.mean([r["bayes_auc"] for r in runs])),
    })

    for eps in eps_grid:
        runs = [trial(eps, seed) for seed in range(N_TRIALS)]
        rows.append({
            "label": f"SLDP eps_s={eps}", "eps_s": eps,
            "lr_acc_mean": float(np.mean([r["lr_acc"] for r in runs])),
            "lr_acc_ci": float(1.96*np.std([r["lr_acc"] for r in runs], ddof=1)/math.sqrt(N_TRIALS)),
            "bayes_acc_mean": float(np.mean([r["bayes_acc"] for r in runs])),
            "bayes_acc_ci": float(1.96*np.std([r["bayes_acc"] for r in runs], ddof=1)/math.sqrt(N_TRIALS)),
            "lr_auc_mean": float(np.mean([r["lr_auc"] for r in runs])),
            "bayes_auc_mean": float(np.mean([r["bayes_auc"] for r in runs])),
        })

    print("=" * 90)
    print(f"Real Firefox schema attack: M={M} probes, C={C} flavors, "
          f"prevalence={PREVALENCE.tolist()}")
    print(f"Schema sources: gecko-dev/browser/components/metrics.yaml "
          f"+ Histograms.json")
    print("=" * 90)
    print(f"{'eps_s':>8} | {'LR':>16} | {'Bayes-opt':>16} | "
          f"{'rare AUC (Bayes)':>16}")
    print("-" * 90)
    for r in rows:
        eps_s = "DAP" if math.isinf(r["eps_s"]) else f"{r['eps_s']:.2f}"
        print(f"{eps_s:>8} | "
              f"{r['lr_acc_mean']:.3f} +/- {r['lr_acc_ci']:.3f} | "
              f"{r['bayes_acc_mean']:.3f} +/- {r['bayes_acc_ci']:.3f} | "
              f"{r['bayes_auc_mean']:.3f}")

    out = {"schema_size": M, "n_flavors": C, "N": N, "N_TRIALS": N_TRIALS,
           "prevalence": PREVALENCE.tolist(),
           "probe_names": fs.PROBE_NAMES,
           "flavors": fs.FLAVORS,
           "rows": rows,
           "random_guess": float(PREVALENCE.max())}
    with open(ROOT / "results_real_schema.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sldp = [r for r in rows if not math.isinf(r["eps_s"])]
    eps = np.array([r["eps_s"] for r in sldp])
    lr = np.array([r["lr_acc_mean"] for r in sldp])
    lr_ci = np.array([r["lr_acc_ci"] for r in sldp])
    by = np.array([r["bayes_acc_mean"] for r in sldp])
    by_ci = np.array([r["bayes_acc_ci"] for r in sldp])

    ax.errorbar(eps, lr, yerr=lr_ci, marker="o", color="C0", capsize=3,
                label="logistic regression")
    ax.errorbar(eps, by, yerr=by_ci, marker="^", color="C2", capsize=3,
                label="Bayes-optimal")
    dap = rows[0]
    ax.axhline(dap["bayes_acc_mean"], linestyle="--", color="C3",
               label=f"DAP plaintext baseline = {dap['bayes_acc_mean']:.2f}")
    ax.axhline(PREVALENCE.max(), linestyle=":", color="gray",
               label=f"random guess = {PREVALENCE.max():.2f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"structural privacy budget $\varepsilon_s$")
    ax.set_ylabel("attack accuracy on Firefox flavor")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"Real Firefox schema ({M} probes from gecko-dev): "
                 f"DAP vs SLDP")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "real_schema_attack.pdf")
    fig.savefig(FIGS / "real_schema_attack.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
