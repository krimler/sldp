"""Result 4: LR vs gradient-boosted tree vs closed-form Bayes-optimal."""

import json
import math
import pathlib
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
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

# Re-use the same population as attack_sim.py so numbers stack.
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


def sample_population(n, rng):
    classes = rng.choice(C, size=n, p=PREVALENCE)
    profiles = DEVICE_PROFILES[classes]
    S = (rng.random((n, M)) < profiles).astype(np.int8)
    return classes, S


def rr_keep_prob(eps):
    return math.exp(eps) / (math.exp(eps) + 1.0)


def sldp_structure(S, eps_s, rng):
    a = rr_keep_prob(eps_s)
    keep = rng.random(S.shape) < a
    return np.where(keep, S, 1 - S).astype(np.int8)


def attacker_lr(S_train, y_train, S_test, y_test):
    sc = StandardScaler()
    Xtr = sc.fit_transform(S_train.astype(float))
    Xte = sc.transform(S_test.astype(float))
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(Xtr, y_train)
    proba = clf.predict_proba(Xte)
    return proba.argmax(1), proba


def attacker_gbt(S_train, y_train, S_test, y_test):
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                         random_state=0)
    clf.fit(S_train.astype(float), y_train)
    proba = clf.predict_proba(S_test.astype(float))
    return proba.argmax(1), proba


def attacker_bayes(S_test, eps_s, profiles, prior):
    # Closed-form posterior; q[c, i] = Pr[Y_i = 1 | class = c].
    a = rr_keep_prob(eps_s)
    q = np.clip(a * profiles + (1 - a) * (1 - profiles), 1e-12, 1 - 1e-12)
    log_q, log_1mq = np.log(q), np.log(1 - q)
    Y = S_test.astype(np.int8)
    log_post = Y @ log_q.T + (1 - Y) @ log_1mq.T + np.log(prior)[None, :]
    log_post -= log_post.max(axis=1, keepdims=True)
    p = np.exp(log_post)
    p /= p.sum(axis=1, keepdims=True)
    return p.argmax(1), p


def run_attacker(name, fn_train, fn_predict, S_train, y_train, S_test, y_test,
                 **kwargs):
    if fn_train is not None:
        pred, proba = fn_train(S_train, y_train, S_test, y_test)
    else:
        pred, proba = fn_predict(S_test, **kwargs)
    acc = float((pred == y_test).mean())
    auc = float(roc_auc_score((y_test == 4).astype(int), proba[:, 4]))
    return acc, auc


def trial(eps_s, seed):
    rng = np.random.default_rng(seed)
    classes, S_true = sample_population(N, rng)
    perm = rng.permutation(N)
    split = int(0.7 * N)
    train_idx, test_idx = perm[:split], perm[split:]

    if math.isinf(eps_s):
        S_obs = S_true
    else:
        S_obs = sldp_structure(S_true, eps_s, rng)

    out = {}
    out["lr"] = run_attacker(
        "lr", attacker_lr, None,
        S_obs[train_idx], classes[train_idx],
        S_obs[test_idx], classes[test_idx])
    out["gbt"] = run_attacker(
        "gbt", attacker_gbt, None,
        S_obs[train_idx], classes[train_idx],
        S_obs[test_idx], classes[test_idx])

    if math.isinf(eps_s):
        # Plaintext baseline: Bayes uses profile-only classification.
        a_used = 1.0 - 1e-12
        eps_s_for_bayes = math.log(a_used / (1 - a_used))
    else:
        eps_s_for_bayes = eps_s

    out["bayes"] = run_attacker(
        "bayes", None, attacker_bayes,
        None, None, S_obs[test_idx], classes[test_idx],
        eps_s=eps_s_for_bayes,
        profiles=DEVICE_PROFILES,
        prior=PREVALENCE)
    return out


def main():
    eps_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    rows = []
    # DAP baseline (no SLDP).
    dap_runs = [trial(math.inf, seed) for seed in range(N_TRIALS)]
    rows.append({
        "label": "DAP (no SLDP)", "eps_s": float("inf"),
        "lr_acc": np.mean([r["lr"][0] for r in dap_runs]),
        "lr_acc_ci": 1.96 * np.std([r["lr"][0] for r in dap_runs], ddof=1) / math.sqrt(N_TRIALS),
        "gbt_acc": np.mean([r["gbt"][0] for r in dap_runs]),
        "gbt_acc_ci": 1.96 * np.std([r["gbt"][0] for r in dap_runs], ddof=1) / math.sqrt(N_TRIALS),
        "bayes_acc": np.mean([r["bayes"][0] for r in dap_runs]),
        "bayes_acc_ci": 1.96 * np.std([r["bayes"][0] for r in dap_runs], ddof=1) / math.sqrt(N_TRIALS),
        "lr_auc": np.mean([r["lr"][1] for r in dap_runs]),
        "gbt_auc": np.mean([r["gbt"][1] for r in dap_runs]),
        "bayes_auc": np.mean([r["bayes"][1] for r in dap_runs]),
    })
    # SLDP curve.
    for eps in eps_grid:
        runs = [trial(eps, seed) for seed in range(N_TRIALS)]
        rows.append({
            "label": f"SLDP eps_s={eps}", "eps_s": eps,
            "lr_acc": np.mean([r["lr"][0] for r in runs]),
            "lr_acc_ci": 1.96 * np.std([r["lr"][0] for r in runs], ddof=1) / math.sqrt(N_TRIALS),
            "gbt_acc": np.mean([r["gbt"][0] for r in runs]),
            "gbt_acc_ci": 1.96 * np.std([r["gbt"][0] for r in runs], ddof=1) / math.sqrt(N_TRIALS),
            "bayes_acc": np.mean([r["bayes"][0] for r in runs]),
            "bayes_acc_ci": 1.96 * np.std([r["bayes"][0] for r in runs], ddof=1) / math.sqrt(N_TRIALS),
            "lr_auc": np.mean([r["lr"][1] for r in runs]),
            "gbt_auc": np.mean([r["gbt"][1] for r in runs]),
            "bayes_auc": np.mean([r["bayes"][1] for r in runs]),
        })

    print("=" * 80)
    print(f"{'eps_s':>8} | {'LR':>16} | {'GBT':>16} | {'Bayes-opt':>16} | rare AUC (Bayes)")
    print("-" * 80)
    for r in rows:
        eps_s = "DAP" if math.isinf(r["eps_s"]) else f"{r['eps_s']:.2f}"
        print(f"{eps_s:>8} | "
              f"{r['lr_acc']:.3f} +/- {r['lr_acc_ci']:.3f} | "
              f"{r['gbt_acc']:.3f} +/- {r['gbt_acc_ci']:.3f} | "
              f"{r['bayes_acc']:.3f} +/- {r['bayes_acc_ci']:.3f} | "
              f"{r['bayes_auc']:.3f}")

    out_json = {"rows": rows, "PREVALENCE": PREVALENCE.tolist(),
                "M": M, "N": N, "N_TRIALS": N_TRIALS}
    with open(ROOT / "results_bayes.json", "w") as f:
        json.dump(out_json, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)

    # ---- figure -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sldp = [r for r in rows if not math.isinf(r["eps_s"])]
    eps = np.array([r["eps_s"] for r in sldp])

    for key, marker, color in [("lr", "o", "C0"), ("gbt", "s", "C1"),
                               ("bayes", "^", "C2")]:
        y = np.array([r[f"{key}_acc"] for r in sldp])
        ci = np.array([r[f"{key}_acc_ci"] for r in sldp])
        ax.errorbar(eps, y, yerr=ci, marker=marker, capsize=3, color=color,
                    label={"lr": "Logistic Regression",
                           "gbt": "Gradient Boosting",
                           "bayes": "Bayes-optimal"}[key])

    dap = rows[0]
    ax.axhline(dap["bayes_acc"], linestyle="--", color="C3",
               label=f"DAP baseline (Bayes) = {dap['bayes_acc']:.2f}")
    ax.axhline(PREVALENCE.max(), linestyle=":", color="gray",
               label=f"random guess = {PREVALENCE.max():.2f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"structural privacy budget $\varepsilon_s$")
    ax.set_ylabel("attack accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("SLDP defense vs. three attackers (incl. Bayes-optimal)")
    fig.tight_layout()
    fig.savefig(FIGS / "bayes_attack.pdf")
    fig.savefig(FIGS / "bayes_attack.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
