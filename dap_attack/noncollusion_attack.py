"""Result 6: leakage from DAP-17 S4.5.2 visible-metadata under non-collusion.
Channels: task vector, submission rate, public_share length, batch."""

import json
import math
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

# ---- deployment topology ---------------------------------------------
T = 20            # number of DAP tasks the adversary sees
N = 10_000        # clients
N_TRIALS = 5

# Per-class task-subscription probabilities. Each class participates in
# a class-distinctive subset of tasks (e.g., only Mobile reports to the
# tasks tied to mobile-only features). We over-load the same 5 device
# classes used in the rest of the bundle so cross-section comparisons
# are clean.
CLASS_PROFILES = np.array([
    # 20 tasks: t0..t19
    [0.95, 0.90, 0.85, 0.80, 0.10, 0.05, 0.05, 0.10, 0.50, 0.50,
     0.05, 0.05, 0.95, 0.90, 0.05, 0.05, 0.10, 0.10, 0.40, 0.30],   # desktop
    [0.10, 0.10, 0.85, 0.80, 0.95, 0.90, 0.05, 0.10, 0.50, 0.50,
     0.05, 0.05, 0.10, 0.10, 0.95, 0.90, 0.10, 0.10, 0.05, 0.05],   # mobile
    [0.95, 0.90, 0.10, 0.10, 0.10, 0.05, 0.95, 0.90, 0.50, 0.50,
     0.05, 0.05, 0.10, 0.90, 0.05, 0.05, 0.10, 0.10, 0.30, 0.20],   # tablet
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95,
     0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.05, 0.05],   # iot
    [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.95, 0.05,
     0.95, 0.05, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.95, 0.95],   # researcher
])
PREVALENCE = np.array([0.40, 0.30, 0.20, 0.07, 0.03])
C = len(PREVALENCE)

# ---- public_share / batch / rate auxiliary leakage strength ----------
# Models VDAF-specific structure: e.g., Mastic prefix-tree depth varies
# by client population. We model these as WEAK side channels, modulated
# by `SIDE_LEAK` in [0, 1] -- 0 = side channels carry no class info
# (e.g., Prio3 with fixed-size shares); 1 = strongly class-correlated
# (Mastic-like). This lets us isolate the task-vector leakage.
SIDE_LEAK = 0.20


def sample_population(rng):
    classes = rng.choice(C, size=N, p=PREVALENCE)
    profiles = CLASS_PROFILES[classes]                # (N, T)
    Tparticipation = (rng.random((N, T)) < profiles).astype(np.int8)
    # Submission rate per task: weakly class-conditional Poisson.
    base_rate = 3.0
    class_factor = 1.0 + SIDE_LEAK * (CLASS_PROFILES[classes] - 0.5)
    rates = profiles * base_rate * class_factor
    counts = rng.poisson(np.maximum(rates, 0.01)).astype(np.int32)
    # public_share length: weakly class-conditional 3-bin categorical.
    pshare = np.zeros((N, T), dtype=np.int8)
    for c in range(C):
        idx = (classes == c)
        if idx.sum() == 0:
            continue
        for t in range(T):
            p_active = CLASS_PROFILES[c, t]
            base = np.array([1/3, 1/3, 1/3])
            tilt = np.array([0.5*(1-p_active), 0.4, 0.1 + 0.5*p_active])
            tilt = tilt / tilt.sum()
            probs = (1 - SIDE_LEAK) * base + SIDE_LEAK * tilt
            pshare[idx, t] = rng.choice(3, size=idx.sum(), p=probs)
    # Batch ID: 4 batches, weakly class-conditional preference.
    base_batch = np.full(4, 0.25)
    tilts = np.array([
        [0.6, 0.2, 0.1, 0.1],
        [0.1, 0.4, 0.4, 0.1],
        [0.3, 0.3, 0.2, 0.2],
        [0.1, 0.1, 0.4, 0.4],
        [0.25, 0.25, 0.25, 0.25],
    ])
    batch = np.zeros(N, dtype=np.int8)
    for c in range(C):
        idx = np.where(classes == c)[0]
        probs = (1 - SIDE_LEAK) * base_batch + SIDE_LEAK * tilts[c]
        batch[idx] = rng.choice(4, size=len(idx), p=probs)
    return classes, Tparticipation, counts, pshare, batch


def featurize_attacker_view(Tparticipation, counts, pshare, batch,
                            include=("task", "rate", "pshare", "batch")):
    feats = []
    if "task" in include:
        feats.append(Tparticipation.astype(float))                 # (N, T)
    if "rate" in include:
        feats.append(np.log1p(counts).astype(float))               # (N, T)
    if "pshare" in include:
        # One-hot per task: 3 length bins
        oh = np.zeros((pshare.shape[0], T * 3), dtype=float)
        for t in range(T):
            for b in range(3):
                oh[:, t*3 + b] = (pshare[:, t] == b).astype(float)
        feats.append(oh)
    if "batch" in include:
        bb = np.zeros((batch.shape[0], 4), dtype=float)
        for b in range(4):
            bb[:, b] = (batch == b).astype(float)
        feats.append(bb)
    return np.hstack(feats)


def attack_lr(X_train, y_train, X_test, y_test):
    sc = StandardScaler(with_mean=False)
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    clf.fit(Xtr, y_train)
    proba = clf.predict_proba(Xte)
    pred = proba.argmax(1)
    acc = (pred == y_test).mean()
    auc = roc_auc_score((y_test == 4).astype(int), proba[:, 4])
    return float(acc), float(auc)


def sldp_task_vector(Tparticipation, eps_s, rng):
    a = math.exp(eps_s) / (math.exp(eps_s) + 1.0)
    keep = rng.random(Tparticipation.shape) < a
    return np.where(keep, Tparticipation, 1 - Tparticipation).astype(np.int8)


def run_one(eps_s_or_inf, ablation, seed):
    rng = np.random.default_rng(seed)
    classes, Tp, counts, pshare, batch = sample_population(rng)
    if math.isinf(eps_s_or_inf):
        Tp_obs = Tp
    else:
        Tp_obs = sldp_task_vector(Tp, eps_s_or_inf, rng)

    X = featurize_attacker_view(Tp_obs, counts, pshare, batch,
                                include=ablation)
    perm = rng.permutation(N)
    split = int(0.7 * N)
    tr, te = perm[:split], perm[split:]
    return attack_lr(X[tr], classes[tr], X[te], classes[te])


def main():
    eps_grid = [0.25, 0.5, 1.0, 2.0, 4.0]

    # --- main result: which channels matter? --------------------------
    ablations = [
        ("task only",                    ("task",)),
        ("task + pshare",                ("task", "pshare")),
        ("task + rate",                  ("task", "rate")),
        ("task + batch",                 ("task", "batch")),
        ("all visible metadata (DAP-17 S4.5.2)",
                                         ("task", "rate", "pshare", "batch")),
    ]
    print("=" * 90)
    print("Single honest-but-curious aggregator, no collusion (DAP-17 S2.1 hypothesis HOLDS).")
    print("Adversary view is restricted to fields enumerated public in DAP-17 S4.5.2.")
    print("=" * 90)
    print(f"{'Channels':>40} | {'attack_acc':>14} | rare AUC")
    print("-" * 90)

    rows_ablation = []
    for label, ablation in ablations:
        accs, aucs = [], []
        for s in range(N_TRIALS):
            a, u = run_one(math.inf, ablation, s)
            accs.append(a); aucs.append(u)
        acc_m = np.mean(accs); acc_ci = 1.96*np.std(accs, ddof=1)/math.sqrt(N_TRIALS)
        auc_m = np.mean(aucs)
        rows_ablation.append({"label": label, "channels": list(ablation),
                              "acc_mean": float(acc_m), "acc_ci": float(acc_ci),
                              "rare_auc_mean": float(auc_m)})
        print(f"{label:>40} | {acc_m:.3f} +/- {acc_ci:.3f} | {auc_m:.3f}")

    # --- TWO SLDP defenses --------------------------------------------
    # D-partial: SLDP only on task vector; side channels remain visible.
    # D-full   : SLDP on task vector AND drop side channels from view
    #            (operational equivalent: clients submit at fixed rate,
    #            with fixed-size public_share, into a uniformly-random
    #            batch; only the SLDP-randomised task vector is left).
    print()
    print("Two defense modes:")
    print("  partial = SLDP RR on task vector; side channels still visible")
    print("  full    = SLDP RR on task vector AND randomise/pad rate, "
          "pshare, batch")
    print(f"{'eps_s':>8} | {'partial':>16} | {'full':>16}")
    print("-" * 60)
    rows_defense = []
    # No-defense baseline.
    accs_b, _ = ([], [])
    for s in range(N_TRIALS):
        a, _ = run_one(math.inf, ablations[-1][1], s)
        accs_b.append(a)
    rows_defense.append({"eps_s": float("inf"), "mode": "no_defense",
                         "acc_mean": float(np.mean(accs_b)),
                         "acc_ci": float(1.96*np.std(accs_b, ddof=1)/math.sqrt(N_TRIALS))})
    print(f"{'inf':>8} | "
          f"{np.mean(accs_b):.3f} +/- "
          f"{1.96*np.std(accs_b, ddof=1)/math.sqrt(N_TRIALS):.3f} | "
          f"(no defense)")

    for eps in eps_grid:
        # partial: full attacker view with eps_s on task vector
        accs_p = [run_one(eps, ("task", "rate", "pshare", "batch"), s)[0]
                  for s in range(N_TRIALS)]
        # full: only task vector visible (the rest is randomised/padded
        # so it carries no class info).
        accs_f = [run_one(eps, ("task",), s)[0] for s in range(N_TRIALS)]
        rows_defense.append({
            "eps_s": eps, "mode": "partial",
            "acc_mean": float(np.mean(accs_p)),
            "acc_ci": float(1.96*np.std(accs_p, ddof=1)/math.sqrt(N_TRIALS)),
        })
        rows_defense.append({
            "eps_s": eps, "mode": "full",
            "acc_mean": float(np.mean(accs_f)),
            "acc_ci": float(1.96*np.std(accs_f, ddof=1)/math.sqrt(N_TRIALS)),
        })
        print(f"{eps:>8.2f} | "
              f"{np.mean(accs_p):.3f} +/- "
              f"{1.96*np.std(accs_p, ddof=1)/math.sqrt(N_TRIALS):.3f} | "
              f"{np.mean(accs_f):.3f} +/- "
              f"{1.96*np.std(accs_f, ddof=1)/math.sqrt(N_TRIALS):.3f}")

    out = {"T": T, "N": N, "N_TRIALS": N_TRIALS,
           "ablation_no_sldp": rows_ablation,
           "sldp_defense": rows_defense,
           "random_guess": float(PREVALENCE.max())}
    with open(ROOT / "results_noncollusion.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.0))

    # Left: ablation under collusion-free DAP.
    labels = [r["label"] for r in rows_ablation]
    accs = [r["acc_mean"] for r in rows_ablation]
    cis = [r["acc_ci"] for r in rows_ablation]
    y = np.arange(len(labels))
    ax1.barh(y, accs, xerr=cis, color="C0", alpha=0.85, capsize=3)
    ax1.axvline(PREVALENCE.max(), linestyle=":", color="gray",
                label=f"random guess = {PREVALENCE.max():.2f}")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("attack accuracy")
    ax1.set_xlim(0.3, 1.0)
    ax1.set_title("Non-collusion leak by metadata channel")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.invert_yaxis()

    # Right: two SLDP defenses on the same axis.
    base = [r for r in rows_defense if r.get("mode") == "no_defense"][0]
    partial = [r for r in rows_defense if r.get("mode") == "partial"]
    full = [r for r in rows_defense if r.get("mode") == "full"]
    eps_p = np.array([r["eps_s"] for r in partial])
    acc_p = np.array([r["acc_mean"] for r in partial])
    ci_p = np.array([r["acc_ci"] for r in partial])
    eps_f = np.array([r["eps_s"] for r in full])
    acc_f = np.array([r["acc_mean"] for r in full])
    ci_f = np.array([r["acc_ci"] for r in full])
    ax2.errorbar(eps_p, acc_p, yerr=ci_p, marker="o", capsize=3, color="C1",
                 label="partial: SLDP on task only (rate/pshare/batch leak)")
    ax2.errorbar(eps_f, acc_f, yerr=ci_f, marker="s", capsize=3, color="C0",
                 label="full: SLDP on task + side channels padded/randomised")
    ax2.axhline(base["acc_mean"], linestyle="--", color="C3",
                label=f"DAP no-defense = {base['acc_mean']:.2f}")
    ax2.axhline(PREVALENCE.max(), linestyle=":", color="gray",
                label=f"random guess = {PREVALENCE.max():.2f}")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\varepsilon_s$ on task-participation vector")
    ax2.set_ylabel("attack accuracy")
    ax2.set_ylim(0.3, 1.05)
    ax2.set_title("SLDP under non-collusion: partial vs full defense")
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "noncollusion_attack.pdf")
    fig.savefig(FIGS / "noncollusion_attack.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
