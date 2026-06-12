"""Tests for SLDP mechanisms and per-script helpers."""

import math
import pathlib
import sys
from collections import Counter

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import attack_sim
import bayes_attack
import value_accuracy
import hybrid_pipeline
import noncollusion_attack
import defense_comparison
import scale_sparsity
import real_schema_attack
import firefox_schema as fs

SEED = 0


# Bitwise RR -----------------------------------------------------------

@pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 2.0, 4.0])
def test_rr_keep_prob_matches_formula(eps):
    expected = math.exp(eps) / (math.exp(eps) + 1.0)
    for v in [attack_sim.rr_flip_prob(eps),
              bayes_attack.rr_keep_prob(eps),
              hybrid_pipeline.rr_keep(eps),
              real_schema_attack.rr_keep(eps)]:
        assert v == pytest.approx(expected, abs=1e-12)


def test_sldp_structure_outputs_binary():
    rng = np.random.default_rng(SEED)
    S = (rng.random((500, 16)) < 0.4).astype(np.int8)
    Y = attack_sim.sldp_structure(S, eps_s=1.0, rng=rng)
    assert Y.dtype == np.int8
    assert set(np.unique(Y).tolist()).issubset({0, 1})
    assert Y.shape == S.shape


def test_sldp_structure_empirical_flip_rate():
    rng = np.random.default_rng(SEED)
    S = np.zeros((100_000, 16), dtype=np.int8)
    Y = attack_sim.sldp_structure(S, eps_s=1.0, rng=rng)
    expected_flip = 1.0 / (math.exp(1.0) + 1.0)
    assert float(Y.mean()) == pytest.approx(expected_flip, abs=2e-3)


# k-RR -----------------------------------------------------------------

@pytest.mark.parametrize("K,eps", [(2, 0.5), (5, 1.0), (10, 2.0)])
def test_krr_keep_other_probs(K, eps):
    a = hybrid_pipeline.krr_keep(eps, K)
    b = hybrid_pipeline.krr_other(eps, K)
    assert a == pytest.approx(math.exp(eps) / (math.exp(eps) + K - 1))
    assert b == pytest.approx(1.0 / (math.exp(eps) + K - 1))
    assert a + (K - 1) * b == pytest.approx(1.0, abs=1e-12)
    assert math.log(a / b) == pytest.approx(eps, abs=1e-12)


def test_krr_estimate_unbiased():
    rng = np.random.default_rng(SEED)
    K, eps, n, R = 8, 1.5, 5_000, 100
    f_true = rng.dirichlet(np.ones(K))
    means = np.zeros(K)
    for _ in range(R):
        samples = rng.choice(K, size=n, p=f_true)
        means += value_accuracy.krr_estimate(samples, K, eps, rng)
    means /= R
    a = math.exp(eps) / (math.exp(eps) + K - 1)
    b = 1.0 / (math.exp(eps) + K - 1)
    se = 0.5 / ((a - b) * math.sqrt(R * n))
    assert np.max(np.abs(means - f_true)) < 4 * se


def test_krr_output_in_domain():
    rng = np.random.default_rng(SEED)
    K = 6
    samples = rng.integers(0, K, size=1000)
    out = hybrid_pipeline.randomize_value_krr(samples, K, 1.0, rng)
    assert out.min() >= 0 and out.max() < K
    assert out.shape == samples.shape


def test_dap_estimate_recovers_truth():
    rng = np.random.default_rng(SEED)
    K, n = 5, 1_000_000
    f_true = rng.dirichlet(np.ones(K))
    samples = rng.choice(K, size=n, p=f_true)
    f_hat = value_accuracy.dap_estimate(samples, K)
    assert np.max(np.abs(f_hat - f_true)) < 0.005


# Population samplers --------------------------------------------------

def test_attack_sim_population_shapes():
    rng = np.random.default_rng(SEED)
    classes, S = attack_sim.sample_population(500, rng)
    assert classes.shape == (500,)
    assert S.shape == (500, attack_sim.M)
    assert classes.min() >= 0 and classes.max() < len(attack_sim.PREVALENCE)


def test_attack_sim_seeded_reproducible():
    c1, S1 = attack_sim.sample_population(200, np.random.default_rng(SEED))
    c2, S2 = attack_sim.sample_population(200, np.random.default_rng(SEED))
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(S1, S2)


def test_real_schema_sample_shape(monkeypatch):
    monkeypatch.setattr(real_schema_attack, "N", 200)
    classes, S = real_schema_attack.sample_population(np.random.default_rng(SEED))
    assert classes.shape == (200,)
    assert S.shape == (200, real_schema_attack.M)


def test_make_population_sparsity():
    rng = np.random.default_rng(SEED)
    _, S, profiles, prior, _, _ = scale_sparsity.make_population(
        m=64, sparsity=0.10, n=2000, rng=rng)
    assert S.shape == (2000, 64)
    assert profiles.shape == (5, 64)
    assert abs(sum(prior) - 1.0) < 1e-9
    assert 0.03 < S.mean() < 0.35


def test_make_population_dense_vs_sparse():
    _, S_dense, *_ = scale_sparsity.make_population(
        64, 1.0, 1000, np.random.default_rng(SEED))
    _, S_sparse, *_ = scale_sparsity.make_population(
        64, 0.01, 1000, np.random.default_rng(SEED + 1))
    assert S_dense.mean() > S_sparse.mean()


# Bayes attacker -------------------------------------------------------

def test_bayes_proba_in_simplex():
    rng = np.random.default_rng(SEED)
    classes, S = bayes_attack.sample_population(300, rng)
    pred, proba = bayes_attack.attacker_bayes(
        S, eps_s=1.0,
        profiles=bayes_attack.DEVICE_PROFILES,
        prior=bayes_attack.PREVALENCE)
    assert proba.shape == (300, len(bayes_attack.PREVALENCE))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert pred.min() >= 0 and pred.max() < proba.shape[1]


def test_bayes_high_eps_recovers_class():
    rng = np.random.default_rng(SEED)
    classes, S_true = bayes_attack.sample_population(500, rng)
    S_obs = bayes_attack.sldp_structure(S_true, 8.0, rng)
    pred, _ = bayes_attack.attacker_bayes(
        S_obs, 8.0,
        bayes_attack.DEVICE_PROFILES, bayes_attack.PREVALENCE)
    assert (pred == classes).mean() > 0.85


# Defense comparison ---------------------------------------------------

def _defense_pop():
    rng = np.random.default_rng(SEED)
    classes, S, theta = defense_comparison.sample_population(rng)
    return rng, classes, S, theta


def test_d_no_defense_passthrough():
    rng, _, S, _ = _defense_pop()
    S_obs, mask, theta = defense_comparison.D_no_defense(S, rng)
    np.testing.assert_array_equal(S_obs, S)
    assert mask.all()


def test_d_pad_ones():
    rng, _, S, _ = _defense_pop()
    S_obs, mask, theta = defense_comparison.D_pad_ones(S, rng)
    assert (S_obs == 1).all()
    assert (theta == 1).all()


def test_d_pad_pattern_alternates():
    rng, _, S, _ = _defense_pop()
    expected = np.array([1, 0] * (S.shape[1] // 2))
    S_obs, _, theta = defense_comparison.D_pad_pattern(S, rng)
    np.testing.assert_array_equal(S_obs[0], expected)
    np.testing.assert_array_equal(theta, expected.astype(float))


def test_d_k_anonymity_drops_minorities():
    rng, _, S, _ = _defense_pop()
    _, mask, _ = defense_comparison.D_k_anonymity(S, rng, k=10)
    ctr = Counter(tuple(r) for r in S)
    for i, keep in enumerate(mask):
        if keep:
            assert ctr[tuple(S[i])] >= 10


def test_d_random_drop_uses_sentinel():
    rng, _, S, _ = _defense_pop()
    S_obs, _, _ = defense_comparison.D_random_drop(S, rng, delta=0.5)
    unique = set(np.unique(S_obs).tolist())
    assert -1 in unique
    assert 0.4 < float((S_obs == -1).mean()) < 0.6


def test_d_sldp_returns_int8():
    rng, _, S, _ = _defense_pop()
    S_obs, _, theta = defense_comparison.D_sldp(S, rng, eps_s=1.0)
    assert S_obs.dtype == np.int8
    assert (theta >= 0).all() and (theta <= 1).all()


def test_population_mse_zero():
    v = np.array([0.1, 0.2, 0.7])
    assert defense_comparison.population_mse(v, v) == pytest.approx(0)


# Hybrid pipeline ------------------------------------------------------

def test_hybrid_dap_estimate_simplex():
    rng = np.random.default_rng(SEED)
    f = hybrid_pipeline.estimate_freq_dap(rng.integers(0, 5, size=1000), 5)
    assert f.shape == (5,)
    assert f.sum() == pytest.approx(1.0, abs=1e-9)


def test_hybrid_sldp_estimate_unbiased():
    rng = np.random.default_rng(SEED)
    K, eps, n = 4, 2.0, 200_000
    f_true = np.array([0.1, 0.2, 0.3, 0.4])
    X = rng.choice(K, size=n, p=f_true)
    X_obs = hybrid_pipeline.randomize_value_krr(X, K, eps, rng)
    f_hat = hybrid_pipeline.estimate_freq_sldp(X_obs, K, eps)
    assert np.max(np.abs(f_hat - f_true)) < 0.02


def test_hybrid_run_pipeline_each_branch(monkeypatch):
    monkeypatch.setattr(hybrid_pipeline, "N", 1000)
    for name in ["DAP", "SLDP", "Hybrid"]:
        acc, mse = hybrid_pipeline.run_pipeline(
            name, 1.0, 1.0, np.random.default_rng(SEED))
        assert 0 <= acc <= 1 and mse >= 0


def test_hybrid_run_pipeline_unknown_raises():
    with pytest.raises(ValueError):
        hybrid_pipeline.run_pipeline("UNKNOWN", 1.0, 1.0,
                                     np.random.default_rng(SEED))


# Cube-root rule -------------------------------------------------------

@pytest.mark.parametrize("m,K,eps", [(64, 10, 2.0), (10, 100, 1.0)])
def test_cube_root_minimises_risk(m, K, eps):
    eps_s_star = scale_sparsity.cube_root_optimum(m, K, eps)
    grid = np.linspace(0.001, eps - 0.001, 4000)
    risk = m / grid ** 2 + K / (eps - grid) ** 2
    assert eps_s_star == pytest.approx(float(grid[np.argmin(risk)]), abs=2e-3)


def test_cube_root_extreme_m():
    assert scale_sparsity.cube_root_optimum(10_000_000, 10, 2.0) \
        == pytest.approx(2.0, abs=0.05)


def test_cube_root_extreme_K():
    assert scale_sparsity.cube_root_optimum(10, 10_000_000, 2.0) \
        == pytest.approx(0.0, abs=0.05)


# Non-collusion attack -------------------------------------------------

def test_noncollusion_population_arrays(monkeypatch):
    monkeypatch.setattr(noncollusion_attack, "N", 500)
    classes, Tp, counts, pshare, batch = \
        noncollusion_attack.sample_population(np.random.default_rng(SEED))
    assert classes.shape == (500,)
    assert Tp.shape == (500, noncollusion_attack.T)
    assert counts.shape == (500, noncollusion_attack.T)
    assert pshare.shape == (500, noncollusion_attack.T)
    assert batch.shape == (500,)
    assert (counts >= 0).all()


def test_featurize_ablations(monkeypatch):
    monkeypatch.setattr(noncollusion_attack, "N", 200)
    _, Tp, counts, pshare, batch = noncollusion_attack.sample_population(
        np.random.default_rng(SEED))
    T = noncollusion_attack.T
    X_task = noncollusion_attack.featurize_attacker_view(
        Tp, counts, pshare, batch, include=("task",))
    assert X_task.shape == (200, T)
    X_full = noncollusion_attack.featurize_attacker_view(
        Tp, counts, pshare, batch,
        include=("task", "rate", "pshare", "batch"))
    assert X_full.shape == (200, T + T + T * 3 + 4)


def test_sldp_task_vector_binary():
    rng = np.random.default_rng(SEED)
    Tp = (rng.random((500, 20)) < 0.5).astype(np.int8)
    Y = noncollusion_attack.sldp_task_vector(Tp, 1.0, rng)
    assert Y.shape == Tp.shape
    assert set(np.unique(Y).tolist()).issubset({0, 1})


# Firefox schema -------------------------------------------------------

def test_firefox_probe_count_and_unique():
    assert len(fs.PROBE_NAMES) >= 64
    assert len(set(fs.PROBE_NAMES)) == len(fs.PROBE_NAMES)


def test_firefox_flavors_and_prevalence():
    assert len(fs.FLAVORS) == 5
    assert len(fs.PREVALENCE) == 5
    assert sum(fs.PREVALENCE) == pytest.approx(1.0)


def test_firefox_profile_matrix():
    M = fs.build_profile_matrix()
    assert M.shape == (5, len(fs.PROBE_NAMES))
    assert (M >= 0).all() and (M <= 1).all()


def test_firefox_categories_resolve():
    for name in fs.PROBE_NAMES:
        assert fs._category_of(name) in fs.CATEGORY_RATES


def test_real_schema_attacker_bayes(monkeypatch):
    monkeypatch.setattr(real_schema_attack, "N", 300)
    classes, S = real_schema_attack.sample_population(
        np.random.default_rng(SEED))
    acc, auc = real_schema_attack.attacker_bayes(S, classes, eps_s=1.0)
    assert 0 <= acc <= 1
    assert 0 <= auc <= 1


# k-anonymity ----------------------------------------------------------

def test_k_anonymity_all_unique():
    n_uniq, singleton_frac, _ = attack_sim.k_anonymity(np.eye(8, dtype=np.int8))
    assert n_uniq == 8
    assert singleton_frac == 1.0


def test_k_anonymity_all_equal():
    n_uniq, singleton_frac, _ = attack_sim.k_anonymity(
        np.zeros((50, 4), dtype=np.int8))
    assert n_uniq == 1
    assert singleton_frac == 0.0
