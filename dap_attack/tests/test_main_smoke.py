"""Smoke tests for each main(). Runs are tiny so coverage stays cheap."""

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_outputs(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    for fname in os.listdir(ROOT):
        src = ROOT / fname
        if src.is_file() and (src.suffix == ".py"
                               or src.name in ("requirements.txt", ".coveragerc")):
            os.symlink(src, work / fname)
    yield


def test_attack_sim_main(monkeypatch):
    import attack_sim
    monkeypatch.setattr(attack_sim, "N", 200)
    monkeypatch.setattr(attack_sim, "N_TRIALS", 2)
    attack_sim.main()


def test_bayes_attack_main(monkeypatch):
    import bayes_attack
    monkeypatch.setattr(bayes_attack, "N", 200)
    monkeypatch.setattr(bayes_attack, "N_TRIALS", 2)
    bayes_attack.main()


def test_value_accuracy_main():
    import value_accuracy
    value_accuracy.main()


def test_hybrid_pipeline_main(monkeypatch):
    import hybrid_pipeline
    monkeypatch.setattr(hybrid_pipeline, "N", 200)
    monkeypatch.setattr(hybrid_pipeline, "N_TRIALS", 2)
    hybrid_pipeline.main()


def test_noncollusion_attack_main(monkeypatch):
    import noncollusion_attack
    monkeypatch.setattr(noncollusion_attack, "N", 300)
    monkeypatch.setattr(noncollusion_attack, "N_TRIALS", 2)
    noncollusion_attack.main()


def test_defense_comparison_main(monkeypatch):
    import defense_comparison
    monkeypatch.setattr(defense_comparison, "N", 300)
    monkeypatch.setattr(defense_comparison, "N_TRIALS", 2)
    defense_comparison.main()


def test_scale_sparsity_main(monkeypatch):
    import scale_sparsity
    monkeypatch.setattr(scale_sparsity, "N", 800)
    monkeypatch.setattr(scale_sparsity, "N_TRIALS", 2)
    monkeypatch.setattr(scale_sparsity, "EPS_S_GRID",
                        np.linspace(0.3, 1.7, 5))
    scale_sparsity.main()


def test_real_schema_attack_main(monkeypatch):
    import real_schema_attack
    monkeypatch.setattr(real_schema_attack, "N", 300)
    monkeypatch.setattr(real_schema_attack, "N_TRIALS", 2)
    real_schema_attack.main()


def test_verify_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify.py")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL CHECKS PASSED" in proc.stdout
