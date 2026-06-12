# DAP structural-leakage attack and SLDP defense

Artifact for *On Structural Local Differential Privacy* (CCS '26).
Generates the figures and numbers cited in the paper, plus a test
suite that checks every mechanism.

DAP target: `draft-ietf-ppm-dap-17` (2026-01-30). VDAF target:
`draft-irtf-cfrg-vdaf-18`. Conformance details in `CONFORMANCE.md`.

## Run

Single command builds everything (`.venv`, conformance, tests,
all 9 result scripts):

```
make
```

Individual targets:

```
make install     # create .venv, install pinned deps
make verify      # 37 conformance checks (~10s)
make test        # 54 pytest tests (~20s)
make coverage    # pytest with line-coverage report
make figures     # run the 9 result scripts (~10min total)
make clean       # remove pytest build artifacts
make distclean   # also remove .venv, results_*.json, figures/*
```

If you prefer to drive the scripts directly without `make`:

```
.venv/bin/python verify.py
.venv/bin/python -m pytest tests/
.venv/bin/python attack_sim.py             # Result 1+2
.venv/bin/python value_accuracy.py         # Result 3
.venv/bin/python bayes_attack.py           # Result 4 (~5min)
.venv/bin/python hybrid_pipeline.py        # Result 5
.venv/bin/python noncollusion_attack.py    # Result 6
.venv/bin/python defense_comparison.py     # Result 7
.venv/bin/python scale_sparsity.py         # Result 8 (~3min)
.venv/bin/python real_schema_attack.py     # Result 9
```

## Files

- `attack_sim.py` — Result 1 (DAP plaintext leak) + Result 2 (SLDP defense).
- `value_accuracy.py` — Result 3 (DAP vs SLDP value MSE).
- `bayes_attack.py` — Result 4 (LR / GBT / Bayes-optimal attackers).
- `hybrid_pipeline.py` — Result 5 (DAP, SLDP, hybrid frontier).
- `noncollusion_attack.py` — Result 6 (leak under non-collusion).
- `defense_comparison.py` — Result 7 (SLDP vs simpler defenses).
- `scale_sparsity.py` — Result 8 (m × sparsity, cube-root rule).
- `real_schema_attack.py` — Result 9 (real Firefox schema).
- `firefox_schema.py` — 66 real probe names from `mozilla/gecko-dev`,
  audited 2026-04-29.
- `verify.py` — analytical and Monte-Carlo conformance checks.
- `tests/` + `.coveragerc` — pytest suite, 99% line coverage.
- `figures/` — every PDF cited in the paper.
- `results_*.json` — raw numbers for every cited table.
- `requirements.txt` — pinned Python dependencies.
- `Makefile` — single-command driver for every target.
- `CONFORMANCE.md` — DAP threat-model and SLDP-mechanism mapping.

## Headline numbers

| | Attack acc. | Value MSE |
|---|---|---|
| DAP-only (collusion adversary) | 0.989 | 1.0e-4 |
| SLDP-only (ε_s=0.25, ε_v=1) | 0.445 | 3.9e-3 |
| **Hybrid (ε_s=0.25)** | **0.439** | **8.6e-5** |

Real Firefox schema (66 probes, 5 flavors): DAP attack 0.970, SLDP at
ε_s=0.25 drops to 0.518.

## License

Code: MIT. Data / figures: CC-BY 4.0. Seeds fixed; runs deterministic.
