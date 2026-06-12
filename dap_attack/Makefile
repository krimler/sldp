# DAP / SLDP artifact build.
#
# Targets:
#   make            -> all (install, verify, test, figures)
#   make install    -> create .venv and install pinned deps
#   make verify     -> run conformance checker (37 checks)
#   make test       -> pytest
#   make coverage   -> pytest with coverage report
#   make figures    -> run the 9 result scripts (regenerates JSON + PDFs)
#   make clean      -> remove pytest build artifacts
#   make distclean  -> also remove .venv and generated outputs

PY := .venv/bin/python
PIP := .venv/bin/pip

SCRIPTS := \
    attack_sim.py \
    value_accuracy.py \
    bayes_attack.py \
    hybrid_pipeline.py \
    noncollusion_attack.py \
    defense_comparison.py \
    scale_sparsity.py \
    real_schema_attack.py

.PHONY: all install verify test coverage figures clean distclean

all: install verify test figures

install: .venv/.installed
.venv/.installed: requirements.txt
	test -d .venv || python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@touch $@

verify: install
	$(PY) verify.py

test: install
	$(PY) -m pytest tests/ -q

coverage: install
	$(PY) -m pytest tests/ --cov --cov-config=.coveragerc --cov-report=term-missing

figures: install
	@for s in $(SCRIPTS); do \
	    echo "==> $$s"; \
	    $(PY) $$s || exit 1; \
	done

clean:
	rm -rf __pycache__ tests/__pycache__ .pytest_cache .coverage

distclean: clean
	rm -rf .venv results_*.json figures/*.pdf figures/*.png
