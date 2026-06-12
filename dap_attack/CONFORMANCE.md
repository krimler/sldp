# Conformance statement

This bundle implements the SLDP mechanisms as defined in the
accompanying paper (*On Structural Local Differential Privacy*) and
models the DAP threat model as specified in the IETF draft cited
below. Every numerical claim in the paper is reproducible from this
bundle. Run:

```
make verify     # 37 conformance checks
make test       # pytest, 99% coverage
make figures    # all 9 result scripts
```

## Versions we conform to

| Component | Spec | Version pinned |
|----|----|----|
| DAP protocol | `draft-ietf-ppm-dap` | **17** (2026-01-30) |
| VDAF substrate | `draft-irtf-cfrg-vdaf` | **18** (2026-01) |
| Task binding | `draft-ietf-ppm-dap-taskprov` | **03** (2025-09) |
| DP extension | `draft-thomson-ppm-dap-dp-ext` | **00** (2024-10) |
| SLDP definition | paper Section 3 | this commit |
| SLDP mechanisms | paper Section 6 (Algorithm SLDP-Report) | this commit |

## SLDP implementation correctness

`verify.py` checks the following analytically and via Monte-Carlo:

**A. Structure mechanism** (paper Sec 6.1, per-bit binary randomised response):
- For each ε_s ∈ {0.1, 0.5, 1, 2, 4, 8}, the 2×2 transition matrix has
  max |log P/Q| equal to ε_s to within 10⁻⁹.
- Bit-wise composition: m independent ε_s-RR bits compose to m·ε_s-LDP
  under the worst-case all-flip adjacency, matching the paper's claim
  that single-bit changes consume only ε_s.
- The unbiased estimator p̂ᵢ = (Ȳ − (1−a)) / (2a−1) where
  a = e^{ε_s}/(e^{ε_s}+1) is unbiased to within Monte-Carlo SE
  (verified at n=2·10⁵, p_true=0.3).

**B. Value mechanism** (paper Sec 6.2, k-ary randomised response):
- For each (K, ε_v) ∈ {2,5,10}×{0.5,1,2}, log(a/b) equals ε_v exactly
  where a = e^{ε_v}/(e^{ε_v}+K−1) and b = 1/(e^{ε_v}+K−1).
- Empirical sampling at K=5, ε_v=1, n=5·10⁵ gives a max log-ratio of
  1.006 against the analytical bound of 1.0 (within Monte-Carlo slack).
- The unbiased estimator f̂ᵢ = (f_obsᵢ − b) / (a − b) is unbiased: with
  R=80 runs of n=5·10⁴, the absolute bias is 1.18·10⁻³ vs a 4·SE budget
  of 3.87·10⁻³.

## DAP threat-model fidelity

We do **not** re-implement DAP. Instead, we model the adversary view in
the regime where DAP fails to protect structure. The DAP draft (Sec 2.1
and Sec 8.7) makes two relevant claims:

1. **Non-collusion is the security hypothesis.** Under non-collusion,
   "given only one of the input shares, it is impossible to deduce the
   plaintext measurement." Our DAP baseline simulation does not target
   this regime — we model the regime *outside* the security hypothesis.

2. **Visible metadata under any threat model.** Even under non-collusion,
   the Task ID, public_share, report metadata (timestamps, extensions),
   and batch composition are visible to both aggregators in plaintext.
   Several of these encode structural information directly (the Task ID
   selects which schema is in force; public_share carries VDAF-specific
   structural data such as Mastic prefix-tree levels).

Our `attack_sim.py` operates in the **two-aggregator collusion regime**:
the adversary observes the plaintext (S, X) per client. This is the
regime explicitly excluded by DAP's security hypothesis, but is exactly
the regime SLDP is designed for (paper Sec 1.4, "compromised-server
threat models" and "single-server settings"). The empirical attack
quantifies what a colluding adversary can reconstruct, both with and
without an SLDP layer applied to S before sharding.

This bundle therefore demonstrates two things together:

- DAP-style deployments do not protect structure under collusion
  (`attack_sim.py`: 99.0% device-class accuracy from S alone).
- SLDP applied to S before sharding suppresses structural recovery to
  near random-guess at ε_s = 0.25 (44.5% accuracy) and to a meaningful
  fraction at ε_s = 1.0 (72.5% accuracy), at a value-accuracy cost of
  roughly 50× DAP's MSE at ε_v = 1.0 (`value_accuracy.py`).

## Non-collusion attack conformance

`noncollusion_attack.py` operates under DAP's stated security hypothesis
(non-collusion holds, encrypted shares are never decrypted) and uses
only the fields enumerated as public in `draft-ietf-ppm-dap-17`
§4.5.2:

| DAP field (per draft-17 §4.5.2) | Used in attack | Decrypted? |
|---|---|---|
| `task_id`                              | yes (task-participation vector) | n/a (public) |
| `report_metadata.time`                 | yes (batch assignment proxy)    | n/a (public) |
| `report_metadata.report_id`            | yes (client linking)            | n/a (public) |
| `public_share`                         | yes (length category, 3 bins)   | n/a (public) |
| `encrypted_input_shares[].payload`     | length only (not contents)      | **NO** |
| value plaintext                        | NO                              | NO |

The "full" SLDP defense in this experiment requires additional
client-side mechanisms beyond bit-wise RR on the schema vector:
fixed-rate dummy submission (per dap-dp-ext draft), fixed-size
public_share padding (a VDAF-level extension), and uniform-random batch
assignment. These extensions are not yet standardised; we model them as
"randomise/pad to a fixed distribution" in the experiment.

## Real-schema conformance

`real_schema_attack.py` uses 66 probe identifiers taken verbatim from
the upstream Firefox source tree. Reviewers can verify each identifier
appears in at least one of:

- `mozilla/gecko-dev:browser/components/metrics.yaml` (Glean)
- `mozilla/gecko-dev:toolkit/components/telemetry/Histograms.json`
  (legacy)

Both files are public on the `master` branch of `mozilla/gecko-dev`.
The probe list is enumerated in `firefox_schema.py::PROBE_NAMES`; we
encourage reviewers to grep the upstream files for each name as part
of artefact evaluation.

We do **not** claim the per-flavor probe-fill probabilities match
Mozilla's real population distribution; that data is not public. The
*schema* is what the DAP/Glean wire format exposes to an adversary,
and that is real.

## Cube-root rule conformance

`verify.py` runs five analytical checks of the closed-form cube-root
optimum at (m, K, ε) tuples spanning the experimental sweep.
The grid-argmin matches the cube-root formula to within 10⁻² in
each case. Experimental deviations from the rule
(`scale_sparsity.py` results) reflect the per-mechanism Fisher-information
constants treated as O(1) in the budget-allocation proposition; the more
general allocation result absorbs these.

## What `verify.py` does NOT cover

- We do not implement the DAP wire format (HPKE encryption, Prio3
  shares, Mastic preparation rounds). The DAP threat-model claim is
  supported by quoting the draft, not by re-running the protocol.
- The pan-privacy / intrusion-resilience comparison
  (paper Sec 9, Theorem on incomparability with Feldman et al. 2025) is
  proven analytically and not the subject of `verify.py`.

## License

- Code: MIT.
- Tex / data / figures: CC-BY 4.0.
- All randomness is seeded; runs are deterministic.
