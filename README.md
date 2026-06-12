# Structural Local Differential Privacy — artifact

Code and results for *On Structural Local Differential Privacy*. Everything is
seeded and deterministic. The repository has two independent parts.

## `dap_attack/`

The empirical evaluation. A colluding-aggregator attack on a
`draft-ietf-ppm-dap-17` deployment recovers device class from structural
metadata alone (99.0% accuracy), and an SLDP layer on the structure vector
suppresses it to near random-guess at ε_s = 0.25. Includes nine result scripts
(every figure and number cited in the paper), a conformance checker against the
IETF DAP/VDAF drafts, and a pytest suite.

Conformance to the IETF drafts (`draft-ietf-ppm-dap-17`,
`draft-irtf-cfrg-vdaf-18`, `draft-ietf-ppm-dap-taskprov-03`,
`draft-thomson-ppm-dap-dp-ext-00`) is documented field-by-field in
`dap_attack/CONFORMANCE.md`. Build with `make` (see `dap_attack/README.md`).

## `rate_validation/`

A focused check that the debiased randomized-response estimator attains the
theoretical structure rate Θ(m / (n·ε_s²)) — slope −1 on a log-log plot of MSE
against the number of users. See `rate_validation/README.md`.

## License

Code: MIT. Data and figures: CC-BY 4.0.
