# Implementation roadmap

## Objectives
Validity (hard biological constraints), fidelity (population distributions and dependencies), privacy (assess disclosure without claiming anonymity). CPU-first Python package, CLI and API; local operation, no telemetry.

| Milestone | Scope |
|---|---|
| 0.0.1 (implemented) | Package, strict YAML and table contracts, demo cohort, pytest, Ruff, CI |
| 0.0.2 (implemented) | Independent baseline, bounded phenotype generation, constraints and taxonomy aggregation |
| 0.0.3 (implemented) | Outcome selection, joint categorical strata, proportional/original/balanced/custom/marginal allocation, sparse-stratum policies |
| 0.0.4 (implemented; experimental) | Joint latent model, correlated presence, log-abundance, phenotypes and missingness |
| 0.0.5 | Statistical fidelity, privacy diagnostics, export gates and reports |
| 0.1.0 | Integration, benchmarks, documentation and release review |

## Statistical design (planned)
Baseline models marginals by stratum. Primary candidate: regularized low-rank latent Gaussian model with hurdle presence and positive log-abundance, bounded phenotype marginals and categorical decoding. Sample new latent vectors, never simply copy rows. Normalize masked positive abundances to sum one; reject all-zero draws. Generate terminal taxonomic features and aggregate ancestors. Preserve correlations and missingness jointly across modalities. Validate approximation on artificial data before real cohorts.

Stratify by disease, disease plus sex, or other declared categorical variables. Explicitly select outcomes. Do not invent group combinations by default. Allow balanced or custom counts; deterministic integer allocation. Partial pooling toward parent strata with candidate weight n/(n+lambda); suppress rare groups separately, since pooling does not guarantee privacy. Minimum group size 30 is a starting heuristic, not a safety guarantee. Continuous conditioning and longitudinal data are later work.

## Acceptance and benchmarks
Hard checks: finite nonnegative abundance, sum one, valid categories and bounds, consistent derived phenotypes, taxonomy, ID alignment and fresh IDs. Hard failures block export. Fidelity: prevalence, missingness, quantiles, diversity, Bray-Curtis, log ratios, correlations, disease effect sizes, real-vs-synthetic classifier. Compare globally and per stratum; account for intentional rebalancing. Privacy: exact matches, nearest neighbors against holdout, membership/attribute inference, rare combinations. Never infer anonymity from these alone.

Cosmic hamster reference: artificial disease signals, sparse communities, correlated phenotypes and missingness. Scale sample count and feature count independently (100 to 1M samples, 100 to 10k taxa, subject to resource limits); record runtime, peak RAM, throughput, fidelity. Batch generation and avoid dense feature covariance where possible.

Future export layout: synthetic tables plus screened manifest in synthetic/; sensitive diagnostics and fitted parameters stay in local_audit/. 0.0.4 writes local candidates only; disclosure diagnostics and release gates are not implemented.
