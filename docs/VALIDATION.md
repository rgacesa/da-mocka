# Local validation and disclosure diagnostics (0.0.5)

Structural input/output checks always run. Full statistical/privacy diagnostics are opt-in
because they can be computationally expensive and their reports contain sensitive cohort statistics.
Use a new report directory outside the candidate directory, on approved storage.

```bash
da-mocka generate --config work/hamsters/config.yaml --output work/mock --engine joint_latent --n-samples 5000 --audit work/local_audit

da-mocka validate --config work/hamsters/config.yaml --synthetic work/mock --report work/recheck
```

For stratified outputs repeat the exact stratification, selection, allocation and suppression
settings in the source YAML or CLI. The validator uses the source schema, not bounds or categories
declared by the candidate YAML. Candidate counts, IDs, taxonomy and compositions are checked.
Fidelity compares the eligible source to candidates globally (unweighted, so intentional balancing
may change this result), per represented stratum, and as a candidate-count-weighted mean of
conditional errors. Zero-output strata are omitted from conditional fidelity. Between-stratum
mean-abundance contrasts use the first represented stratum as reference. Gate thresholds for
fidelity use the worst represented stratum when stratified.

## Metrics
- Phenotype missingness error, categorical total variation (observed values), numeric KS and
  range-normalized Wasserstein distances; undefined comparisons are null.
- Taxon prevalence, mean abundance L1, zero fraction, mean CLR difference (pseudocount 1e-6).
- Shannon, Simpson and richness distributions; Bray-Curtis distributions sampled from distinct pairs.
- Spearman dependency errors for phenotype indicators, microbial presence/abundance and cross-modal
  correlations; constant or undefined pairs omitted. Taxa selected by source variance; phenotype
  indicators capped in schema order. These are diagnostics, not significance tests.
- Full-table hash exact overlap excluding identifiers, separately for phenotype, microbiome and
  combined records. Floating point exact matches are representation-sensitive; hash collision is possible.
- Capped nearest-neighbour distances using range-scaled mixed phenotype distance (missingness as
  mismatch), microbiome Bray-Curtis and their equally weighted mean. Report minima, 5th percentile
  and median. Full-source comparisons include excluded/suppressed rows. Conditional privacy reports
  compare within group; the full-source report also checks cross-group overlap.
- Rare observed categorical combinations below the configured source stratum threshold and unseen
  combinations. No guarantee of k-anonymity or protection from linkage.

## Independent holdout
Pass `--holdout path/to/config.yaml` to generate or validate. It must describe real data never used
for training, tuning, preprocessing decisions or selection; disjoint sample IDs are checked but
cannot prove participant independence. Supply representative, comparable holdout populations.
Membership inference uses negative nearest-candidate distance as attack score and AUC, with
equally sized capped samples of eligible training-reference and holdout rows. In pooled runs,
the training reference conservatively includes all eligible parent rows. For external candidates
you must correctly identify the fitting source and settings; the tool cannot verify provenance.
Candidate-to-holdout distances use the same reference sample size as the membership reference.
Attribute inference predicts configured categorical targets from nearest synthetic phenotypes
excluding the target itself, compared with the training-majority baseline. Other variables can
be proxies. No holdout means membership and attribute metrics are unavailable; attribute targets
require a holdout. These limited attacks are not formal privacy tests.

## Configurable gates
Optional source YAML:
```yaml
validation:
  max_samples: 1000
  max_features: 64
  seed: 42
  attribute_targets: [disease]
  gates:
    max_prevalence_mae: 0.10
    max_joint_match_rate: 0.0
    max_membership_auc: 0.65
```

Supported gates: max_prevalence_mae, max_phenotype_ks_mean, max_cross_modal_mae,
max_joint_match_rate, max_rare_match_rate, min_joint_p05, max_membership_auc,
max_attribute_advantage. Membership uses the worst direction (max(AUC, 1-AUC)) across modalities.
Attribute uses worst configured target advantage. Global and conditional overlap/distance gates
take the worst result; unavailable configured metrics fail closed. No thresholds are defaults;
the examples are illustrations only, not scientifically established release criteria.
Configured gates require --audit during generation. If they fail, generation removes the candidate
directory, retains the audit, and returns nonzero. Standalone validation never deletes its input.
Passing does not authorize publication. The manifest always says unreviewed_not_anonymous.

Reports are JSON with source counts and group/feature names: sensitive, local only. Do not ship them.
The audit cap bounds pairwise distance matrices (up to 5000 rows per side; lower for memory-limited
systems); exact overlap and marginal statistics use all rows. Full source and candidate matrices
remain in memory. Advanced attack calibration, cross-validated classifier tests, multiple-testing
control, robust log-ratio differential abundance and formal differential privacy remain future work.

Since 0.0.6, every expected rank export must agree with the canonical terminal components. Rank-wise prevalence and mean-abundance fidelity are reported, globally and per stratum. See [taxonomy](TAXONOMY.md).
