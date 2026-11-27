# Stratification (0.0.3)

Fit an independent baseline within each observed categorical group, using aligned phenotype and microbiome rows. No unobserved joint strata are invented. The selected grouping columns are assigned exactly; all other columns are sampled from the applicable model. Cross-feature associations within a group are not explicitly modelled.

## CLI
```bash
da-mocka generate --config work/hamsters/config.yaml --output work/disease \
  --stratify disease --include 'disease=[A,B]' --n-samples 10000

da-mocka generate --config work/hamsters/config.yaml --output work/disease_sex \
  --stratify disease sex --include disease=A,B --balance disease --n-samples 10000
```

Repeated --include options select different stratification columns (AND across columns, OR within a column). Use YAML for category labels that contain commas/brackets. --balance implies marginal allocation unless explicitly overridden.

## YAML
Add to the existing input config:
```yaml
stratification:
  columns: [disease, sex]
  include:
    disease: [A, B]
  allocation: marginal
  balance: [disease]
  min_stratum_size: 30
  rare_strata: error
  missing_strata: error
  pooling_strength: 0
```

Allocation modes:
- proportional: preserve eligible joint group proportions with exact integer total.
- balanced: equal counts across eligible joint strata, differing by at most one.
- marginal: balance observed joint combinations of the selected balance columns, retaining conditional proportions of remaining columns (not iterative marginal raking).
- original: retain eligible source counts; omit --n-samples or supply the exact matching total.
- custom: explicit counts; omit --n-samples or supply their sum. Omitted groups receive zero. Unavailable strata are errors, even if requested count is zero.

Example custom section:
```yaml
stratification:
  columns: [disease, sex]
  allocation: custom
  custom:
    - values: [A, female]
      count: 1000
    - values: [B, male]
      count: 500
```

The values follow the configured column order. All count allocations use deterministic largest-remainder rounding; tiny requested totals can give some eligible strata zero output.

## Sparse and missing strata
Default minimum source group size is 30, a heuristic and not a privacy guarantee. Default policy fails if any selected observed group is below the threshold; rare_strata: suppress explicitly excludes small groups and reallocates the total, warning locally. Suppressed groups do not enter parent fits. Missing stratification values fail by default; missing_strata: exclude drops them with a warning. A missing value on an explicitly filtered column is outside that selection.

Optional pooling_strength > 0 enables a hierarchical mixture: a group with n eligible source rows uses its own independent model with probability n/(n+pooling_strength); otherwise it tries the prefix parent (drop the last stratification column), recursively up to the eligible population. Group labels remain fixed; parent models may introduce features or other phenotype values not observed in the specific group. Pooling is off by default, does not rescue suppressed groups, and is not formal privacy protection. Earlier columns in the list have priority for fallback. Parent fits use all eligible groups including those allocated zero output, never filtered or suppressed groups.

Generated tables are shuffled jointly and contain no source IDs. The manifest records allocation mode and grouping column names, not source stratum counts. Output config is for validation and intentionally omits generation settings. All outputs remain local unreviewed candidates.
