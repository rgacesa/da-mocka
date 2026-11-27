# Independent engine (0.0.2)

The deliberately simple reference baseline is CPU-only. Each nonidentifier phenotype is modelled independently: categorical and integer values are drawn from empirical frequencies, continuous values from linear interpolation between sorted observations, and nullable fields have independent Bernoulli missingness. An all-missing field stays missing. No new categories are invented. The observed support is not a privacy guarantee.

Microbiome generation draws presence independently for each taxon with its observed prevalence. Positive abundances use interpolation in log space. All-zero communities are redrawn (bounded retries), and nonempty rows are normalized. Taxa never observed remain absent. Closure and rejection can change marginal prevalence/abundance; this baseline does not promise exact fidelity. Genera and families are sums of terminal features using the validated mapping.

Optional BMI constraint:
```yaml
bmi:
  target: BMI
  weight: weight
  height: height
  atol: 0.1
```
Declare those columns as positive nonnullable continuous variables with explicit cohort-appropriate bounds. Weight is kg, height metres. Source rounding is tolerated by atol; candidate BMI is calculated from sampled weight and height, and invalid pairs are redrawn. Infeasible generation fails after bounded retries. Without this declaration BMI is just another marginal; no biological rule is inferred from a name.

```bash
da-mocka generate --config work/hamsters/config.yaml --output work/mock --n-samples 5000
```

Output: phenotypes.tsv, microbiome.tsv, config.yaml, manifest.json, plus taxonomy.tsv and microbiome_genus.tsv / microbiome_family.tsv if taxonomy is supplied. Configuration uses output-relative paths and omits identifier fields. Candidate tables are re-read and validated before successful completion; failure cleans newly created output. Existing output directories are never intentionally overwritten.

Python API: load_config, load_dataset, IndependentEngine.fit, sample_phenotypes, sample_microbiome, and generation.generate. Fit only validated data. Random seeds reproduce output with the same library versions and input ordering. No fitted model is saved.

For stratified fits, see [stratification](STRATIFICATION.md). Limitations: no within-stratum joint correlations, correlated missingness, formal privacy guarantees, exact-match checks, or release approval. Generated categories can be combined in new ways. Constant and rare values can be reproduced. Full data and generated tables are currently held in memory. Treat all generated files as sensitive until reviewed. Do not publish source-derived outputs from this milestone.
