# Data contract (0.0.5)

YAML has exactly sample_id, phenotypes, microbiome, optional taxonomy, optional bmi, optional stratification, optional engine, optional validation, and columns. Paths are relative to the configuration file. See configs/example.yaml. Columns must declare type continuous, integer, categorical, or identifier; numeric bounds and categorical allowed values are mandatory; nullable defaults to false. Identifier fields are recognized, not anonymized by check-input. Unknown columns fail. Empty fields represent missing values; other strings such as NA are literal. Numeric values must be finite. Optional bmi mapping declares target, weight, height (kg and metres) and atol (default 0.1 for source rounding). All three columns must be distinct, positive, nonnullable continuous fields. Source BMI must match the formula within tolerance; generation derives BMI exactly and rejects draws outside its declared bounds. Other cross-field rules are future work.

TSV/CSV headers must be nonempty and unique. Samples are rows; sample IDs are nonempty strings (leading zeros retained), unique, and identical as sets across modalities. Microbiome rows are aligned to phenotype order. Abundances are fractions (not percentages), finite, nonnegative, with absolute sum tolerance 1e-6. No silent renormalization. All features must be terminal, nonoverlapping taxa; optional taxonomy TSV/CSV has exactly taxon, genus, family columns with a row for every feature, and each genus maps to exactly one family. Mixed rank or unresolved lineages must be resolved by the caller in this milestone; mapping is not a taxonomic database or proof of biological validity.

Demo data are entirely invented, with adult ages, BMI derived from weight and height, disease and sex categories, missing optional biomarker, and sparse microbial compositions. Only artificial demo data may be committed. Real inputs remain local. The check-input command reports counts only on success and generic error messages on failure; these counts can still be sensitive.

For categorical grouping and generation options, see [stratification](STRATIFICATION.md). Input validation does not certify the eligibility or privacy of a stratum; generation checks selected groups before fitting.

Engine configuration is described in [joint latent engine](JOINT_LATENT.md). The output configuration intentionally omits generation options.

See [validation](VALIDATION.md) for audit settings and source-authoritative candidate checks. Reports are written outside candidate directories.
