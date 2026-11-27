# da mocka

Constrained, stratified, privacy-aware synthetic biological data generation.

**Fake individuals. Real patterns. No poo.**

Status: **0.0.7** — local fidelity and disclosure diagnostics, experimental joint latent engine, categorical stratification and biological constraints. Outputs are local unreviewed synthetic candidates, not anonymous or approved for release.

## Quick start (Linux / Snellius)

Use an available Python 3.11–3.13 interpreter.

```bash
git clone https://github.com/rgacesa/da-mocka.git
cd da-mocka
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
da-mocka --version
da-mocka demo --output work/hamsters --n-samples 2000 --seed 42
da-mocka check-input --config work/hamsters/config.yaml
da-mocka generate --config work/hamsters/config.yaml --output work/mock --n-samples 5000 --seed 42
da-mocka check-input --config work/mock/config.yaml
```

The demo is generated from invented population parameters, not patient data. Output paths must not already exist. Input checks are structural, not a release/privacy review.

For your own data, start from configs/example.yaml and edit paths, phenotype types, bounds, and allowed categories explicitly. Paths are relative to the YAML file. Tables are TSV or CSV; sample IDs must be unique and match across tables. Microbiome features must be mutually exclusive terminal taxa, finite and nonnegative, summing to one per row. Optional taxonomy supports the legacy genus/family mapping or an explicit multi-rank tree, including overlapping inclusive inputs and separate rank tables. See [taxonomy](docs/TAXONOMY.md).

The baseline supports categorical stratification but intentionally does not model within-stratum correlations. It samples observed categories, empirical numerical marginals and taxon presence/positive abundance independently, then normalizes communities. Derived BMI is opt-in. Identifier columns are dropped and sample IDs regenerated. See [baseline engine](docs/BASELINE.md) for limitations.

Stratify by disease or disease and sex, select outcomes, and balance or customize cohort sizes. See [stratification](docs/STRATIFICATION.md) for CLI and YAML examples.

Select `--engine joint_latent` to model approximate within-stratum dependencies, including microbial presence, abundance, phenotypes and missingness. The independent engine remains the default. See [joint engine](docs/JOINT_LATENT.md).

Run a full local audit with `--audit work/local_audit` during generation, or `da-mocka validate --config work/hamsters/config.yaml --synthetic work/mock --report work/recheck`. See [validation and disclosure diagnostics](docs/VALIDATION.md). Reports are sensitive and must stay local.

Native MetaPhlAn merged profiles and individual profile manifests are supported; see [MetaPhlAn](docs/METAPHLAN.md). No MetaPhlAn installation is required.

See [roadmap](docs/ROADMAP.md), [data contract](docs/DATA_CONTRACT.md), and [privacy policy](docs/PRIVACY.md).

Never commit real cohort data, fitted models, or local audit reports to this public repository.
