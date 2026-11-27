"""Generate local, unreviewed synthetic candidates using the independent engine."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from da_mocka import __version__
from da_mocka.config import Config, ContractError, load_config
from da_mocka.engines.independent import IndependentEngine
from da_mocka.io import Dataset, load_dataset
from da_mocka.stratification import plan_strata, sample_strata
from da_mocka.taxonomy import aggregate_taxonomy


def _fresh_ids(data: Dataset, config: Config, n: int, rng: np.random.Generator) -> list[str]:
    seen = set(data.phenotypes.index)
    for name, spec in config.columns.items():
        if spec.kind == "identifier":
            seen.update(data.phenotypes[name].dropna())
    ids = []
    while len(ids) < n:
        identifier = "MOCK_" + rng.bytes(16).hex()
        if identifier not in seen:
            seen.add(identifier)
            ids.append(identifier)
    return ids


def _output_config(config: Config) -> dict:
    columns = {}
    for name, spec in config.columns.items():
        if spec.kind == "identifier":
            continue
        column = {"type": spec.kind, "nullable": spec.nullable}
        if spec.kind in ("integer", "continuous"):
            column.update(min=spec.minimum, max=spec.maximum)
        elif spec.kind == "categorical":
            column["categories"] = list(spec.categories)
        columns[name] = column
    result = {
        "sample_id": config.sample_id,
        "phenotypes": "phenotypes.tsv",
        "microbiome": "microbiome.tsv",
        "columns": columns,
    }
    if config.taxonomy is not None:
        result["taxonomy"] = "taxonomy.tsv"
    if config.bmi is not None:
        result["bmi"] = asdict(config.bmi)
    return result


def generate(config: Config, output: Path | str, n_samples: int | None, seed: int = 42) -> None:
    """Fit locally, generate, validate a round trip, and write candidate files."""
    if (n_samples is not None and n_samples < 1) or seed < 0:
        raise ContractError("Positive sample count and nonnegative seed required")
    output = Path(output)
    if output.exists():
        raise ContractError("Generation output directory already exists")
    data = load_dataset(config)
    strata = plan_strata(data, config, n_samples)
    n_samples = sum(group.count for group in strata)
    rng = np.random.default_rng(seed)
    if config.stratification.columns:
        phenotypes, microbiome = sample_strata(data, config, strata, rng)
    else:
        engine = IndependentEngine.fit(data, config)
        phenotypes = engine.sample_phenotypes(n_samples, rng)
        microbiome = engine.sample_microbiome(n_samples, rng)
    ids = _fresh_ids(data, config, n_samples, rng)
    for frame in (phenotypes, microbiome):
        frame.index = ids
        frame.index.name = config.sample_id
    aggregates = {}
    if data.taxonomy is not None:
        for rank in ("genus", "family"):
            aggregates[rank] = aggregate_taxonomy(microbiome, data.taxonomy, rank)

    output.mkdir(parents=True, exist_ok=False)
    try:
        phenotypes.to_csv(output / "phenotypes.tsv", sep="\t")
        microbiome.to_csv(output / "microbiome.tsv", sep="\t")
        if data.taxonomy is not None:
            data.taxonomy.to_csv(output / "taxonomy.tsv", sep="\t", index_label="taxon")
            for rank, frame in aggregates.items():
                frame.to_csv(output / f"microbiome_{rank}.tsv", sep="\t")
        path = output / "config.yaml"
        path.write_text(yaml.safe_dump(_output_config(config), sort_keys=False), encoding="utf-8")
        validated = load_dataset(load_config(path))
        if len(validated.phenotypes) != n_samples:
            raise ContractError("Generated sample count mismatch")
        manifest = {
            "version": __version__,
            "engine": "independent",
            "stratification": list(config.stratification.columns),
            "allocation": config.stratification.allocation,
            "n_samples": n_samples,
            "seed": seed,
            "release_status": "unreviewed_not_anonymous",
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output)
        raise
