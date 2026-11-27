"""Fully invented cosmic hamster reference cohort."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from da_mocka.config import ContractError


def make_demo(output: Path | str, n_samples: int = 2000, seed: int = 42) -> None:
    """Generate an artificial cohort; never read source data or overwrite files."""
    if n_samples < 1 or seed < 0:
        raise ContractError("Positive sample count and nonnegative seed required")
    output = Path(output)
    if output.exists():
        raise ContractError("Demo output directory already exists")
    rng = np.random.default_rng(seed)
    n = n_samples
    ids = [f"HAMSTER_{i + 1:08d}" for i in range(n)]
    disease = rng.choice(["control", "A", "B"], n, p=[0.6, 0.3, 0.1])
    sex = rng.choice(["female", "male"], n)
    age = np.clip(np.rint(rng.normal(48 + 8 * (disease != "control"), 12)), 18, 85)
    height = np.clip(rng.normal(1.70 + 0.10 * (sex == "male"), 0.07), 1.4, 2.1)
    bmi = np.clip(rng.normal(24 + 3 * (disease == "A"), 3), 18, 42)
    weight = bmi * height**2
    marker = rng.normal(3 + 2 * (disease == "B") + 0.03 * age, 0.8)
    marker = np.clip(marker, -10, 20)
    marker[rng.random(n) < 0.1] = np.nan
    phenotypes = pd.DataFrame(
        {
            "sample_id": ids,
            "disease": disease,
            "sex": sex,
            "age": age.astype(int),
            "height": height,
            "weight": weight,
            "BMI": bmi,
            "marker": marker,
        }
    )
    p = 200
    latent = rng.normal(size=(n, 10))
    community = latent[:, np.arange(p) % 10]
    log_abundance = rng.normal(-3, 1, (n, p)) + 0.6 * community
    log_abundance[:, :10] += 1.5 * (disease == "A")[:, None]
    log_abundance[:, 10:20] += 1.2 * (disease == "B")[:, None]
    probability = 1 / (1 + np.exp(0.9 - 0.7 * community))
    present = rng.random((n, p)) < probability
    present[~present.any(axis=1), 0] = True
    abundance = np.exp(log_abundance) * present
    abundance /= abundance.sum(axis=1, keepdims=True)
    taxa = [f"mock_species_{i:03d}" for i in range(p)]
    microbiome = pd.DataFrame(abundance, columns=taxa)
    microbiome.insert(0, "sample_id", ids)
    taxonomy = pd.DataFrame(
        {
            "taxon": taxa,
            "genus": [f"mock_genus_{i // 4:02d}" for i in range(p)],
            "family": [f"mock_family_{(i // 4) % 15:02d}" for i in range(p)],
        }
    )
    config = {
        "sample_id": "sample_id",
        "phenotypes": "phenotypes.tsv",
        "microbiome": "microbiome.tsv",
        "taxonomy": "taxonomy.tsv",
        "bmi": {"target": "BMI", "weight": "weight", "height": "height"},
        "columns": {
            "disease": {"type": "categorical", "categories": ["control", "A", "B"]},
            "sex": {"type": "categorical", "categories": ["female", "male"]},
            "age": {"type": "integer", "min": 18, "max": 85},
            "height": {"type": "continuous", "min": 1.4, "max": 2.1},
            "weight": {"type": "continuous", "min": 35, "max": 186},
            "BMI": {"type": "continuous", "min": 18, "max": 42},
            "marker": {"type": "continuous", "min": -10, "max": 20, "nullable": True},
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (
        ("phenotypes", phenotypes),
        ("microbiome", microbiome),
        ("taxonomy", taxonomy),
    ):
        frame.to_csv(output / f"{name}.tsv", sep="\t", index=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
