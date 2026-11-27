import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from da_mocka.cli import main
from da_mocka.config import ContractError, load_config
from da_mocka.engines.independent import IndependentEngine, Marginal
from da_mocka.generation import generate
from da_mocka.io import load_dataset, read_table
from da_mocka.taxonomy import aggregate_taxonomy


def test_generate(cohort, tmp_path):
    source = load_config(cohort / "config.yaml")
    original = load_dataset(source)
    output = tmp_path / "mock"
    generate(source, output, 300, seed=19)
    result = load_dataset(load_config(output / "config.yaml"))
    p = result.phenotypes
    x = result.microbiome
    assert x.shape == (300, 200)
    assert p.index.equals(x.index)
    assert p.index.is_unique
    assert not set(p.index) & set(original.phenotypes.index)
    assert all(value.startswith("MOCK_") for value in p.index)
    assert np.allclose(x.sum(axis=1), 1)
    assert (x >= 0).all().all()
    assert np.allclose(p["BMI"], p["weight"] / p["height"] ** 2)
    assert set(p["sex"]) <= set(original.phenotypes["sex"])
    assert p["age"].between(18, 85).all()
    assert p["marker"].isna().any()
    assert 0.5 < (x == 0).to_numpy().mean() < 0.9
    for rank in ("genus", "family"):
        expected = aggregate_taxonomy(x, result.taxonomy, rank)
        actual = pd.read_csv(output / f"microbiome_{rank}.tsv", sep="\t", index_col=0)
        assert np.allclose(actual, expected)
        assert np.allclose(actual.sum(axis=1), 1)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["release_status"] == "unreviewed_not_anonymous"
    assert manifest["n_samples"] == 300
    assert not any(str(cohort) in f.read_text() for f in output.iterdir())


def test_seed_and_no_overwrite(cohort, tmp_path):
    source = load_config(cohort / "config.yaml")
    a, b, c = (tmp_path / name for name in ("a", "b", "c"))
    generate(source, a, 20, seed=1)
    generate(source, b, 20, seed=1)
    generate(source, c, 20, seed=2)
    assert all(f.read_bytes() == (b / f.name).read_bytes() for f in a.iterdir())
    assert (a / "microbiome.tsv").read_bytes() != (c / "microbiome.tsv").read_bytes()
    with pytest.raises(ContractError, match="already exists"):
        generate(source, a, 20)


@pytest.mark.parametrize(("n", "seed"), [(0, 0), (1, -1)])
def test_bad_arguments(cohort, tmp_path, n, seed):
    output = tmp_path / "bad"
    with pytest.raises(ContractError):
        generate(load_config(cohort / "config.yaml"), output, n, seed)
    assert not output.exists()


def test_identifier_and_all_missing(cohort, tmp_path):
    path = cohort / "phenotypes.tsv"
    p = read_table(path)
    p["patient"] = "PRIVATE_ID"
    p["marker"] = ""
    p.to_csv(path, sep="\t", index=False)
    path = cohort / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["columns"]["patient"] = {"type": "identifier"}
    raw.pop("taxonomy")
    raw["columns"]["sex"]["categories"].append("unobserved")
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "mock"
    generate(load_config(path), output, 100)
    result = load_dataset(load_config(output / "config.yaml"))
    assert "patient" not in result.phenotypes
    assert result.phenotypes["marker"].isna().all()
    assert "unobserved" not in result.phenotypes["sex"].values
    assert not (output / "taxonomy.tsv").exists()
    assert all("PRIVATE_ID" not in f.read_text() for f in output.iterdir())


def test_constant_and_zero_taxa(cohort, tmp_path):
    path = cohort / "microbiome.tsv"
    x = read_table(path)
    x.iloc[:, 1:] = "0"
    x.iloc[:, 1] = "1"
    x.to_csv(path, sep="\t", index=False)
    source = load_config(cohort / "config.yaml")
    output = tmp_path / "mock"
    generate(source, output, 12)
    result = load_dataset(load_config(output / "config.yaml"))
    assert (result.microbiome.iloc[:, 0] == 1).all()
    assert (result.microbiome.iloc[:, 1:] == 0).all().all()


def test_reject_inconsistent_bmi(cohort, tmp_path):
    path = cohort / "phenotypes.tsv"
    p = read_table(path)
    p.loc[0, "BMI"] = "42"
    p.to_csv(path, sep="\t", index=False)
    output = tmp_path / "bad"
    with pytest.raises(ContractError, match="Inconsistent BMI"):
        generate(load_config(cohort / "config.yaml"), output, 10)
    assert not output.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"target": "unknown"},
        {"target": "age"},
        {"weight": "height"},
        {"atol": -1},
    ],
)
def test_invalid_bmi_rule(cohort, change):
    path = cohort / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["bmi"].update(change)
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContractError):
        load_config(path)


def test_failed_community(cohort, monkeypatch):
    config = load_config(cohort / "config.yaml")
    engine = IndependentEngine.fit(load_dataset(config), config)
    monkeypatch.setattr(engine, "prevalence", np.zeros_like(engine.prevalence))
    with pytest.raises(ContractError, match="nonempty"):
        engine.sample_microbiome(2, np.random.default_rng(42))


def test_failed_bmi_bounds(cohort):
    config = load_config(cohort / "config.yaml")
    engine = IndependentEngine.fit(load_dataset(config), config)
    engine.marginals["weight"] = Marginal(config.columns["weight"], np.array([185.0]), 0)
    engine.marginals["height"] = Marginal(config.columns["height"], np.array([1.4]), 0)
    with pytest.raises(ContractError, match="BMI bounds"):
        engine.sample_phenotypes(2, np.random.default_rng(42))


def test_no_bmi_rule(cohort, tmp_path):
    config = replace(load_config(cohort / "config.yaml"), bmi=None)
    output = tmp_path / "mock"
    generate(config, output, 10)
    assert load_config(output / "config.yaml").bmi is None


def test_aggregation_errors(cohort):
    data = load_dataset(load_config(cohort / "config.yaml"))
    with pytest.raises(ContractError, match="Unsupported"):
        aggregate_taxonomy(data.microbiome, data.taxonomy, "order")
    with pytest.raises(ContractError, match="cover"):
        aggregate_taxonomy(data.microbiome, data.taxonomy.iloc[1:], "genus")


def test_generate_cli(cohort, tmp_path, capsys):
    args = [
        "generate",
        "--config",
        str(cohort / "config.yaml"),
        "--output",
        str(tmp_path / "out"),
        "--n-samples",
        "20",
    ]
    assert main(args) == 0
    assert "Not reviewed" in capsys.readouterr().out
    assert main(args) == 2
