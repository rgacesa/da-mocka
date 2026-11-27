from pathlib import Path

import pytest
import yaml

from da_mocka.config import ContractError, load_config


def test_relative_paths(cohort):
    config = load_config(cohort / "config.yaml")
    assert config.phenotypes == cohort / "phenotypes.tsv"
    assert config.columns["age"].minimum == 18
    assert config.columns["marker"].nullable
    assert config.columns["sex"].categories == ("female", "male")


@pytest.mark.parametrize(
    "change",
    [
        {"extra": True},
        {"sample_id": ""},
        {"columns": {}},
        {"columns": {"sample_id": {"type": "identifier"}}},
        {"columns": {" ": {"type": "identifier"}}},
        {"phenotypes": 12},
        {"columns": {"age": {"type": "unknown"}}},
        {"columns": {"age": {"type": "integer", "nullable": "yes"}}},
        {"columns": {"age": {"type": "integer", "min": 20, "max": 10}}},
        {"columns": {"age": {"type": "integer", "min": 1.5, "max": 10}}},
        {"columns": {"age": {"type": "continuous", "min": True, "max": 10}}},
        {"columns": {"age": {"type": "continuous", "min": 1}}},
        {"columns": {"sex": {"type": "categorical", "categories": []}}},
        {"columns": {"sex": {"type": "categorical", "categories": [1]}}},
        {"columns": {"sex": {"type": "categorical", "categories": ["a", "a"]}}},
        {"columns": {"id": {"type": "identifier", "min": 1}}},
    ],
)
def test_invalid_config(cohort, change):
    path = cohort / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw.update(change)
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContractError):
        load_config(path)


@pytest.mark.parametrize("text", ["sample_id: x\nsample_id: y\n", "columns: [", "1: a"])
def test_invalid_yaml(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ContractError):
        load_config(path)


def test_missing_config(tmp_path):
    with pytest.raises(ContractError):
        load_config(tmp_path / "missing.yaml")


def test_identifier_and_no_taxonomy(cohort: Path):
    path = cohort / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw.pop("taxonomy")
    raw["columns"]["local_id"] = {"type": "identifier"}
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert config.taxonomy is None
    assert config.columns["local_id"].kind == "identifier"
