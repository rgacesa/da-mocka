import numpy as np
import pandas as pd
import pytest

from da_mocka.config import ContractError, load_config
from da_mocka.io import load_dataset, read_table


def test_alignment_and_composition(cohort):
    path = cohort / "microbiome.tsv"
    frame = read_table(path).iloc[::-1]
    frame.to_csv(path, sep="\t", index=False)
    result = load_dataset(load_config(cohort / "config.yaml"))
    assert result.phenotypes.index.equals(result.microbiome.index)
    assert result.microbiome.shape == (40, 200)
    assert np.allclose(result.microbiome.sum(axis=1), 1)
    assert (result.microbiome >= 0).all().all()
    assert result.taxonomy.shape == (200, 2)
    assert str(result.phenotypes["age"].dtype) == "Int64"


@pytest.mark.parametrize(
    ("filename", "column", "value", "error"),
    [
        ("phenotypes", "sample_id", "", "IDs"),
        ("phenotypes", "sample_id", "HAMSTER_00000002", "IDs"),
        ("phenotypes", "age", "x", "numeric"),
        ("phenotypes", "age", "inf", "Nonfinite"),
        ("phenotypes", "age", "12", "bounds"),
        ("phenotypes", "age", "34.5", "Nonintegral"),
        ("phenotypes", "age", "", "missing"),
        ("phenotypes", "sex", "invented", "categorical"),
        ("microbiome", "sample_id", "unknown", "sets differ"),
        ("microbiome", "mock_species_000", "-1", "nonnegative"),
        ("microbiome", "mock_species_000", "", "finite"),
        ("microbiome", "mock_species_000", "0.9", "sum to one"),
        ("taxonomy", "taxon", "unknown", "cover"),
        ("taxonomy", "taxon", "mock_species_001", "Duplicate"),
        ("taxonomy", "family", "", "Incomplete"),
        ("taxonomy", "family", "other", "inconsistent"),
    ],
)
def test_invalid_cells(cohort, filename, column, value, error):
    path = cohort / f"{filename}.tsv"
    frame = read_table(path)
    frame.loc[0, column] = value
    frame.to_csv(path, sep="\t", index=False)
    with pytest.raises(ContractError, match=error):
        load_dataset(load_config(cohort / "config.yaml"))


@pytest.mark.parametrize(
    ("name", "contents"),
    [
        ("a.txt", "a\n1\n"),
        ("a.tsv", ""),
        ("a.tsv", "a\ta\n1\t2\n"),
        ("a.tsv", "a\tb\n1\n"),
        ("a.tsv", "a\n"),
        ("a.tsv", 'a\n"unterminated\n'),
    ],
)
def test_invalid_tables(tmp_path, name, contents):
    path = tmp_path / name
    path.write_text(contents)
    with pytest.raises(ContractError):
        read_table(path)


def test_missing_table(tmp_path):
    with pytest.raises(ContractError):
        read_table(tmp_path / "missing.tsv")


def test_literal_strings(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("sample_id,label\n001,NA\n002,\n")
    frame = read_table(path)
    assert frame.loc[0, "sample_id"] == "001"
    assert frame.loc[0, "label"] == "NA"
    assert pd.isna(frame.loc[1, "label"])


@pytest.mark.parametrize(
    ("filename", "column"),
    [("phenotypes", "age"), ("phenotypes", "sample_id"), ("taxonomy", "family")],
)
def test_missing_columns(cohort, filename, column):
    path = cohort / f"{filename}.tsv"
    read_table(path).drop(columns=column).to_csv(path, sep="\t", index=False)
    with pytest.raises(ContractError):
        load_dataset(load_config(cohort / "config.yaml"))


def test_zero_features(cohort):
    path = cohort / "microbiome.tsv"
    read_table(path)[["sample_id"]].to_csv(path, sep="\t", index=False)
    with pytest.raises(ContractError, match="No microbiome features"):
        load_dataset(load_config(cohort / "config.yaml"))
