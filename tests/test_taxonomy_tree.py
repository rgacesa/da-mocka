from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from da_mocka.config import ContractError, EngineSettings, Stratification, load_config
from da_mocka.generation import generate
from da_mocka.io import load_dataset, read_table, validate_rank_exports
from da_mocka.taxonomy import aggregate_taxonomy, taxonomy_ranks
from da_mocka.tree import RESIDUAL, canonicalize
from da_mocka.validation.audit import load_candidate


@pytest.fixture
def tree():
    rows = [
        ("D", "", "domain"),
        ("K", "D", "kingdom"),
        ("P", "K", "phylum"),
        ("C", "P", "class"),
        ("O", "C", "order"),
        ("F", "O", "family"),
        ("G1", "F", "genus"),
        ("G2", "F", "genus"),
        ("S1", "G1", "species"),
        ("S2", "G1", "species"),
        ("S3", "G2", "species"),
        ("T1", "S1", "strain"),
        ("T2", "S1", "strain"),
    ]
    frame = pd.DataFrame(rows, columns=["taxon", "parent", "rank"])
    return frame.set_index("taxon").astype("string")


def test_all_ranks(tree):
    x = pd.DataFrame({"T1": [0.2], "T2": [0.1], "S2": [0.3], "S3": [0.4]})
    x, tax = canonicalize(x, tree, "terminal")
    assert len(taxonomy_ranks(tax)) == 9
    for rank in taxonomy_ranks(tax):
        result = aggregate_taxonomy(x, tax, rank)
        assert np.allclose(result.sum(axis=1), 1)
    species = aggregate_taxonomy(x, tax, "species")
    assert species.loc[0, "S1"] == pytest.approx(0.3)
    genus = aggregate_taxonomy(x, tax, "genus")
    assert genus.loc[0, "G1"] == pytest.approx(0.6)
    assert genus.loc[0, "G2"] == pytest.approx(0.4)
    strain = aggregate_taxonomy(x, tax, "strain")
    assert strain.loc[0, RESIDUAL + "strain:S2"] == pytest.approx(0.3)


def test_inclusive_and_unclassified(tree):
    source = pd.DataFrame({"G1": [0.6], "S1": [0.25], "S2": [0.2], "G2": [0.4]})
    x, tax = canonicalize(source, tree, "hierarchical")
    assert x.loc[0, RESIDUAL + "G1"] == pytest.approx(0.15)
    assert np.allclose(x.sum(axis=1), 1)
    genus = aggregate_taxonomy(x, tax, "genus")
    assert genus.loc[0, "G1"] == pytest.approx(0.6)
    species = aggregate_taxonomy(x, tax, "species")
    assert species.loc[0, RESIDUAL + "species:G1"] == pytest.approx(0.15)
    assert species.loc[0, RESIDUAL + "species:G2"] == pytest.approx(0.4)
    again, canonical = canonicalize(x, tax, "terminal")
    pd.testing.assert_frame_equal(x, again)
    pd.testing.assert_frame_equal(tax, canonical)
    with pytest.raises(ContractError, match="overlapping"):
        canonicalize(source, tree, "terminal")
    with pytest.raises(ContractError, match="overlapping"):
        aggregate_taxonomy(source, tree, "genus")


def test_nearest_measured_ancestor(tree):
    source = pd.DataFrame({"F": [1.0], "G1": [0.6], "S1": [0.4], "S3": [0.2]})
    x, tax = canonicalize(source, tree, "hierarchical")
    assert x.loc[0, RESIDUAL + "F"] == pytest.approx(0.2)
    assert x.loc[0, RESIDUAL + "G1"] == pytest.approx(0.2)
    assert aggregate_taxonomy(x, tax, "family").loc[0, "F"] == pytest.approx(1)


def test_genus_only_and_skipped_rank(tree):
    x = pd.DataFrame({"G1": [0.6], "G2": [0.4]})
    x, tax = canonicalize(x, tree, "terminal")
    assert "species" not in taxonomy_ranks(tax)
    assert "strain" not in taxonomy_ranks(tax)
    tree.loc["S1", "parent"] = "F"
    x, tax = canonicalize(pd.DataFrame({"S1": [0.4], "G2": [0.6]}), tree, "terminal")
    genus = aggregate_taxonomy(x, tax, "genus")
    assert genus.loc[0, RESIDUAL + "genus:F"] == pytest.approx(0.4)
    assert np.allclose(genus.sum(axis=1), 1)


@pytest.mark.parametrize(
    ("node", "column", "value", "message"),
    [
        ("D", "parent", "G1", "Cyclic"),
        ("G1", "parent", "unknown", "Unknown taxonomic parent"),
        ("G1", "rank", "nonsense", "Unknown taxonomic rank"),
        ("G1", "rank", "family", "rank order"),
        ("G1", "rank", "unresolved", "Invalid unresolved"),
    ],
)
def test_invalid_tree(tree, node, column, value, message):
    tree.loc[node, column] = value
    with pytest.raises(ContractError, match=message):
        canonicalize(pd.DataFrame({"S1": [1.0]}), tree, "terminal")


def test_invalid_abundances(tree):
    with pytest.raises(ContractError, match="exceed"):
        canonicalize(pd.DataFrame({"G1": [0.4], "S1": [0.5], "G2": [0.6]}), tree, "hierarchical")
    with pytest.raises(ContractError, match="sum to one"):
        canonicalize(pd.DataFrame({"S1": [0.4]}), tree, "terminal")
    with pytest.raises(ContractError, match="cover"):
        canonicalize(pd.DataFrame({"unknown": [1.0]}), tree, "terminal")


@pytest.fixture
def hierarchical(cohort, tree):
    path = cohort / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["taxonomy_mode"] = "hierarchical"
    path.write_text(yaml.safe_dump(raw))
    ids = read_table(cohort / "phenotypes.tsv")["sample_id"]
    rng = np.random.default_rng(42)
    g = rng.uniform(0.35, 0.75, len(ids))
    x = pd.DataFrame(
        {
            "sample_id": ids,
            "G1": g,
            "G2": 1 - g,
            "S1": g * 0.5,
            "S2": g * 0.25,
        }
    )
    x.to_csv(cohort / "microbiome.tsv", sep="\t", index=False)
    tree.to_csv(cohort / "taxonomy.tsv", sep="\t")
    return load_config(path)


@pytest.mark.parametrize("engine", ["independent", "joint_latent"])
@pytest.mark.parametrize("stratified", [False, True])
def test_generation_all_ranks(hierarchical, tmp_path, engine, stratified):
    config = replace(hierarchical, engine=EngineSettings(name=engine, rank=4))
    if stratified:
        config = replace(
            config,
            stratification=Stratification(columns=("disease", "sex"), min_stratum_size=1),
        )
    output = tmp_path / "mock"
    generate(config, output, 60, seed=7)
    result = load_candidate(config, output)
    assert result.microbiome.shape == (60, 4)
    assert result.phenotypes.index.equals(result.microbiome.index)
    assert np.allclose(result.microbiome.sum(axis=1), 1)
    assert RESIDUAL + "G1" in result.microbiome
    assert "G1" not in result.microbiome
    assert "S3" not in result.microbiome
    validate_rank_exports(result, output, config.sample_id)
    genus = read_table(output / "microbiome_genus.tsv")
    species = read_table(output / "microbiome_species.tsv")
    assert np.allclose(
        genus["G1"].astype(float),
        species["S1"].astype(float)
        + species["S2"].astype(float)
        + species[RESIDUAL + "species:G1"].astype(float),
    )
    # Keep total exactly unchanged but corrupt taxonomic relationships.
    genus["G1"] = "0.01"
    genus["G2"] = "0.99"
    genus.to_csv(output / "microbiome_genus.tsv", sep="\t", index=False)
    with pytest.raises(ContractError, match="disagree"):
        load_candidate(config, output)


def test_missing_rank_export(hierarchical, tmp_path):
    output = tmp_path / "mock"
    generate(hierarchical, output, 10)
    (output / "microbiome_genus.tsv").unlink()
    with pytest.raises(ContractError, match="Missing"):
        load_candidate(hierarchical, output)


def test_multilevel(hierarchical):
    config = hierarchical
    source = read_table(config.microbiome)
    genus = source[["sample_id", "G1", "G2"]]
    rank_path = config.microbiome.parent / "genus.tsv"
    genus.iloc[::-1].to_csv(rank_path, sep="\t", index=False)
    expected = load_dataset(config)
    source[["sample_id", "S1", "S2"]].to_csv(config.microbiome, sep="\t", index=False)
    config = replace(config, taxonomy_mode="multilevel", rank_tables={"genus": rank_path})
    actual = load_dataset(config)
    pd.testing.assert_frame_equal(
        expected.microbiome.sort_index(axis=1), actual.microbiome.sort_index(axis=1)
    )
    genus = genus.copy()
    genus["G1"] = "0.01"
    genus["G2"] = "0.99"
    genus.to_csv(rank_path, sep="\t", index=False)
    with pytest.raises(ContractError, match="exceed"):
        load_dataset(config)


def test_multilevel_duplicate_conflict(hierarchical):
    config = hierarchical
    source = read_table(config.microbiome)
    path = config.microbiome.parent / "genus.tsv"
    genus = source[["sample_id", "G1", "G2"]].copy()
    genus["G1"] = "0.01"
    genus.to_csv(path, sep="\t", index=False)
    config = replace(config, taxonomy_mode="multilevel", rank_tables={"genus": path})
    with pytest.raises(ContractError, match="Conflicting"):
        load_dataset(config)


def test_duplicate_ids_and_wrong_rank(hierarchical):
    config = hierarchical
    source = read_table(config.taxonomy)
    pd.concat([source, source.iloc[:1]]).to_csv(config.taxonomy, sep="\t", index=False)
    with pytest.raises(ContractError, match="unique"):
        load_dataset(config)
    source.to_csv(config.taxonomy, sep="\t", index=False)
    config = replace(config, taxonomy_mode="multilevel", rank_tables={"genus": config.microbiome})
    with pytest.raises(ContractError, match="declared rank"):
        load_dataset(config)


@pytest.mark.parametrize(
    "change",
    [
        {"taxonomy_mode": "bad"},
        {"rank_tables": []},
        {"rank_tables": {"genus": "genus.tsv"}},
        {"taxonomy_mode": "multilevel"},
        {"taxonomy_mode": "multilevel", "rank_tables": {"genus": 1}},
    ],
)
def test_invalid_config(hierarchical, change):
    path = hierarchical.microbiome.parent / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw.update(change)
    with pytest.raises(ContractError):
        path.write_text(yaml.safe_dump(raw))
        load_config(path)
