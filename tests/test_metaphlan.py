import gzip
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from da_mocka.cli import main
from da_mocka.config import ContractError, EngineSettings, Stratification, load_config
from da_mocka.generation import generate
from da_mocka.io import load_dataset, read_table
from da_mocka.metaphlan import load_metaphlan, merged_view
from da_mocka.taxonomy import aggregate_taxonomy, taxonomy_ranks
from da_mocka.validation.audit import load_candidate

K = "k__Bacteria"
G = K + "|p__Firmicutes|c__Bacilli|o__Lactobacillales|f__Lactobacillaceae|g__Mock"
S = G + "|s__Mock_alpha"
T = S + "|t__SGB123"
U = G + "|s__Mock_SGB456|t__SGB456"


def write_merged(path, rows, header="clade_name\t001\t002"):
    path.write_text("#mpa_vJan21_CHOCOPhlAnSGB_202103\n" + header + "\n" + rows)


def test_native_profiles(tmp_path):
    path = tmp_path / "merged.txt"
    write_merged(
        path,
        f"UNCLASSIFIED\t20\t100\n{K}\t80\t0\n{G}\t80\t0\n{S}\t50\t0\n{T}\t50\t0\n{U}\t30\t0\n",
    )
    x, tree = load_metaphlan(path)
    assert list(x.index) == ["001", "002"]
    assert np.allclose(x.sum(axis=1), 1)
    assert x.loc["001", "UNCLASSIFIED"] == 0.2
    assert x.loc["002", "UNCLASSIFIED"] == 1
    assert "sgb" in taxonomy_ranks(tree)
    assert "strain" not in taxonomy_ranks(tree)
    assert tree.loc[T, "rank"] == "sgb"
    for rank in taxonomy_ranks(tree):
        view = aggregate_taxonomy(x, tree, rank)
        assert np.allclose(view.sum(axis=1), 1)
        assert view.loc["001", "UNCLASSIFIED"] == 0.2
    assert aggregate_taxonomy(x, tree, "genus").loc["001", G] == pytest.approx(0.8)
    assert merged_view(x, tree).loc["001", S] == pytest.approx(0.5)


def test_filtered_gzip_with_metadata(tmp_path):
    path = tmp_path / "filtered.tsv.gz"
    text = f"#clade_name\tNCBI_tax_id\t001\t002\n{T}\t2|123\t70\t40\n{U}\t2|456\t30\t60\n"
    with gzip.open(path, "wt") as handle:
        handle.write(text)
    x, tree = load_metaphlan(path)
    assert list(x.index) == ["001", "002"]
    assert set(x.columns) == {T, U}
    assert tree.loc[T, "parent"] == S
    assert aggregate_taxonomy(x, tree, "species").shape == (2, 2)


def test_legacy_and_rounding(tmp_path):
    path = tmp_path / "legacy.txt"
    write_merged(
        path,
        f"UNKNOWN\t20\t0\n{K}\t80\t100\n{S}\t80.0002\t100\n",
    )
    x, tree = load_metaphlan(path)
    assert np.allclose(x.sum(axis=1), 1)
    assert np.allclose(merged_view(x, tree)[K], [0.8, 1], atol=1e-5)
    assert "UNCLASSIFIED" in x
    assert "UNKNOWN" not in x


def test_manifest(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    header = (
        "#mpa_vJan21_CHOCOPhlAnSGB_202103\n"
        "# /private/source/command\n"
        "#clade_name\tNCBI_tax_id\trelative_abundance\tadditional_species\n"
    )
    a.write_text(header + f"{T}\t2|123\t100\tother\n")
    b.write_text(header + f"{U}\t2|456\t100\tother\n")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("sample_id\tprofile\n001\ta.txt\n002\tb.txt\n")
    x, _ = load_metaphlan(manifest, profiles=True)
    assert list(x.index) == ["001", "002"]
    assert x.loc["001", U] == 0
    assert x.loc["002", T] == 0
    b.write_text(b.read_text().replace("202103", "202307"))
    with pytest.raises(ContractError, match="different MetaPhlAn database"):
        load_metaphlan(manifest, profiles=True)
    manifest.write_text("sample_id\tprofile\n001\ta.txt\n001\tb.txt\n")
    with pytest.raises(ContractError, match="Duplicate sample"):
        load_metaphlan(manifest, profiles=True)


@pytest.mark.parametrize(
    ("header", "rows", "message"),
    [
        ("other\t001\t002", f"{T}\t100\t100\n", "clade_name"),
        ("clade_name\t001\t001", f"{T}\t100\t100\n", "Duplicate"),
        ("clade_name\t001\t002", f"{T}\t100\n", "ragged"),
        ("clade_name\t001\t002", f"{T}\t100\t100\n{T}\t0\t0\n", "Duplicate"),
        ("clade_name\t001\t002", f"{T}\tNaN\t100\n", "finite percentages"),
        ("clade_name\t001\t002", f"{T}\t-1\t100\n", "finite percentages"),
        ("clade_name\t001\t002", f"{T}\t1000\t100\n", "finite percentages"),
        ("clade_name\t001\t002", f"{T}\tbad\t100\n", "Invalid MetaPhlAn"),
        ("clade_name\t001\t002", f"{T}\t1\t1\n", "sum to one"),
        ("clade_name\t001\t002", f"{K}\t50\t100\n{T}\t60\t100\n", "exceed"),
        ("clade_name\t001\t002", "g__\t100\t100\n", "Invalid MetaPhlAn lineage"),
        ("clade_name\t001\t002", "x__Unknown\t100\t100\n", "Unsupported"),
        ("clade_name\t001\t002", "s__A|g__B\t100\t100\n", "rank order"),
        ("clade_name\trelative_abundance", f"{T}\t100\n", "manifest"),
    ],
)
def test_invalid_merged(tmp_path, header, rows, message):
    path = tmp_path / "bad.txt"
    write_merged(path, rows, header)
    with pytest.raises(ContractError, match=message):
        load_metaphlan(path)


@pytest.fixture
def native(cohort):
    path = cohort / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw.pop("taxonomy")
    raw["microbiome"] = "merged.txt"
    raw["microbiome_format"] = "metaphlan"
    path.write_text(yaml.safe_dump(raw))
    ids = read_table(cohort / "phenotypes.tsv")["sample_id"].tolist()
    rng = np.random.default_rng(42)
    a = rng.uniform(10, 70, len(ids))
    frame = pd.DataFrame(
        [np.full(len(ids), 10), np.full(len(ids), 90), a, a, 90 - a],
        index=["UNCLASSIFIED", K, S, T, U],
        columns=ids,
    )
    frame.loc[:, ids[::-1]].to_csv(cohort / "merged.txt", sep="\t", index_label="clade_name")
    return load_config(path)


@pytest.mark.parametrize("engine", ["independent", "joint_latent"])
def test_generate_native(native, tmp_path, engine):
    config = replace(
        native,
        engine=EngineSettings(name=engine, rank=4),
        stratification=Stratification(columns=("disease", "sex"), min_stratum_size=1),
    )
    output = tmp_path / "mock"
    generate(config, output, 60, audit_dir=tmp_path / "audit")
    result = load_candidate(config, output)
    assert result.phenotypes.index.equals(result.microbiome.index)
    assert np.allclose(result.microbiome.sum(axis=1), 1)
    assert (output / "microbiome_sgb.tsv").is_file()
    assert not (output / "microbiome_strain.tsv").exists()
    assert load_config(output / "config.yaml").microbiome_format == "table"
    assert all(value.startswith("MOCK_") for value in result.microbiome.index)
    exported, tax = load_metaphlan(output / "metaphlan.tsv")
    expected = aggregate_taxonomy(result.microbiome, result.taxonomy, "sgb")
    actual = aggregate_taxonomy(exported, tax, "sgb")
    assert np.allclose(actual.loc[expected.index, expected.columns], expected)
    frame = pd.read_csv(output / "metaphlan.tsv", sep="\t", index_col=0)
    frame.loc[K] = 100
    frame.to_csv(output / "metaphlan.tsv", sep="\t")
    with pytest.raises(ContractError, match="disagrees"):
        load_candidate(config, output)


def test_alignment_and_cli(native):
    data = load_dataset(native)
    assert data.phenotypes.index.equals(data.microbiome.index)
    assert main(["check-input", "--config", str(native.phenotypes.parent / "config.yaml")]) == 0
    frame = pd.read_csv(native.microbiome, sep="\t", index_col=0)
    frame = frame.rename(columns={frame.columns[0]: "unknown_sample"})
    frame.to_csv(native.microbiome, sep="\t")
    with pytest.raises(ContractError, match="Sample ID sets"):
        load_dataset(native)


def test_config_conflict(native):
    path = native.phenotypes.parent / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["taxonomy"] = "taxonomy.tsv"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContractError, match="omit manual taxonomy"):
        load_config(path)
