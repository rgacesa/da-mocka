"""Read and validate aligned phenotype and compositional tables."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from da_mocka.config import Config, ContractError
from da_mocka.constraints import check_bmi
from da_mocka.metaphlan import load_metaphlan
from da_mocka.taxonomy import aggregate_taxonomy, taxonomy_ranks
from da_mocka.tree import RANKS, TOLERANCE, canonicalize


@dataclass
class Dataset:
    phenotypes: pd.DataFrame
    microbiome: pd.DataFrame
    taxonomy: pd.DataFrame | None


def read_table(path: Path) -> pd.DataFrame:
    """Read literal strings; only empty fields are missing."""
    separators = {".tsv": "\t", ".csv": ","}
    if path.suffix.lower() not in separators:
        raise ContractError("Expected TSV or CSV input")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=separators[path.suffix.lower()], strict=True)
            header = next(reader, [])
            if not header or any(not x.strip() for x in header):
                raise ContractError("Empty table or header")
            if len(set(header)) != len(header):
                raise ContractError("Duplicate table headers")
            rows = []
            for row in reader:
                if len(row) != len(header):
                    raise ContractError("Ragged table")
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError("Cannot read input table") from exc
    if not rows:
        raise ContractError("Empty input table")
    return pd.DataFrame(rows, columns=header, dtype="string").replace("", pd.NA)


def _indexed(table, sample_id):
    if sample_id not in table:
        raise ContractError("Missing sample ID column")
    ids = table[sample_id]
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ContractError("Sample IDs must be nonempty and unique")
    return table.set_index(sample_id)


def _numeric(series):
    converted = pd.to_numeric(series, errors="coerce")
    if (series.notna() & converted.isna()).any():
        raise ContractError("Invalid numeric value")
    if not np.isfinite(converted.dropna().to_numpy(dtype=float)).all():
        raise ContractError("Nonfinite numeric value")
    return converted


def _abundance(path: Path, sample_id: str, index: pd.Index) -> pd.DataFrame:
    table = _indexed(read_table(path), sample_id)
    if set(table.index) != set(index):
        raise ContractError("Sample ID sets differ between tables")
    if not table.shape[1]:
        raise ContractError("No microbiome features")
    table = table.loc[index].apply(_numeric)
    x = table.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ContractError("Abundances must be finite and nonnegative")
    return pd.DataFrame(x, index=table.index, columns=table.columns)


def _tree_data(config: Config, x: pd.DataFrame, tree: pd.DataFrame):
    if tree["taxon"].isna().any() or tree["taxon"].duplicated().any():
        raise ContractError("Taxon IDs must be nonempty and unique")
    tree = tree.set_index("taxon")
    if bool(config.rank_tables) != (config.taxonomy_mode == "multilevel"):
        raise ContractError("Only multilevel mode requires rank_tables")
    for rank, path in config.rank_tables.items():
        if rank not in RANKS:
            raise ContractError("Unknown rank table level")
        other = _abundance(path, config.sample_id, x.index)
        if not set(other.columns) <= set(tree.index[tree["rank"] == rank]):
            raise ContractError("Rank table feature does not match declared rank")
        for col in other:
            if col in x:
                if not np.allclose(x[col], other[col], atol=TOLERANCE, rtol=0):
                    raise ContractError("Conflicting abundance in separate rank tables")
            else:
                x[col] = other[col]
    return canonicalize(x, tree, config.taxonomy_mode)


def load_dataset(config: Config) -> Dataset:
    """Validate all inputs and align microbiome rows to phenotype order."""
    phenotypes = _indexed(read_table(config.phenotypes), config.sample_id)
    if set(phenotypes.columns) != set(config.columns):
        raise ContractError("Phenotype columns do not match declared schema")
    for name, spec in config.columns.items():
        values = phenotypes[name]
        if not spec.nullable and values.isna().any():
            raise ContractError("Unexpected missing phenotype")
        if spec.kind in ("integer", "continuous"):
            values = _numeric(values)
            observed = values.dropna()
            if ((observed < spec.minimum) | (observed > spec.maximum)).any():
                raise ContractError("Phenotype outside declared bounds")
            if spec.kind == "integer":
                if (observed % 1 != 0).any():
                    raise ContractError("Nonintegral integer phenotype")
                phenotypes[name] = values.astype("Int64")
            else:
                phenotypes[name] = values.astype("Float64")
        elif spec.kind == "categorical" and not values.dropna().isin(spec.categories).all():
            raise ContractError("Undeclared categorical value")

    check_bmi(phenotypes, config.bmi)
    if config.microbiome_format != "table":
        if config.microbiome_format not in ("metaphlan", "metaphlan_profiles"):
            raise ContractError("Unknown microbiome format")
        if config.taxonomy is not None or config.rank_tables or config.taxonomy_mode != "terminal":
            raise ContractError("MetaPhlAn derives its own taxonomy")
        x, taxonomy = load_metaphlan(
            config.microbiome, profiles=config.microbiome_format == "metaphlan_profiles"
        )
        if set(x.index) != set(phenotypes.index):
            raise ContractError("Sample ID sets differ between tables")
        x = x.loc[phenotypes.index]
        x.index = phenotypes.index
        return Dataset(phenotypes, x, taxonomy)
    microbiome = _abundance(config.microbiome, config.sample_id, phenotypes.index)
    taxonomy = None
    if config.taxonomy is not None:
        raw = read_table(config.taxonomy)
        if set(raw.columns) == {"taxon", "parent", "rank"}:
            microbiome, taxonomy = _tree_data(config, microbiome, raw)
            return Dataset(phenotypes, microbiome, taxonomy)
    if config.taxonomy_mode != "terminal" or config.rank_tables:
        raise ContractError("Hierarchical input requires an explicit taxonomy tree")
    if not np.allclose(microbiome.sum(axis=1), 1.0, atol=TOLERANCE, rtol=0):
        raise ContractError("Abundance rows must sum to one")
    if config.taxonomy is not None:
        taxonomy = raw
        if set(taxonomy.columns) != {"taxon", "genus", "family"}:
            raise ContractError("Taxonomy requires taxon, genus, family")
        blank = taxonomy.apply(lambda col: col.str.strip().eq("")).any().any()
        if taxonomy.isna().any().any() or blank:
            raise ContractError("Incomplete taxonomy mapping")
        if taxonomy["taxon"].duplicated().any():
            raise ContractError("Duplicate taxonomy feature")
        if set(taxonomy["taxon"]) != set(microbiome.columns):
            raise ContractError("Taxonomy must cover exactly the microbiome features")
        if taxonomy.groupby("genus")["family"].nunique().gt(1).any():
            raise ContractError("Genus has inconsistent family mapping")
        taxonomy = taxonomy.set_index("taxon").loc[microbiome.columns]
    return Dataset(phenotypes, microbiome, taxonomy)


def validate_rank_exports(data: Dataset, directory: Path, sample_id: str) -> None:
    """Check all exported rank tables against terminal components, not each other."""
    ranks = taxonomy_ranks(data.taxonomy) if data.taxonomy is not None else ()
    expected_files = {f"microbiome_{rank}.tsv" for rank in ranks}
    actual_files = {path.name for path in directory.glob("microbiome_*.tsv")}
    if actual_files != expected_files:
        raise ContractError("Missing or unexpected exported taxonomic rank table")
    for rank in ranks:
        expected = aggregate_taxonomy(data.microbiome, data.taxonomy, rank)
        actual = _abundance(directory / f"microbiome_{rank}.tsv", sample_id, expected.index)
        if set(actual.columns) != set(expected.columns):
            raise ContractError("Exported taxonomic rank features differ from tree")
        if not np.allclose(actual.loc[:, expected.columns], expected, atol=TOLERANCE, rtol=0):
            raise ContractError("Exported taxonomic abundances disagree with terminal components")
        if not np.allclose(actual.sum(axis=1), 1, atol=TOLERANCE, rtol=0):
            raise ContractError("Exported taxonomic rank must sum to one")
