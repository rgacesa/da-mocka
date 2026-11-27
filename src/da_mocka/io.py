"""Read and validate aligned phenotype and compositional tables."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from da_mocka.config import Config, ContractError
from da_mocka.constraints import check_bmi


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
    microbiome = _indexed(read_table(config.microbiome), config.sample_id)
    if set(microbiome.index) != set(phenotypes.index):
        raise ContractError("Sample ID sets differ between tables")
    if microbiome.shape[1] == 0:
        raise ContractError("No microbiome features")
    microbiome = microbiome.loc[phenotypes.index].apply(_numeric)
    x = microbiome.to_numpy(dtype=float, na_value=np.nan)
    microbiome = pd.DataFrame(x, index=microbiome.index, columns=microbiome.columns)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ContractError("Abundances must be finite and nonnegative")
    if not np.allclose(x.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ContractError("Abundance rows must sum to one")

    taxonomy = None
    if config.taxonomy is not None:
        taxonomy = read_table(config.taxonomy)
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
