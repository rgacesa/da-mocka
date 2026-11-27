"""Aggregate mutually exclusive terminal taxa without double counting."""

import pandas as pd

from da_mocka.config import ContractError


def aggregate_taxonomy(
    microbiome: pd.DataFrame, taxonomy: pd.DataFrame, rank: str
) -> pd.DataFrame:
    """Sum terminal features by a validated genus or family mapping."""
    if rank not in ("genus", "family"):
        raise ContractError("Unsupported taxonomic rank")
    if not taxonomy.index.is_unique or set(taxonomy.index) != set(microbiome.columns):
        raise ContractError("Taxonomy must cover exactly the microbiome features")
    if rank not in taxonomy or taxonomy[rank].isna().any():
        raise ContractError("Incomplete taxonomy mapping")
    mapping = taxonomy.loc[microbiome.columns, rank].to_numpy()
    result = microbiome.T.groupby(mapping, sort=True).sum().T
    if microbiome.index.name in result.columns:
        raise ContractError("Taxonomic label collides with sample ID column")
    return result
