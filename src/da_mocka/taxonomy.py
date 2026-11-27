"""Aggregate mutually exclusive terminal taxa without double counting."""

import pandas as pd

from da_mocka.config import ContractError
from da_mocka.tree import available_ranks, rank_mapping


def aggregate_taxonomy(microbiome: pd.DataFrame, taxonomy: pd.DataFrame, rank: str) -> pd.DataFrame:
    """Sum terminal features by a validated rank mapping."""
    if rank not in taxonomy_ranks(taxonomy):
        raise ContractError("Unsupported taxonomic rank")
    if "parent" in taxonomy:
        mapping = pd.Index(rank_mapping(taxonomy, microbiome.columns, rank)).to_numpy()
    else:
        if not taxonomy.index.is_unique or set(taxonomy.index) != set(microbiome.columns):
            raise ContractError("Taxonomy must cover exactly the microbiome features")
        if rank not in taxonomy or taxonomy[rank].isna().any():
            raise ContractError("Incomplete taxonomy mapping")
        mapping = taxonomy.loc[microbiome.columns, rank].to_numpy()
    result = microbiome.T.groupby(mapping, sort=True).sum().T
    if microbiome.index.name in result.columns:
        raise ContractError("Taxonomic label collides with sample ID column")
    return result


def taxonomy_ranks(taxonomy: pd.DataFrame) -> tuple[str, ...]:
    """Keep legacy genus/family ordering; expose all represented tree ranks."""
    if "parent" in taxonomy:
        return available_ranks(taxonomy)
    return ("genus", "family")
