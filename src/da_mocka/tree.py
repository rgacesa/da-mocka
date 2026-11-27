"""Explicit taxon IDs, parent links, and disjoint abundance components."""

from collections import defaultdict

import numpy as np
import pandas as pd

from da_mocka.config import ContractError

RANKS = ("domain", "kingdom", "phylum", "class", "order", "family", "genus", "species", "strain")
RESIDUAL = "__unresolved__:"
TOLERANCE = 1e-6


def _paths(tree: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Validate links and strict rank order; return node-to-root paths."""
    if set(tree.columns) != {"parent", "rank"} or not tree.index.is_unique:
        raise ContractError("Tree requires unique taxon IDs, parent and rank")
    if tree.empty or tree.index.to_series().str.strip().eq("").any():
        raise ContractError("Empty taxonomic tree or node ID")
    parents = tree["parent"].fillna("").to_dict()
    ranks = tree["rank"].to_dict()
    order = {rank: i for i, rank in enumerate((*RANKS, "unresolved"))}
    for node, parent in parents.items():
        rank = ranks[node]
        if pd.isna(rank) or rank not in order:
            raise ContractError("Unknown taxonomic rank")
        if rank == "unresolved":
            if not parent or node != RESIDUAL + parent:
                raise ContractError("Invalid unresolved component ID")
        elif node.startswith(RESIDUAL):
            raise ContractError("Reserved taxon ID prefix")
        if parent:
            if parent not in parents:
                raise ContractError("Unknown taxonomic parent")
            if pd.isna(ranks[parent]) or ranks[parent] not in order:
                raise ContractError("Unknown taxonomic rank")
            if ranks[parent] == "unresolved":
                raise ContractError("Unresolved component cannot have children")
            if order[ranks[parent]] >= order[rank]:
                raise ContractError("Cyclic hierarchy or inconsistent rank order")
    paths = {}
    for node in parents:
        path = [node]
        while parents[path[-1]]:
            parent = parents[path[-1]]
            if parent in path:
                raise ContractError("Cyclic taxonomic hierarchy")
            path.append(parent)
        paths[node] = tuple(path)
    return paths


def canonicalize(
    abundance: pd.DataFrame, tree: pd.DataFrame, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert measured inclusive totals into exclusive terminal components."""
    paths = _paths(tree)
    measured = set(abundance.columns)
    if not measured <= paths.keys():
        raise ContractError("Taxonomy must cover every measured feature")
    if mode not in ("terminal", "hierarchical", "multilevel"):
        raise ContractError("Unknown taxonomy input mode")
    children = defaultdict(list)
    for node in abundance.columns:
        parent = next((p for p in paths[node][1:] if p in measured), None)
        if parent is not None:
            children[parent].append(node)
    if mode == "terminal" and children:
        raise ContractError("Terminal input contains overlapping ancestor and descendant taxa")
    used = {ancestor for node in measured for ancestor in paths[node]}
    canonical = tree.loc[sorted(used), ["parent", "rank"]].copy()
    components = {}
    for node in abundance.columns:
        values = abundance[node].to_numpy(dtype=float)
        if node in children:
            remainder = values - abundance[children[node]].sum(axis=1).to_numpy()
            if (remainder < -TOLERANCE).any():
                raise ContractError("Measured children exceed parent abundance")
            label = RESIDUAL + node
            if label in canonical.index:
                raise ContractError("Overlapping explicit and inferred unresolved components")
            canonical.loc[label] = [node, "unresolved"]
            components[label] = np.maximum(remainder, 0)
        else:
            components[node] = values
    result = pd.DataFrame(components, index=abundance.index)
    total = result.sum(axis=1).to_numpy()
    if not np.allclose(total, 1, atol=TOLERANCE, rtol=0):
        raise ContractError("Disjoint abundance components must sum to one")
    # Correct only rounding-sized differences in inclusive input.
    if mode != "terminal":
        result = result.div(total, axis=0)
    canonical = canonical.sort_index().astype("string")
    canonical_paths = _paths(canonical)
    if mode != "terminal":
        for node in abundance.columns:
            descendants = [col for col in result if node in canonical_paths[col]]
            reconstructed = result[descendants].sum(axis=1)
            if not np.allclose(reconstructed, abundance[node], atol=TOLERANCE, rtol=0):
                raise ContractError("Reconstructed parent differs from supplied abundance")
    return result, canonical


def available_ranks(tree: pd.DataFrame) -> tuple[str, ...]:
    return tuple(rank for rank in RANKS if rank in set(tree["rank"]))


def rank_mapping(tree: pd.DataFrame, features: pd.Index, rank: str) -> list[str]:
    """Partition leaves at rank, retaining unresolved mass under known ancestors."""
    if rank not in available_ranks(tree):
        raise ContractError("Unsupported taxonomic rank")
    paths = _paths(tree)
    if not set(features) <= paths.keys():
        raise ContractError("Taxonomy must cover every measured feature")
    measured = set(features)
    if any(measured.intersection(paths[node][1:]) for node in features):
        raise ContractError("Cannot aggregate overlapping terminal taxa")
    ranks = tree["rank"].to_dict()
    level = RANKS.index(rank)
    labels = []
    for node in features:
        path = paths[node]
        match = next((p for p in path if ranks[p] == rank), None)
        if match is None:
            ancestor = next(
                (p for p in path if ranks[p] in RANKS and RANKS.index(ranks[p]) < level),
                None,
            )
            # Missing upper ranks are kept distinct by their highest known descendant.
            anchor = ancestor if ancestor is not None else path[-1]
            match = f"{RESIDUAL}{rank}:{anchor}"
            if match in paths:
                raise ContractError("Unresolved rank label collides with taxon ID")
        labels.append(match)
    return labels
