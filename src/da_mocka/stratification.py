"""Categorical stratum planning, exact allocation and conservative pooling."""

import warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from da_mocka.config import Config, ContractError, validate_stratification
from da_mocka.engines.independent import IndependentEngine
from da_mocka.io import Dataset

Key = tuple[str, ...]


@dataclass
class Stratum:
    key: Key
    ids: pd.Index
    count: int


def apportion(total: int, weights: list[int]) -> list[int]:
    """Integer largest remainder allocation, with deterministic tie breaking."""
    denominator = sum(weights)
    if denominator <= 0 or total < 0:
        raise ContractError("Invalid allocation weights or total")
    parts = [divmod(total * weight, denominator) for weight in weights]
    counts = [whole for whole, _ in parts]
    order = sorted(range(len(parts)), key=lambda i: (-parts[i][1], i))
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    return counts


def _allocate(groups: dict[Key, pd.Index], config: Config, n: int | None) -> list[int]:
    spec = config.stratification
    keys = list(groups)
    sizes = [len(groups[key]) for key in keys]
    if spec.allocation == "original":
        counts = sizes
    elif spec.allocation == "custom":
        if set(spec.custom) - set(keys):
            raise ContractError("Custom allocation references an unavailable stratum")
        counts = [spec.custom.get(key, 0) for key in keys]
    else:
        if n is None:
            raise ContractError("Sample count is required for this allocation")
        if spec.allocation == "balanced":
            return apportion(n, [1] * len(keys))
        if spec.allocation == "marginal":
            positions = [spec.columns.index(name) for name in spec.balance]
            buckets = defaultdict(list)
            for i, key in enumerate(keys):
                buckets[tuple(key[j] for j in positions)].append(i)
            counts = [0] * len(keys)
            ordered = sorted(buckets)
            totals = apportion(n, [1] * len(ordered))
            for bucket, total in zip(ordered, totals, strict=True):
                members = buckets[bucket]
                allocated = apportion(total, [sizes[i] for i in members])
                for i, count in zip(members, allocated, strict=True):
                    counts[i] = count
            return counts
        return apportion(n, sizes)
    if n is not None and sum(counts) != n:
        raise ContractError("Requested sample count conflicts with allocation")
    return counts


def plan_strata(data: Dataset, config: Config, n: int | None) -> list[Stratum]:
    """Filter outcomes, reject or exclude missing keys, and gate sparse groups."""
    spec = config.stratification
    validate_stratification(spec, config.columns)
    if n is not None and n < 1:
        raise ContractError("Positive sample count required")
    p = data.phenotypes
    if not spec.columns:
        groups = {(): p.index}
        counts = _allocate(groups, config, n)
        return [Stratum((), p.index, counts[0])]
    for name, outcomes in spec.include.items():
        # Missing values on an explicitly selected column are outside the selection.
        p = p.loc[p[name].isin(outcomes)]
    if p.empty:
        raise ContractError("Outcome selection has no eligible samples")
    selected = list(spec.columns)
    missing = p[selected].isna().any(axis=1)
    if missing.any():
        if spec.missing_strata == "error":
            raise ContractError("Missing stratification values")
        warnings.warn("Rows with missing stratification values excluded", stacklevel=2)
        p = p.loc[~missing]
    for name, outcomes in spec.include.items():
        if not set(outcomes) <= set(p[name]):
            raise ContractError("Selected outcome not observed in eligible data")
    grouped = p.groupby(selected, sort=True, observed=True)
    groups = {}
    suppressed = False
    for key, frame in grouped:
        if len(frame) < spec.min_stratum_size:
            if spec.rare_strata == "error":
                raise ContractError("Stratum below minimum source size")
            suppressed = True
            continue
        groups[key] = frame.index
    if not groups:
        raise ContractError("No eligible strata remain")
    if suppressed:
        warnings.warn("Small strata suppressed; allocations use eligible strata only", stacklevel=2)
    counts = _allocate(groups, config, n)
    return [Stratum(key, groups[key], count) for key, count in zip(groups, counts, strict=True)]


def _fit_subset(data: Dataset, config: Config, ids: pd.Index) -> IndependentEngine:
    subset = Dataset(data.phenotypes.loc[ids], data.microbiome.loc[ids], data.taxonomy)
    return IndependentEngine.fit(subset, config)


def sample_strata(
    data: Dataset, config: Config, strata: list[Stratum], rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit independent models per stratum, optionally mix with prefix-parent models."""
    spec = config.stratification
    phenotype_parts = []
    microbiome_parts = []
    cache: dict[Key, IndependentEngine] = {}
    for group in strata:
        if not group.count:
            continue
        remaining = group.count
        key = group.key
        while remaining:
            ids = group.ids
            if key != group.key:
                ids = pd.Index(
                    [
                        sample
                        for sibling in strata
                        if sibling.key[: len(key)] == key
                        for sample in sibling.ids
                    ]
                )
            if key and spec.pooling_strength:
                weight = len(ids) / (len(ids) + spec.pooling_strength)
                count = int(rng.binomial(remaining, weight))
            else:
                count = remaining
            if count:
                if key not in cache:
                    cache[key] = _fit_subset(data, config, ids)
                engine = cache[key]
                p = engine.sample_phenotypes(count, rng)
                x = engine.sample_microbiome(count, rng)
                for name, value in zip(spec.columns, group.key, strict=True):
                    p[name] = pd.Series(value, index=p.index, dtype="string")
                phenotype_parts.append(p)
                microbiome_parts.append(x)
            remaining -= count
            key = key[:-1]
    p = pd.concat(phenotype_parts, ignore_index=True)
    x = pd.concat(microbiome_parts, ignore_index=True)
    # Shuffle both modalities identically so group order is not exposed by row order.
    order = rng.permutation(len(p))
    return p.iloc[order].reset_index(drop=True), x.iloc[order].reset_index(drop=True)
