"""Bounded local disclosure diagnostics; none provides a privacy guarantee."""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from da_mocka.config import Config, ContractError
from da_mocka.io import Dataset


def subset(data: Dataset, positions: np.ndarray) -> Dataset:
    return Dataset(data.phenotypes.iloc[positions], data.microbiome.iloc[positions], data.taxonomy)


def bounded(data: Dataset, cap: int, rng: np.random.Generator) -> Dataset:
    if len(data.phenotypes) <= cap:
        return data
    return subset(data, np.sort(rng.choice(len(data.phenotypes), cap, replace=False)))


def _hashes(data: Dataset, config: Config) -> tuple[np.ndarray, np.ndarray]:
    names = [name for name, spec in config.columns.items() if spec.kind != "identifier"]
    p = pd.util.hash_pandas_object(data.phenotypes[names], index=False).to_numpy()
    x = data.microbiome.astype(float).copy()
    # Canonicalize signed zeros.
    x = x.mask(x == 0, 0.0)
    m = pd.util.hash_pandas_object(x, index=False).to_numpy()
    return p, m


def exact_overlap(real: Dataset, mock: Dataset, config: Config) -> dict:
    """Full-table hash overlap, excluding declared identifiers; no row hashes exported."""
    a, b = _hashes(real, config), _hashes(mock, config)
    source_pairs = set(zip(*a, strict=True))
    joint = sum(pair in source_pairs for pair in zip(*b, strict=True))
    return {
        "phenotype_match_rate": float(np.isin(b[0], a[0]).mean()),
        "microbiome_match_rate": float(np.isin(b[1], a[1]).mean()),
        "joint_match_rate": joint / len(b[0]),
    }


def _phenotype_distance(
    a: Dataset, b: Dataset, config: Config, exclude: str | None = None
) -> np.ndarray:
    """Range-scaled mixed-type Gower-like distance with explicit missingness."""
    result = np.zeros((len(a.phenotypes), len(b.phenotypes)))
    n = 0
    for name, spec in config.columns.items():
        if spec.kind == "identifier" or name == exclude:
            continue
        av, bv = a.phenotypes[name], b.phenotypes[name]
        am, bm = av.isna().to_numpy(), bv.isna().to_numpy()
        if spec.kind == "categorical":
            x = av.to_numpy(dtype=object, na_value=None)
            y = bv.to_numpy(dtype=object, na_value=None)
            delta = (x[:, None] != y[None, :]).astype(float)
        else:
            x = av.to_numpy(dtype=float, na_value=0)
            y = bv.to_numpy(dtype=float, na_value=0)
            scale = spec.maximum - spec.minimum or 1.0
            delta = np.minimum(np.abs(x[:, None] - y[None, :]) / scale, 1)
        delta[am[:, None] | bm[None, :]] = (am[:, None] != bm[None, :])[am[:, None] | bm[None, :]]
        result += delta
        n += 1
    if not n:
        raise ContractError("No phenotype predictors available for distance")
    return result / n


def distances(a: Dataset, b: Dataset, config: Config) -> dict[str, np.ndarray]:
    p = _phenotype_distance(a, b, config)
    x = (
        cdist(
            a.microbiome.to_numpy(dtype=float),
            b.microbiome.loc[:, a.microbiome.columns].to_numpy(dtype=float),
            metric="cityblock",
        )
        / 2
    )
    return {"phenotype": p, "microbiome": x, "joint": (p + x) / 2}


def _summary(values: np.ndarray) -> dict:
    return {
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
    }


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    ranks = rankdata(np.concatenate([positive, negative]))
    n, m = len(positive), len(negative)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def _rare(real: Dataset, mock: Dataset, config: Config, threshold: int) -> dict:
    names = [name for name, spec in config.columns.items() if spec.kind == "categorical"]
    if not names:
        return {"rare_match_rate": None, "unseen_combination_rate": None}
    a = pd.util.hash_pandas_object(real.phenotypes[names], index=False)
    b = pd.util.hash_pandas_object(mock.phenotypes[names], index=False)
    counts = Counter(a)
    sizes = np.array([counts[value] for value in b])
    return {
        "rare_match_rate": float(((sizes > 0) & (sizes < threshold)).mean()),
        "unseen_combination_rate": float((sizes == 0).mean()),
    }


def _attribute(real: Dataset, mock: Dataset, holdout: Dataset, config: Config, target: str) -> dict:
    """Nearest synthetic phenotype attack versus training-majority baseline."""
    if target not in config.columns or config.columns[target].kind != "categorical":
        raise ContractError("Attribute target must be a declared categorical phenotype")
    if not any(
        name != target and spec.kind != "identifier" for name, spec in config.columns.items()
    ):
        raise ContractError("Attribute attack requires another phenotype predictor")
    a = real.phenotypes[target].dropna()
    b = mock.phenotypes[target].notna().to_numpy()
    h = holdout.phenotypes[target].notna().to_numpy()
    if a.empty or not b.any() or not h.any():
        return {"accuracy": None, "baseline_accuracy": None, "advantage": None}
    candidates = subset(mock, np.flatnonzero(b))
    queries = subset(holdout, np.flatnonzero(h))
    distance = _phenotype_distance(queries, candidates, config, exclude=target)
    predicted = candidates.phenotypes[target].to_numpy()[distance.argmin(axis=1)]
    truth = queries.phenotypes[target].to_numpy()
    accuracy = float((predicted == truth).mean())
    baseline = float((truth == a.value_counts().index[0]).mean())
    return {"accuracy": accuracy, "baseline_accuracy": baseline, "advantage": accuracy - baseline}


def evaluate(
    real: Dataset,
    mock: Dataset,
    config: Config,
    cap: int,
    seed: int,
    holdout: Dataset | None = None,
    targets: tuple[str, ...] = (),
    *,
    training: Dataset | None = None,
) -> dict:
    """Full overlap plus capped nearest-neighbour and optional holdout attacks."""
    rng = np.random.default_rng(seed)
    a, b = bounded(real, cap, rng), bounded(mock, cap, rng)
    nearest = distances(a, b, config)
    result = {
        "exact": exact_overlap(real, mock, config),
        "rare_categories": _rare(real, mock, config, config.stratification.min_stratum_size),
        "distance_sample_sizes": {"source": len(a.phenotypes), "candidate": len(b.phenotypes)},
        "candidate_to_source": {key: _summary(v.min(axis=0)) for key, v in nearest.items()},
        "membership": None,
        "attribute": None,
    }
    if holdout is not None:
        train = real if training is None else training
        size = min(len(train.phenotypes), len(holdout.phenotypes), cap)
        t = bounded(train, size, rng)
        h = bounded(holdout, size, rng)
        member_distances = distances(t, b, config)
        reference = distances(h, b, config)
        result["distance_sample_sizes"]["membership_reference"] = len(t.phenotypes)
        result["candidate_to_membership_reference"] = {
            key: _summary(v.min(axis=0)) for key, v in member_distances.items()
        }
        result["distance_sample_sizes"]["holdout"] = len(h.phenotypes)
        result["candidate_to_holdout"] = {
            key: _summary(v.min(axis=0)) for key, v in reference.items()
        }
        result["membership"] = {
            key: {"auc": _auc(-v.min(axis=1), -reference[key].min(axis=1))}
            for key, v in member_distances.items()
        }
        result["attribute"] = {target: _attribute(t, b, h, config, target) for target in targets}
    elif targets:
        raise ContractError("Attribute inference requires an independent holdout")
    return result
