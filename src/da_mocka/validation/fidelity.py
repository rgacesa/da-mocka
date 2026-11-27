"""Distributional and dependency diagnostics for validated aligned tables."""

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import ks_2samp, wasserstein_distance

from da_mocka.config import Config
from da_mocka.io import Dataset
from da_mocka.taxonomy import aggregate_taxonomy, taxonomy_ranks


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def _distribution(a, b, scale=1.0) -> dict:
    if not len(a) or not len(b):
        return {"ks": None, "wasserstein": None}
    return {
        "ks": float(ks_2samp(a, b).statistic),
        "wasserstein": float(wasserstein_distance(a, b) / scale),
    }


def _tv(a: pd.Series, b: pd.Series) -> float:
    p, q = a.value_counts(normalize=True), b.value_counts(normalize=True)
    p, q = p.align(q, fill_value=0)
    return float((p - q).abs().sum() / 2)


def _phenotypes(real: Dataset, mock: Dataset, config: Config) -> dict:
    details = {}
    for name, spec in config.columns.items():
        if spec.kind == "identifier":
            continue
        a, b = real.phenotypes[name], mock.phenotypes[name]
        result = {"missing_error": float(abs(a.isna().mean() - b.isna().mean()))}
        if spec.kind == "categorical":
            result["tv"] = (
                _tv(a.dropna(), b.dropna()) if a.notna().any() and b.notna().any() else None
            )
        else:
            scale = spec.maximum - spec.minimum or 1.0
            result.update(
                _distribution(
                    a.dropna().to_numpy(dtype=float), b.dropna().to_numpy(dtype=float), scale
                )
            )
        details[name] = result
    return details


def _alpha(x: np.ndarray) -> dict:
    return {
        "richness": (x > 0).sum(axis=1),
        "shannon": -xlogy(x, x).sum(axis=1),
        "simpson": 1 - (x**2).sum(axis=1),
    }


def _beta(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(x) < 2:
        return np.empty(0)
    # Independent random pairs with distinct indices; bounded workspace.
    n = min(1000, len(x) * 5)
    a = rng.integers(len(x), size=n)
    b = (a + rng.integers(1, len(x), size=n)) % len(x)
    return np.array([np.abs(x[i] - x[j]).sum() / 2 for i, j in zip(a, b, strict=True)])


def _dependency_frame(data: Dataset, config: Config, taxa: list[str]) -> tuple[pd.DataFrame, int]:
    columns = {}
    for name, spec in config.columns.items():
        if spec.kind == "identifier":
            continue
        v = data.phenotypes[name]
        columns[("missing", name, "")] = v.isna().astype(float)
        if spec.kind == "categorical":
            for cat in spec.categories:
                columns[("category", name, cat)] = v.eq(cat).fillna(False).astype(float)
        else:
            columns[("numeric", name, "")] = v.to_numpy(dtype=float, na_value=np.nan)
    p = len(columns)
    for taxon in taxa:
        v = data.microbiome[taxon]
        columns[("abundance", taxon, "")] = v
        columns[("presence", taxon, "")] = v.gt(0).astype(float)
    return pd.DataFrame(columns, index=data.phenotypes.index), p


def _correlations(real: Dataset, mock: Dataset, config: Config, max_features: int) -> dict:
    taxa = real.microbiome.var(axis=0).sort_values(ascending=False, kind="stable")
    selected = taxa.index[:max_features].tolist()
    a, p = _dependency_frame(real, config, selected)
    b, _ = _dependency_frame(mock, config, selected)
    # Bound phenotype indicators too; preserve order and use the same features in both.
    positions = [*range(min(p, max_features)), *range(p, a.shape[1])]
    a, b = a.iloc[:, positions], b.iloc[:, positions]
    p = min(p, max_features)
    ca = a.corr(method="spearman", min_periods=3).to_numpy()
    cb = b.corr(method="spearman", min_periods=3).to_numpy()
    delta = np.abs(ca - cb)
    i, j = np.triu_indices(len(delta), k=1)
    return {
        "phenotype": _mean(delta[i[j < p], j[j < p]]),
        "microbiome": _mean(delta[i[i >= p], j[i >= p]]),
        "cross_modal": _mean(delta[:p, p:].ravel()),
        "taxa_evaluated": len(selected),
        "phenotype_indicators_evaluated": p,
    }


def compare(real: Dataset, mock: Dataset, config: Config, max_features: int, seed: int) -> dict:
    """Compare two nonempty, validated datasets; raw global or conditional."""
    a = real.microbiome.to_numpy(dtype=float)
    b = mock.microbiome.loc[:, real.microbiome.columns].to_numpy(dtype=float)
    phenotypes = _phenotypes(real, mock, config)
    prevalence = np.abs((a > 0).mean(axis=0) - (b > 0).mean(axis=0))
    abundance = np.abs(a.mean(axis=0) - b.mean(axis=0))
    clr_a = np.log(a + 1e-6)
    clr_b = np.log(b + 1e-6)
    clr_a -= clr_a.mean(axis=1, keepdims=True)
    clr_b -= clr_b.mean(axis=1, keepdims=True)
    aa, ab = _alpha(a), _alpha(b)
    rng = np.random.default_rng(seed)
    by_rank = {}
    if real.taxonomy is not None:
        for rank in taxonomy_ranks(real.taxonomy):
            ra = aggregate_taxonomy(real.microbiome, real.taxonomy, rank)
            rb = aggregate_taxonomy(mock.microbiome, real.taxonomy, rank)
            by_rank[rank] = {
                "prevalence_mae": float((ra.gt(0).mean() - rb.gt(0).mean()).abs().mean()),
                "mean_abundance_l1": float((ra.mean() - rb.mean()).abs().sum()),
            }
    return {
        "taxonomy_ranks": by_rank,
        "source_n": len(a),
        "candidate_n": len(b),
        "phenotypes": phenotypes,
        "phenotype_missing_mae": _mean(v["missing_error"] for v in phenotypes.values()),
        "phenotype_ks_mean": _mean(v.get("ks") for v in phenotypes.values()),
        "phenotype_tv_mean": _mean(v.get("tv") for v in phenotypes.values()),
        "prevalence_mae": float(prevalence.mean()),
        "prevalence_max_error": float(prevalence.max()),
        "mean_abundance_l1": float(abundance.sum()),
        "clr_mean_mae": float(np.abs(clr_a.mean(axis=0) - clr_b.mean(axis=0)).mean()),
        "zero_fraction_error": float(abs((a == 0).mean() - (b == 0).mean())),
        "alpha": {key: _distribution(aa[key], ab[key]) for key in aa},
        "beta_bray_curtis": _distribution(_beta(a, rng), _beta(b, rng)),
        "correlation_mae": _correlations(real, mock, config, max_features),
    }
