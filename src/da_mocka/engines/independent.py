"""Independent marginal baseline; deliberately does not preserve correlations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from da_mocka.config import Column, Config, ContractError
from da_mocka.constraints import bmi_values
from da_mocka.io import Dataset

MAX_ATTEMPTS = 100


def _interpolate(sorted_values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw from a piecewise-linear empirical quantile function."""
    positions = rng.uniform(0, len(sorted_values) - 1, n)
    return np.interp(positions, np.arange(len(sorted_values)), sorted_values)


@dataclass
class Marginal:
    spec: Column
    observed: np.ndarray
    missing_rate: float

    @classmethod
    def fit(cls, series: pd.Series, spec: Column) -> Marginal:
        observed = series.dropna()
        values = (
            np.sort(observed.to_numpy(dtype=float))
            if spec.kind == "continuous"
            else observed.to_numpy()
        )
        return cls(spec, values, float(series.isna().mean()))

    def sample(self, n: int, rng: np.random.Generator) -> pd.Series:
        dtype = {"continuous": "Float64", "integer": "Int64", "categorical": "string"}[
            self.spec.kind
        ]
        if not len(self.observed):
            return pd.Series(pd.NA, index=range(n), dtype=dtype)
        if self.spec.kind == "continuous":
            values = _interpolate(self.observed, n, rng)
        else:
            values = rng.choice(self.observed, n)
        result = pd.Series(values, dtype=dtype)
        if self.spec.nullable:
            result[rng.random(n) < self.missing_rate] = pd.NA
        return result


@dataclass
class IndependentEngine:
    config: Config
    marginals: dict[str, Marginal]
    taxa: list[str]
    prevalence: np.ndarray
    log_positive: list[np.ndarray]

    @classmethod
    def fit(cls, data: Dataset, config: Config) -> IndependentEngine:
        """Fit univariate marginals from a validated, aligned dataset."""
        target = config.bmi.target if config.bmi else None
        marginals = {
            name: Marginal.fit(data.phenotypes[name], spec)
            for name, spec in config.columns.items()
            if spec.kind != "identifier" and name != target
        }
        if not marginals:
            raise ContractError("No nonidentifier phenotypes available")
        x = data.microbiome.to_numpy(dtype=float)
        positive = x > 0
        logs = [np.sort(np.log(col[col > 0])) for col in x.T]
        return cls(config, marginals, list(data.microbiome.columns), positive.mean(0), logs)

    def sample_phenotypes(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        frame = pd.DataFrame(
            {name: marginal.sample(n, rng) for name, marginal in self.marginals.items()}
        )
        rule = self.config.bmi
        if rule is not None:
            spec = self.config.columns[rule.target]
            values = bmi_values(frame, rule)
            for _ in range(MAX_ATTEMPTS):
                invalid = (values < spec.minimum) | (values > spec.maximum)
                if not invalid.any():
                    break
                for name in (rule.weight, rule.height):
                    replacement = self.marginals[name].sample(int(invalid.sum()), rng)
                    frame.loc[invalid, name] = replacement.to_numpy()
                values = bmi_values(frame, rule)
            if ((values < spec.minimum) | (values > spec.maximum)).any():
                raise ContractError("Unable to satisfy BMI bounds")
            frame[rule.target] = pd.Series(values, dtype="Float64")
        columns = [name for name, spec in self.config.columns.items() if spec.kind != "identifier"]
        return frame[columns]

    def _draw_communities(self, n: int, rng: np.random.Generator) -> np.ndarray:
        x = np.zeros((n, len(self.taxa)), dtype=float)
        for j, positive in enumerate(self.log_positive):
            if not len(positive):
                continue
            present = rng.random(n) < self.prevalence[j]
            x[present, j] = np.exp(_interpolate(positive, int(present.sum()), rng))
        return x

    def sample_microbiome(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        x = self._draw_communities(n, rng)
        for _ in range(MAX_ATTEMPTS):
            empty = ~x.any(axis=1)
            if not empty.any():
                break
            x[empty] = self._draw_communities(int(empty.sum()), rng)
        totals = x.sum(axis=1, keepdims=True)
        if (totals <= 0).any() or not np.isfinite(totals).all():
            raise ContractError("Unable to generate nonempty microbial communities")
        return pd.DataFrame(x / totals, columns=self.taxa)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Draw aligned independent phenotype and microbial tables."""
        return self.sample_phenotypes(n, rng), self.sample_microbiome(n, rng)
