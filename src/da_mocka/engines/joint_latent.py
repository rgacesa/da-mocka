"""Joint phenotype, missingness and microbial hurdle model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from da_mocka.config import Column, Config, ContractError
from da_mocka.constraints import bmi_values
from da_mocka.engines.latent import Encoder, Gate, LatentFactors, Quantile
from da_mocka.io import Dataset

MAX_ATTEMPTS = 100


@dataclass
class Phenotype:
    spec: Column
    missing: Gate
    numeric: Quantile | None
    categories: tuple[str, ...]
    gates: list[Gate]

    @classmethod
    def fit(cls, series: pd.Series, spec: Column, encoder: Encoder) -> Phenotype:
        observed = series.notna().to_numpy()
        missing = encoder.gate(~observed, np.ones(len(series), dtype=bool))
        if spec.kind != "categorical":
            values = series.to_numpy(dtype=float, na_value=np.nan)
            numeric = encoder.quantile(values, observed, discrete=spec.kind == "integer")
            return cls(spec, missing, numeric, (), [])
        values = series.to_numpy(dtype=object, na_value=None)
        categories = tuple(c for c in spec.categories if (values == c).any())
        remaining = observed.copy()
        gates = []
        # Sequential binary gates, each fitted only among remaining observed categories.
        for category in categories[:-1]:
            match = values == category
            gates.append(encoder.gate(match, remaining))
            remaining &= ~match
        return cls(spec, missing, None, categories, gates)

    def decode(self, z: np.ndarray) -> pd.Series:
        if self.numeric is not None:
            dtype = "Int64" if self.spec.kind == "integer" else "Float64"
            result = pd.Series(self.numeric.decode(z), dtype=dtype)
        else:
            result = pd.Series(pd.NA, index=range(len(z)), dtype="string")
            if self.categories:
                result[:] = self.categories[-1]
                remaining = np.ones(len(z), dtype=bool)
                for category, gate in zip(self.categories[:-1], self.gates, strict=True):
                    selected = remaining & gate.decode(z)
                    result.loc[selected] = category
                    remaining &= ~selected
        result.loc[self.missing.decode(z)] = pd.NA
        return result


@dataclass
class JointLatentEngine:
    config: Config
    phenotypes: dict[str, Phenotype]
    taxa: list[str]
    presence: list[Gate]
    abundance: list[Quantile]
    factors: LatentFactors

    @classmethod
    def fit(cls, data: Dataset, config: Config) -> JointLatentEngine:
        encoder = Encoder()
        target = config.bmi.target if config.bmi else None
        phenotypes = {
            name: Phenotype.fit(data.phenotypes[name], spec, encoder)
            for name, spec in config.columns.items()
            if spec.kind != "identifier" and name != target
        }
        if not phenotypes:
            raise ContractError("No nonidentifier phenotypes available")
        presence = []
        abundance = []
        x = data.microbiome.to_numpy(dtype=float)
        for col in x.T:
            present = col > 0
            presence.append(encoder.gate(present, np.ones(len(col), dtype=bool)))
            logs = np.zeros(len(col))
            logs[present] = np.log(col[present])
            abundance.append(encoder.quantile(logs, present))
        factors = LatentFactors.fit(
            np.column_stack(encoder.columns), config.engine.rank, config.engine.shrinkage
        )
        return cls(config, phenotypes, list(data.microbiome.columns), presence, abundance, factors)

    def _draw(
        self, n: int, rng: np.random.Generator
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        z = self.factors.sample(n, rng)
        p = pd.DataFrame({name: model.decode(z) for name, model in self.phenotypes.items()})
        x = np.zeros((n, len(self.taxa)))
        for j, (gate, quantile) in enumerate(zip(self.presence, self.abundance, strict=True)):
            present = gate.decode(z)
            if present.any():
                x[present, j] = np.exp(quantile.decode(z)[present])
        totals = x.sum(axis=1)
        valid = np.isfinite(totals) & (totals > 0)
        x[valid] /= totals[valid, None]
        rule = self.config.bmi
        if rule is not None:
            spec = self.config.columns[rule.target]
            values = bmi_values(p, rule)
            valid &= (values >= spec.minimum) & (values <= spec.maximum)
            p[rule.target] = pd.Series(values, dtype="Float64")
        columns = [name for name, spec in self.config.columns.items() if spec.kind != "identifier"]
        return p[columns], x, valid

    def sample(self, n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Draw jointly; reject entire invalid rows to keep modalities aligned."""
        if n < 1:
            raise ContractError("Positive sample count required")
        p_parts = []
        x_parts = []
        for start in range(0, n, self.config.engine.batch_size):
            remaining = min(n - start, self.config.engine.batch_size)
            for _ in range(MAX_ATTEMPTS):
                p, x, valid = self._draw(remaining, rng)
                p_parts.append(p.loc[valid])
                x_parts.append(x[valid])
                remaining -= int(valid.sum())
                if not remaining:
                    break
            if remaining:
                raise ContractError("Unable to satisfy joint biological constraints")
        return (
            pd.concat(p_parts, ignore_index=True),
            pd.DataFrame(np.concatenate(x_parts), columns=self.taxa),
        )
