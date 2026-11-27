"""Low-rank latent Gaussian approximation for mixed and compositional data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata


@dataclass
class Quantile:
    index: int
    values: np.ndarray
    discrete: bool = False

    def decode(self, z: np.ndarray) -> np.ndarray:
        u = ndtr(z[:, self.index])
        if not len(self.values):
            return np.full(len(z), np.nan)
        if self.discrete:
            indices = np.minimum((u * len(self.values)).astype(int), len(self.values) - 1)
            return self.values[indices]
        return np.interp(u * (len(self.values) - 1), np.arange(len(self.values)), self.values)


@dataclass
class Gate:
    index: int
    probability: float

    def decode(self, z: np.ndarray) -> np.ndarray:
        if self.probability == 0:
            return np.zeros(len(z), dtype=bool)
        if self.probability == 1:
            return np.ones(len(z), dtype=bool)
        return z[:, self.index] > ndtri(1 - self.probability)


class Encoder:
    """Encode observed values; unknown latent coordinates are mean-imputed."""

    def __init__(self) -> None:
        self.columns: list[np.ndarray] = []

    def quantile(self, values: np.ndarray, observed: np.ndarray, discrete=False) -> Quantile:
        selected = values[observed]
        score = np.zeros(len(values))
        if len(selected):
            ranks = rankdata(selected, method="average")
            score[observed] = ndtri((ranks - 0.5) / len(selected))
        result = Quantile(len(self.columns), np.sort(selected), discrete)
        self.columns.append(score)
        return result

    def gate(self, values: np.ndarray, observed: np.ndarray) -> Gate:
        p = float(values[observed].mean()) if observed.any() else 0.0
        score = np.zeros(len(values))
        score[observed] = values[observed].astype(float) - p
        result = Gate(len(self.columns), p)
        self.columns.append(score)
        return result


@dataclass
class LatentFactors:
    loadings: np.ndarray
    residual: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray, rank: int, shrinkage: float) -> LatentFactors:
        """Fit a diagonal-plus-low-rank covariance without a feature covariance matrix."""
        n, p = matrix.shape
        matrix = matrix - matrix.mean(axis=0)
        norm = np.linalg.norm(matrix, axis=0)
        matrix /= np.where(norm > 0, norm, 1)
        k = min(rank, n, p)
        width = min(k + 8, n, p)
        if width < min(n, p):
            rng = np.random.default_rng(0)
            q, _ = np.linalg.qr(matrix @ rng.standard_normal((p, width)), mode="reduced")
            # One power iteration improves the approximation of leading components.
            q, _ = np.linalg.qr(matrix @ (matrix.T @ q), mode="reduced")
            reduced = q.T @ matrix
        else:
            reduced = matrix
        _, singular, vt = np.linalg.svd(reduced, full_matrices=False)
        loadings = vt[:k].T * singular[:k] * np.sqrt(1 - shrinkage)
        residual = np.sqrt(np.maximum(0, 1 - np.sum(loadings**2, axis=1)))
        return cls(loadings, residual)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        shared = rng.standard_normal((n, self.loadings.shape[1])) @ self.loadings.T
        return shared + rng.standard_normal(shared.shape) * self.residual
