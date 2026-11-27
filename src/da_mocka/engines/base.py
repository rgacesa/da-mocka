"""Shared joint sampling interface and engine selection."""

from typing import Protocol

import numpy as np
import pandas as pd

from da_mocka.config import Config, validate_engine
from da_mocka.engines.independent import IndependentEngine
from da_mocka.engines.joint_latent import JointLatentEngine
from da_mocka.io import Dataset


class Engine(Protocol):
    def sample(self, n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def fit_engine(data: Dataset, config: Config) -> Engine:
    validate_engine(config.engine)
    if config.engine.name == "joint_latent":
        return JointLatentEngine.fit(data, config)
    return IndependentEngine.fit(data, config)
