"""Explicit biological relationships; no executable user expressions."""

import numpy as np
import pandas as pd

from da_mocka.config import BmiRule, ContractError


def bmi_values(frame: pd.DataFrame, rule: BmiRule) -> np.ndarray:
    """Calculate BMI in kg/m²; reject invalid arithmetic."""
    weight = frame[rule.weight].to_numpy(dtype=float)
    height = frame[rule.height].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = weight / height**2
    if not np.isfinite(result).all() or (height <= 0).any() or (weight <= 0).any():
        raise ContractError("Invalid BMI inputs")
    return result


def check_bmi(frame: pd.DataFrame, rule: BmiRule | None) -> None:
    """Validate a declared BMI relationship, allowing configured source rounding."""
    if rule is not None:
        actual = frame[rule.target].to_numpy(dtype=float)
        if not np.allclose(actual, bmi_values(frame, rule), atol=rule.atol, rtol=0):
            raise ContractError("Inconsistent BMI relationship")
