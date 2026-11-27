"""Explicit configuration contracts; no implicit type inference."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

import yaml


class ContractError(ValueError):
    """Invalid input contract."""


@dataclass(frozen=True)
class Column:
    kind: str
    nullable: bool = False
    minimum: float | None = None
    maximum: float | None = None
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class BmiRule:
    """Derive BMI from weight in kg and height in metres."""

    target: str
    weight: str
    height: str
    atol: float = 0.1


@dataclass(frozen=True)
class Stratification:
    """Explicit categorical grouping and allocation settings."""

    columns: tuple[str, ...] = ()
    include: dict[str, tuple[str, ...]] = field(default_factory=dict)
    allocation: str = "proportional"
    balance: tuple[str, ...] = ()
    custom: dict[tuple[str, ...], int] = field(default_factory=dict)
    min_stratum_size: int = 30
    rare_strata: str = "error"
    missing_strata: str = "error"
    pooling_strength: float = 0.0


@dataclass(frozen=True)
class Config:
    sample_id: str
    phenotypes: Path
    microbiome: Path
    taxonomy: Path | None
    columns: dict[str, Column]
    bmi: BmiRule | None = None
    stratification: Stratification = field(default_factory=Stratification)


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently overriding them."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ContractError("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor("tag:yaml.org,2002:map", _mapping)


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys():
        raise ContractError("Missing or invalid configuration keys")
    if value.keys() - set(required) - set(optional):
        raise ContractError("Unknown configuration keys")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _column(raw):
    _keys(raw, ["type"], ["nullable", "min", "max", "categories"])
    kind = raw["type"]
    nullable = raw.get("nullable", False)
    supported = ("continuous", "integer", "categorical", "identifier")
    if not isinstance(kind, str) or kind not in supported:
        raise ContractError("Unsupported phenotype type")
    if not isinstance(nullable, bool):
        raise ContractError("nullable must be boolean")
    if kind in ("continuous", "integer"):
        _keys(raw, ["type", "min", "max"], ["nullable"])
        lo, hi = raw["min"], raw["max"]
        if not _number(lo) or not _number(hi) or lo > hi:
            raise ContractError("Numeric bounds must be finite and ordered")
        if kind == "integer" and (int(lo) != lo or int(hi) != hi):
            raise ContractError("Integer bounds must be integral")
        return Column(kind, nullable, lo, hi)
    if kind == "categorical":
        _keys(raw, ["type", "categories"], ["nullable"])
        cats = raw["categories"]
        if not isinstance(cats, list) or not cats:
            raise ContractError("Categories must be a nonempty list")
        if any(not isinstance(c, str) or not c.strip() for c in cats):
            raise ContractError("Categories must be nonempty strings")
        if len(set(cats)) != len(cats):
            raise ContractError("Duplicate categories")
        return Column(kind, nullable, categories=tuple(cats))
    _keys(raw, ["type"], ["nullable"])
    return Column(kind, nullable)


def _bmi_rule(raw, columns: dict[str, Column]) -> BmiRule:
    _keys(raw, ["target", "weight", "height"], ["atol"])
    names = [raw[key] for key in ("target", "weight", "height")]
    if any(not isinstance(name, str) or name not in columns for name in names):
        raise ContractError("BMI rule requires declared columns")
    if len(set(names)) != 3:
        raise ContractError("BMI rule columns must be distinct")
    for name in names:
        spec = columns[name]
        if spec.kind != "continuous" or spec.nullable or spec.minimum <= 0:
            raise ContractError("BMI rule requires positive nonnullable continuous columns")
    atol = raw.get("atol", 0.1)
    if not _number(atol) or atol < 0:
        raise ContractError("BMI tolerance must be finite and nonnegative")
    return BmiRule(*names, atol)


def _strings(value) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ContractError("Expected a list of nonempty strings")
    if len(set(value)) != len(value):
        raise ContractError("Duplicate stratification values")
    return tuple(value)


def validate_stratification(spec: Stratification, columns: dict[str, Column]) -> None:
    """Validate settings from either the CLI, YAML, or Python API."""
    if len(set(spec.columns)) != len(spec.columns):
        raise ContractError("Duplicate stratification columns")
    for name in spec.columns:
        if name not in columns or columns[name].kind != "categorical":
            raise ContractError("Stratification requires declared categorical columns")
    if spec.allocation not in ("proportional", "balanced", "original", "custom", "marginal"):
        raise ContractError("Unknown allocation mode")
    if spec.rare_strata not in ("error", "suppress"):
        raise ContractError("Unknown rare-stratum policy")
    if spec.missing_strata not in ("error", "exclude"):
        raise ContractError("Unknown missing-stratum policy")
    if type(spec.min_stratum_size) is not int or spec.min_stratum_size < 1:
        raise ContractError("Minimum stratum size must be a positive integer")
    if not _number(spec.pooling_strength) or spec.pooling_strength < 0:
        raise ContractError("Pooling strength must be finite and nonnegative")
    if not spec.columns and (
        spec.include
        or spec.balance
        or spec.custom
        or spec.pooling_strength
        or spec.allocation not in ("proportional", "original")
    ):
        raise ContractError("Specify stratification columns first")
    for name, values in spec.include.items():
        if name not in spec.columns or not values:
            raise ContractError("Outcome selection requires a stratification column")
        if not set(values) <= set(columns[name].categories):
            raise ContractError("Selected outcomes must be declared categories")
    if spec.allocation == "marginal":
        if not spec.balance or not set(spec.balance) <= set(spec.columns):
            raise ContractError("Marginal allocation requires balance columns")
        if len(set(spec.balance)) != len(spec.balance):
            raise ContractError("Duplicate balance columns")
    elif spec.balance:
        raise ContractError("Balance columns require marginal allocation")
    if spec.allocation == "custom":
        if not spec.custom:
            raise ContractError("Custom allocation requires explicit counts")
        for key, count in spec.custom.items():
            if len(key) != len(spec.columns) or type(count) is not int or count < 0:
                raise ContractError("Invalid custom stratum or count")
            for name, value in zip(spec.columns, key, strict=True):
                if value not in columns[name].categories:
                    raise ContractError("Custom stratum contains an undeclared category")
        if not sum(spec.custom.values()):
            raise ContractError("Custom allocation must request at least one sample")
    elif spec.custom:
        raise ContractError("Custom counts require custom allocation")


def _stratification(raw, columns: dict[str, Column]) -> Stratification:
    _keys(
        raw,
        ["columns"],
        [
            "include",
            "allocation",
            "balance",
            "custom",
            "min_stratum_size",
            "rare_strata",
            "missing_strata",
            "pooling_strength",
        ],
    )
    include = raw.get("include", {})
    if not isinstance(include, dict):
        raise ContractError("Outcome selection must be a mapping")
    custom = raw.get("custom", [])
    if not isinstance(custom, list):
        raise ContractError("Custom allocation must be a list")
    counts = {}
    for entry in custom:
        _keys(entry, ["values", "count"])
        # Repeated labels across different dimensions are valid.
        values = entry["values"]
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ContractError("Custom values must be strings")
        key = tuple(values)
        if key in counts:
            raise ContractError("Duplicate custom stratum")
        counts[key] = entry["count"]
    spec = Stratification(
        columns=_strings(raw["columns"]),
        include={name: _strings(values) for name, values in include.items()},
        allocation=raw.get("allocation", "proportional"),
        balance=_strings(raw.get("balance", [])),
        custom=counts,
        min_stratum_size=raw.get("min_stratum_size", 30),
        rare_strata=raw.get("rare_strata", "error"),
        missing_strata=raw.get("missing_strata", "error"),
        pooling_strength=raw.get("pooling_strength", 0.0),
    )
    validate_stratification(spec, columns)
    return spec


def load_config(path: Path | str) -> Config:
    """Load a strict YAML config; resolve paths relative to its directory."""
    path = Path(path)
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ContractError("Cannot read configuration") from exc
    _keys(
        raw,
        ["sample_id", "phenotypes", "microbiome", "columns"],
        ["taxonomy", "bmi", "stratification"],
    )
    sample_id = raw["sample_id"]
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ContractError("Invalid sample ID column")
    columns = raw["columns"]
    if not isinstance(columns, dict) or not columns or sample_id in columns:
        raise ContractError("Declare phenotype columns separately from sample ID")
    if any(not name.strip() for name in columns):
        raise ContractError("Empty column name")

    def resolve(key):
        value = raw[key]
        if not isinstance(value, str) or not value.strip():
            raise ContractError("Input paths must be nonempty strings")
        return path.parent / value

    specs = {name: _column(spec) for name, spec in columns.items()}
    return Config(
        sample_id,
        resolve("phenotypes"),
        resolve("microbiome"),
        resolve("taxonomy") if "taxonomy" in raw else None,
        specs,
        _bmi_rule(raw["bmi"], specs) if "bmi" in raw else None,
        _stratification(raw["stratification"], specs)
        if "stratification" in raw
        else Stratification(),
    )
