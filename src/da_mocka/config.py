"""Explicit configuration contracts; no implicit type inference."""

from dataclasses import dataclass
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
class Config:
    sample_id: str
    phenotypes: Path
    microbiome: Path
    taxonomy: Path | None
    columns: dict[str, Column]
    bmi: BmiRule | None = None


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


def load_config(path: Path | str) -> Config:
    """Load a strict YAML config; resolve paths relative to its directory."""
    path = Path(path)
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ContractError("Cannot read configuration") from exc
    _keys(raw, ["sample_id", "phenotypes", "microbiome", "columns"], ["taxonomy", "bmi"])
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
    )
