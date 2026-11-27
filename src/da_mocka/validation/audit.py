"""Authoritative schema checks, allocation-aware audit and configurable gates."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from da_mocka import __version__
from da_mocka.config import Config, ContractError, load_config, validate_audit
from da_mocka.io import Dataset, load_dataset
from da_mocka.stratification import Stratum, plan_strata
from da_mocka.validation.fidelity import compare
from da_mocka.validation.privacy import evaluate, subset


def _check_schema(real: Dataset, mock: Dataset, config: Config) -> None:
    expected = {name for name, spec in config.columns.items() if spec.kind != "identifier"}
    if set(mock.phenotypes.columns) != expected:
        raise ContractError("Candidate phenotype schema differs from source contract")
    if not expected:
        raise ContractError("No nonidentifier phenotypes available for audit")
    if set(mock.microbiome.columns) != set(real.microbiome.columns):
        raise ContractError("Candidate microbial features differ from source")
    if (real.taxonomy is None) != (mock.taxonomy is None):
        raise ContractError("Candidate taxonomy differs from source")
    if real.taxonomy is not None and not real.taxonomy.sort_index().equals(
        mock.taxonomy.sort_index()
    ):
        raise ContractError("Candidate taxonomy differs from source")
    identifiers = set(real.phenotypes.index)
    for name, spec in config.columns.items():
        if spec.kind == "identifier":
            identifiers.update(real.phenotypes[name].dropna())
    if identifiers.intersection(mock.phenotypes.index):
        raise ContractError("Candidate contains source identifiers")
    for name in expected:
        if config.columns[name].kind == "categorical" and not set(
            mock.phenotypes[name].dropna()
        ) <= set(real.phenotypes[name].dropna()):
            raise ContractError("Candidate contains an unobserved categorical value")
    mock.microbiome = mock.microbiome.loc[:, real.microbiome.columns]


def load_candidate(config: Config, directory: Path) -> Dataset:
    """Use source contract, never trust candidate-supplied bounds or categories."""
    schema = replace(
        config,
        phenotypes=directory / "phenotypes.tsv",
        microbiome=directory / "microbiome.tsv",
        taxonomy=directory / "taxonomy.tsv" if config.taxonomy is not None else None,
        columns={k: v for k, v in config.columns.items() if v.kind != "identifier"},
    )
    return load_dataset(schema)


def load_holdout(config: Config, path: Path) -> Dataset:
    """Use only paths from the holdout config; enforce original source contract."""
    other = load_config(path)
    schema = replace(
        config,
        sample_id=other.sample_id,
        phenotypes=other.phenotypes,
        microbiome=other.microbiome,
        taxonomy=other.taxonomy,
    )
    return load_dataset(schema)


def _group_key(frame: pd.DataFrame, names: tuple[str, ...]) -> dict:
    if not names:
        return {(): np.arange(len(frame))}
    if frame[list(names)].isna().any().any():
        raise ContractError("Candidate has missing stratification labels")
    groups = frame.groupby(list(names), sort=True, observed=True).indices
    return {key if isinstance(key, tuple) else (key,): value for key, value in groups.items()}


def _conditional(
    real: Dataset, mock: Dataset, config: Config, strata: list[Stratum]
) -> tuple[list[dict], dict, dict | None]:
    spec = config.validation
    groups = _group_key(mock.phenotypes, config.stratification.columns)
    allowed = {group.key for group in strata if group.count}
    if set(groups) != allowed:
        raise ContractError("Candidate strata differ from requested allocation")
    records = []
    means = []
    for group in strata:
        if not group.count:
            continue
        indices = groups[group.key]
        if len(indices) != group.count:
            raise ContractError("Candidate stratum counts differ from requested allocation")
        a = Dataset(
            real.phenotypes.loc[group.ids], real.microbiome.loc[group.ids], real.taxonomy
        )
        b = subset(mock, indices)
        record = compare(a, b, config, spec.max_features, spec.seed)
        privacy = evaluate(a, b, config, spec.max_samples, spec.seed)
        records.append({"values": list(group.key), "metrics": record, "privacy": privacy})
        means.append((a.microbiome.mean(axis=0).to_numpy(), b.microbiome.mean(axis=0).to_numpy()))
    keys = ("prevalence_mae", "phenotype_ks_mean", "phenotype_tv_mean", "phenotype_missing_mae")
    weighted = {}
    for key in keys:
        available = [
            (r["metrics"][key], r["metrics"]["candidate_n"])
            for r in records
            if r["metrics"][key] is not None
        ]
        weighted[key] = (
            float(np.average([v for v, _ in available], weights=[n for _, n in available]))
            if available
            else None
        )
    contrasts = None
    if len(means) > 1:
        a0, b0 = means[0]
        errors = [float(np.abs((a - a0) - (b - b0)).mean()) for a, b in means[1:]]
        contrasts = {
            "mean_abundance_contrast_mae": float(np.mean(errors)),
            "n_contrasts": len(errors),
        }
    return records, weighted, contrasts


def gate_results(report: dict, config: Config) -> list[dict]:
    """Fail closed for unavailable configured metrics; use worst conditional fidelity."""
    fidelity = report["conditional"] or [{"metrics": report["global"]}]
    privacy = report["privacy"]
    risks = [privacy, *(r["privacy"] for r in report["conditional"])]
    membership = privacy["membership"]
    attributes = privacy["attribute"]
    values = {
        "max_prevalence_mae": [r["metrics"]["prevalence_mae"] for r in fidelity],
        "max_phenotype_ks_mean": [r["metrics"]["phenotype_ks_mean"] for r in fidelity],
        "max_cross_modal_mae": [
            r["metrics"]["correlation_mae"]["cross_modal"] for r in fidelity
        ],
        "max_joint_match_rate": [risk["exact"]["joint_match_rate"] for risk in risks],
        "max_rare_match_rate": [risk["rare_categories"]["rare_match_rate"] for risk in risks],
        "min_joint_p05": [risk["candidate_to_source"]["joint"]["p05"] for risk in risks],
        "max_membership_auc": (
            [max(v["auc"], 1 - v["auc"]) for v in membership.values()] if membership else []
        ),
        "max_attribute_advantage": (
            [v["advantage"] for v in attributes.values()] if attributes else []
        ),
    }
    results = []
    for name, limit in config.validation.gates.items():
        metrics = values[name]
        minimum = name.startswith("min_")
        observed = None
        if metrics and all(v is not None and np.isfinite(v) for v in metrics):
            observed = min(metrics) if minimum else max(metrics)
        passed = observed is not None and (observed >= limit if minimum else observed <= limit)
        results.append({"name": name, "limit": limit, "observed": observed, "passed": passed})
    return results


def audit(
    real: Dataset,
    mock: Dataset,
    config: Config,
    strata: list[Stratum] | None = None,
    holdout: Dataset | None = None,
) -> dict:
    """Audit validated tables, using full source for disclosure and eligible for fidelity."""
    validate_audit(config.validation, config.columns)
    _check_schema(real, mock, config)
    if holdout is not None:
        if set(real.phenotypes.index).intersection(holdout.phenotypes.index):
            raise ContractError("Holdout identifiers overlap source")
        if set(mock.phenotypes.index).intersection(holdout.phenotypes.index):
            raise ContractError("Holdout identifiers overlap candidate")
        if set(real.microbiome.columns) != set(holdout.microbiome.columns):
            raise ContractError("Holdout microbial features differ from source")
        if (real.taxonomy is None) != (holdout.taxonomy is None):
            raise ContractError("Holdout taxonomy differs from source")
        if real.taxonomy is not None and not real.taxonomy.sort_index().equals(
            holdout.taxonomy.sort_index()
        ):
            raise ContractError("Holdout taxonomy differs from source")
    spec = config.validation
    if strata is None:
        strata = plan_strata(real, config, len(mock.phenotypes))
    records, weighted, contrasts = [], {}, None
    eligible = real
    training = real
    if config.stratification.columns:
        records, weighted, contrasts = _conditional(real, mock, config, strata)
        ids = pd.Index([identifier for group in strata if group.count for identifier in group.ids])
        eligible = Dataset(real.phenotypes.loc[ids], real.microbiome.loc[ids], real.taxonomy)
        training = eligible
        if config.stratification.pooling_strength:
            ids = pd.Index([identifier for group in strata for identifier in group.ids])
            training = Dataset(real.phenotypes.loc[ids], real.microbiome.loc[ids], real.taxonomy)
    report = {
        "version": __version__,
        "status": "local_diagnostics_not_anonymity_certification",
        "global": compare(eligible, mock, config, spec.max_features, spec.seed),
        "global_comparison": "raw_unweighted_eligible_source",
        "conditional": records,
        "allocation_weighted_conditional": weighted,
        "between_strata": contrasts,
        "privacy": evaluate(
            real,
            mock,
            config,
            spec.max_samples,
            spec.seed,
            holdout,
            spec.attribute_targets,
            training=training,
        ),
    }
    report["gates"] = gate_results(report, config)
    report["gates_passed"] = all(gate["passed"] for gate in report["gates"])
    return report


def write_report(report: dict, directory: Path) -> None:
    """Create a separate local-only directory, never overwrite a previous audit."""
    if directory.exists():
        raise ContractError("Audit directory already exists")
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    (directory / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def validate_directory(
    config: Config, candidate: Path, report_dir: Path, holdout_path: Path | None = None
) -> None:
    if report_dir.resolve().is_relative_to(candidate.resolve()):
        raise ContractError("Audit directory must be outside candidate directory")
    real = load_dataset(config)
    mock = load_candidate(config, candidate)
    holdout = load_holdout(config, holdout_path) if holdout_path is not None else None
    report = audit(real, mock, config, holdout=holdout)
    write_report(report, report_dir)
    if not report["gates_passed"]:
        raise ContractError("Configured validation gates failed; local report retained")
