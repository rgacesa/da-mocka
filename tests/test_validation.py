import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from da_mocka.cli import main
from da_mocka.config import ContractError, Stratification, ValidationSettings, load_config
from da_mocka.generation import generate
from da_mocka.io import Dataset, load_dataset, read_table
from da_mocka.validation.audit import audit, load_candidate, validate_directory
from da_mocka.validation.privacy import distances, evaluate, exact_overlap, subset


def renamed(data, prefix):
    p, x = data.phenotypes.copy(), data.microbiome.copy()
    p.index = x.index = [f"{prefix}{i}" for i in range(len(p))]
    return Dataset(p, x, data.taxonomy)


def test_copied_data_detected(cohort):
    config = load_config(cohort / "config.yaml")
    data = load_dataset(config)
    candidate = renamed(data, "mock")
    config = replace(config, validation=ValidationSettings(gates={"max_joint_match_rate": 0.0}))
    report = audit(data, candidate, config)
    assert report["privacy"]["exact"]["joint_match_rate"] == 1
    assert report["privacy"]["candidate_to_source"]["joint"]["p05"] == 0
    assert report["global"]["prevalence_mae"] == 0
    assert report["global"]["phenotype_ks_mean"] == 0
    assert report["global"]["correlation_mae"]["cross_modal"] == 0
    assert not report["gates_passed"]
    assert report["privacy"]["membership"] is None
    with pytest.raises(ContractError, match="source identifiers"):
        audit(data, data, config)


def test_report_and_failed_gate(cohort, tmp_path):
    config = load_config(cohort / "config.yaml")
    output, report_dir = tmp_path / "mock", tmp_path / "audit"
    generate(config, output, 50, audit_dir=report_dir)
    report = json.loads((report_dir / "validation.json").read_text())
    assert report["status"] == "local_diagnostics_not_anonymity_certification"
    assert report["global"]["candidate_n"] == 50
    assert report["gates_passed"]
    assert report["privacy"]["membership"] is None
    assert not (output / "validation.json").exists()
    assert json.loads((output / "manifest.json").read_text())["audit_performed"]
    bad = replace(config, validation=ValidationSettings(gates={"max_membership_auc": 0.6}))
    failed, diagnostics = tmp_path / "failed", tmp_path / "failed_audit"
    with pytest.raises(ContractError, match="gates failed"):
        generate(bad, failed, 10, audit_dir=diagnostics)
    assert not failed.exists()
    report = json.loads((diagnostics / "validation.json").read_text())
    assert report["gates"][0]["observed"] is None
    assert not report["gates"][0]["passed"]
    with pytest.raises(ContractError, match="separate audit"):
        generate(bad, failed, 10)
    with pytest.raises(ContractError, match="outside"):
        generate(config, failed, 10, audit_dir=failed / "audit")


def test_holdout_attacks(cohort):
    config = load_config(cohort / "config.yaml")
    data = load_dataset(config)
    train = subset(data, np.arange(20))
    holdout = subset(data, np.arange(20, 40))
    mock = renamed(train, "mock")
    spec = ValidationSettings(max_samples=20, attribute_targets=("disease",))
    config = replace(config, validation=spec)
    report = audit(train, mock, config, holdout=holdout)
    privacy = report["privacy"]
    assert privacy["membership"]["joint"]["auc"] > 0.9
    assert privacy["attribute"]["disease"]["accuracy"] is not None
    assert privacy["distance_sample_sizes"]["holdout"] == 20
    with pytest.raises(ContractError, match="overlap source"):
        audit(train, mock, config, holdout=train)
    with pytest.raises(ContractError, match="requires an independent holdout"):
        audit(train, mock, config)


def test_distances_and_cap(cohort):
    config = load_config(cohort / "config.yaml")
    data = load_dataset(config)
    a = subset(data, np.arange(3))
    matrix = distances(a, a, config)
    for value in matrix.values():
        assert np.allclose(np.diag(value), 0)
        assert np.allclose(value, value.T)
        assert (value >= 0).all()
    result = evaluate(data, renamed(data, "mock"), config, 5, 42)
    assert result["distance_sample_sizes"] == {"source": 5, "candidate": 5}
    assert exact_overlap(data, renamed(data, "mock"), config)["joint_match_rate"] == 1


def test_stratified_audit(cohort, tmp_path):
    config = load_config(cohort / "config.yaml")
    config = replace(
        config,
        stratification=Stratification(
            columns=("disease",), allocation="balanced", min_stratum_size=1
        ),
        validation=ValidationSettings(max_features=8),
    )
    output, report_dir = tmp_path / "mock", tmp_path / "audit"
    generate(config, output, 60, audit_dir=report_dir)
    report = json.loads((report_dir / "validation.json").read_text())
    assert len(report["conditional"]) == 3
    assert report["between_strata"]["n_contrasts"] == 2
    assert all(r["metrics"]["candidate_n"] == 20 for r in report["conditional"])
    assert report["allocation_weighted_conditional"]["prevalence_mae"] >= 0


def test_candidate_contract_and_cli(cohort, tmp_path):
    path = cohort / "config.yaml"
    config = load_config(path)
    output = tmp_path / "mock"
    generate(config, output, 20)
    report = tmp_path / "report"
    args = [
        "validate",
        "--config",
        str(path),
        "--synthetic",
        str(output),
        "--report",
        str(report),
    ]
    assert main(args) == 0
    assert main(args) == 2
    with pytest.raises(ContractError, match="outside"):
        validate_directory(config, output, output / "report")
    # Candidate YAML is not authoritative.
    fake = yaml.safe_load((output / "config.yaml").read_text())
    fake["columns"]["age"]["min"] = 0
    (output / "config.yaml").write_text(yaml.safe_dump(fake))
    frame = read_table(output / "phenotypes.tsv")
    frame.loc[0, "age"] = "2"
    frame.to_csv(output / "phenotypes.tsv", sep="\t", index=False)
    with pytest.raises(ContractError, match="bounds"):
        load_candidate(config, output)


@pytest.mark.parametrize(
    "change",
    [
        {"max_samples": 1},
        {"max_features": 0},
        {"seed": -1},
        {"attribute_targets": ["age"]},
        {"gates": {"unknown": 1}},
        {"gates": {"max_membership_auc": 0.4}},
        {"gates": {"max_joint_match_rate": 2}},
        {"gates": []},
        {"extra": True},
    ],
)
def test_invalid_validation(cohort, change):
    path = cohort / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["validation"] = change
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContractError):
        load_config(path)


def test_valid_yaml(cohort):
    path = cohort / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["validation"] = {
        "max_samples": 50,
        "max_features": 8,
        "attribute_targets": ["disease"],
        "gates": {"max_membership_auc": 0.7},
    }
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).validation.attribute_targets == ("disease",)


def test_holdout_cli(cohort, tmp_path):
    config = load_config(cohort / "config.yaml")
    other = tmp_path / "holdout"
    assert main(["demo", "--output", str(other), "--n-samples", "20", "--seed", "99"]) == 0
    # The demo uses deterministic IDs, so explicitly give the holdout independent IDs.
    for name in ("phenotypes", "microbiome"):
        path = other / f"{name}.tsv"
        table = read_table(path)
        table["sample_id"] = pd.Series([f"HOLDOUT{i}" for i in range(len(table))])
        table.to_csv(path, sep="\t", index=False)
    output, diagnostics = tmp_path / "mock", tmp_path / "audit"
    generate(config, output, 20, audit_dir=diagnostics, holdout_path=other / "config.yaml")
    report = json.loads((diagnostics / "validation.json").read_text())
    assert report["privacy"]["membership"]["joint"]["auc"] is not None
