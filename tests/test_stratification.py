from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from da_mocka.cli import main
from da_mocka.config import ContractError, Stratification, load_config
from da_mocka.generation import generate
from da_mocka.io import Dataset, load_dataset
from da_mocka.stratification import apportion, plan_strata, sample_strata


@pytest.fixture
def grouped(cohort):
    config = load_config(cohort / "config.yaml")
    data = load_dataset(config)
    p = data.phenotypes.copy()
    p["disease"] = ["A"] * 20 + ["B"] * 15 + ["control"] * 5
    p["sex"] = ["female"] * 15 + ["male"] * 5 + ["female"] * 5 + ["male"] * 15
    x = data.microbiome.copy()
    x.iloc[:, :] = 0.0
    x.iloc[:20, 0] = 1.0
    x.iloc[20:35, 1] = 1.0
    x.iloc[35:, 2] = 1.0
    p.to_csv(config.phenotypes, sep="\t")
    x.to_csv(config.microbiome, sep="\t")
    return config, Dataset(p, x, data.taxonomy)


def settings(config, **kwargs):
    defaults = {"columns": ("disease", "sex"), "min_stratum_size": 5}
    return replace(config, stratification=Stratification(**(defaults | kwargs)))


@pytest.mark.parametrize(
    ("allocation", "n", "expected"),
    [
        ("proportional", 80, [30, 10, 10, 20, 10]),
        ("balanced", 12, [3, 3, 2, 2, 2]),
        ("original", None, [15, 5, 5, 10, 5]),
    ],
)
def test_allocation(grouped, allocation, n, expected):
    config, data = grouped
    config = settings(config, allocation=allocation)
    plan = plan_strata(data, config, n)
    assert [g.count for g in plan] == expected
    assert [g.key for g in plan] == [
        ("A", "female"),
        ("A", "male"),
        ("B", "female"),
        ("B", "male"),
        ("control", "male"),
    ]


def test_apportion():
    assert apportion(2, [1, 1, 1]) == [1, 1, 0]
    assert apportion(0, [1, 2]) == [0, 0]
    with pytest.raises(ContractError):
        apportion(1, [0, 0])


def test_marginal(grouped):
    config, data = grouped
    config = settings(
        config,
        allocation="marginal",
        balance=("disease",),
        include={"disease": ("A", "B")},
    )
    plan = plan_strata(data, config, 120)
    assert [g.count for g in plan] == [45, 15, 20, 40]


def test_custom(grouped):
    config, data = grouped
    config = settings(config, allocation="custom", custom={("A", "female"): 7, ("B", "male"): 9})
    plan = plan_strata(data, config, None)
    assert [g.count for g in plan] == [7, 0, 0, 9, 0]
    with pytest.raises(ContractError, match="conflicts"):
        plan_strata(data, config, 17)
    config = settings(config, allocation="custom", custom={("control", "female"): 5})
    with pytest.raises(ContractError, match="unavailable"):
        plan_strata(data, config, None)


def test_sparse(grouped):
    config, data = grouped
    with pytest.raises(ContractError, match="below minimum"):
        plan_strata(data, settings(config, min_stratum_size=10), 100)
    config = settings(config, min_stratum_size=10, rare_strata="suppress")
    with pytest.warns(UserWarning, match="suppressed"):
        plan = plan_strata(data, config, 100)
    assert [g.key for g in plan] == [("A", "female"), ("B", "male")]
    assert [g.count for g in plan] == [60, 40]
    with pytest.raises(ContractError, match="No eligible"):
        plan_strata(data, settings(config, min_stratum_size=100, rare_strata="suppress"), 100)


def test_missing(grouped):
    config, data = grouped
    data.phenotypes.loc[data.phenotypes.index[0], "sex"] = pd.NA
    with pytest.raises(ContractError, match="Missing"):
        plan_strata(data, settings(config), 100)
    with pytest.warns(UserWarning, match="missing"):
        plan = plan_strata(data, settings(config, missing_strata="exclude"), 100)
    assert sum(len(g.ids) for g in plan) == 39


def test_outcome_selection_and_single_column(grouped):
    config, data = grouped
    config = settings(config, columns=("disease",), include={"disease": ("A", "B")})
    plan = plan_strata(data, config, 70)
    assert [g.key for g in plan] == [("A",), ("B",)]
    assert [g.count for g in plan] == [40, 30]


def test_unobserved_selection(grouped):
    config, data = grouped
    data.phenotypes["disease"] = "A"
    with pytest.raises(ContractError, match="not observed"):
        plan_strata(data, settings(config, include={"disease": ("A", "B")}), 10)
    with pytest.raises(ContractError, match="no eligible"):
        plan_strata(data, settings(config, include={"disease": ("B",)}), 10)


def test_conditional_generation(grouped, tmp_path):
    config, _ = grouped
    config = settings(config, allocation="balanced", include={"disease": ("A", "B")})
    output = tmp_path / "mock"
    generate(config, output, 80, seed=7)
    result = load_dataset(load_config(output / "config.yaml"))
    p, x = result.phenotypes, result.microbiome
    assert p.groupby(["disease", "sex"]).size().tolist() == [20, 20, 20, 20]
    assert (x.loc[p["disease"] == "A"].iloc[:, 0] == 1).all()
    assert (x.loc[p["disease"] == "B"].iloc[:, 1] == 1).all()
    assert np.allclose(x.sum(axis=1), 1)
    assert np.allclose(p["BMI"], p["weight"] / p["height"] ** 2)
    assert p.index.equals(x.index)
    other = tmp_path / "other"
    generate(config, other, 80, seed=7)
    assert all(f.read_bytes() == (other / f.name).read_bytes() for f in output.iterdir())


def test_pooling_excludes_suppressed_and_filtered(grouped):
    config, data = grouped
    config = settings(
        config,
        include={"disease": ("A", "B")},
        min_stratum_size=10,
        rare_strata="suppress",
        pooling_strength=1000,
        allocation="balanced",
    )
    with pytest.warns(UserWarning):
        plan = plan_strata(data, config, 500)
    p, x = sample_strata(data, config, plan, np.random.default_rng(42))
    assert set(zip(p["disease"], p["sex"], strict=True)) == {("A", "female"), ("B", "male")}
    assert (x.iloc[:, 2:] == 0).all().all()
    assert (x.loc[p["disease"] == "A"].iloc[:, 1] > 0).any()
    assert np.allclose(x.sum(axis=1), 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"columns": ("age",)},
        {"columns": ("disease", "disease")},
        {"allocation": "bad"},
        {"rare_strata": "bad"},
        {"missing_strata": "bad"},
        {"min_stratum_size": 0},
        {"pooling_strength": -1},
        {"columns": (), "allocation": "balanced"},
        {"include": {"age": ("A",)}},
        {"include": {"disease": ("unknown",)}},
        {"allocation": "marginal"},
        {"balance": ("disease",)},
        {"allocation": "custom"},
        {"allocation": "custom", "custom": {("A",): 1}},
        {"allocation": "custom", "custom": {("A", "male"): 0}},
    ],
)
def test_invalid_settings(grouped, kwargs):
    config, data = grouped
    with pytest.raises(ContractError):
        plan_strata(data, settings(config, **kwargs), 10)


def test_yaml_and_cli(grouped, tmp_path):
    config, _ = grouped
    path = config.phenotypes.parent / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["stratification"] = {
        "columns": ["disease", "sex"],
        "allocation": "custom",
        "min_stratum_size": 5,
        "custom": [{"values": ["A", "female"], "count": 12}],
    }
    path.write_text(yaml.safe_dump(raw))
    parsed = load_config(path)
    assert parsed.stratification.custom == {("A", "female"): 12}
    output = tmp_path / "custom"
    assert main(["generate", "--config", str(path), "--output", str(output)]) == 0
    assert len(load_dataset(load_config(output / "config.yaml")).phenotypes) == 12
    raw.pop("stratification")
    path.write_text(yaml.safe_dump(raw))
    args = [
        "generate",
        "--config",
        str(path),
        "--output",
        str(tmp_path / "cli"),
        "--n-samples",
        "20",
        "--stratify",
        "disease",
        "sex",
        "--include",
        "disease=[A,B]",
        "--balance",
        "disease",
        "--min-stratum-size",
        "5",
    ]
    assert main(args) == 0
    result = load_dataset(load_config(tmp_path / "cli" / "config.yaml"))
    assert result.phenotypes["disease"].value_counts().to_dict() == {"A": 10, "B": 10}
    assert main([*args, "--include", "bad"]) == 2


def test_original_unstratified(grouped):
    config, data = grouped
    plan = plan_strata(data, settings(config, columns=(), allocation="original"), None)
    assert plan[0].count == 40
    with pytest.raises(ContractError, match="required"):
        plan_strata(data, config, None)
