from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from da_mocka.cli import main
from da_mocka.config import Column, ContractError, EngineSettings, Stratification, load_config
from da_mocka.engines.base import fit_engine
from da_mocka.engines.joint_latent import JointLatentEngine
from da_mocka.engines.latent import Encoder, LatentFactors
from da_mocka.generation import generate
from da_mocka.io import Dataset, load_dataset


def joint_config(config, **kwargs):
    return replace(config, engine=EngineSettings(name="joint_latent", **kwargs))


def test_latent_factors():
    rng = np.random.default_rng(42)
    a = rng.normal(size=500)
    matrix = np.column_stack([a, a + rng.normal(0, 0.1, 500), np.ones(500)])
    model = LatentFactors.fit(matrix, 3, 0.1)
    draws = model.sample(5000, rng)
    assert np.corrcoef(draws.T)[0, 1] > 0.8
    assert np.allclose(draws.std(0), 1, atol=0.05)
    assert np.isfinite(draws).all()


def test_randomized_factor_path():
    matrix = np.random.default_rng(5).normal(size=(100, 200))
    factors = LatentFactors.fit(matrix, 5, 0.2)
    assert factors.loadings.shape == (200, 5)
    assert np.allclose((factors.loadings**2).sum(1) + factors.residual**2, 1)
    assert factors.sample(4, np.random.default_rng(6)).shape == (4, 200)


def test_encoder_edges():
    encoder = Encoder()
    observed = np.array([True, True, False])
    quantile = encoder.quantile(np.array([1.0, 3.0, np.nan]), observed, discrete=True)
    none = encoder.quantile(np.zeros(3), ~np.ones(3, dtype=bool))
    always = encoder.gate(observed, observed)
    never = encoder.gate(~observed, observed)
    z = np.zeros((10, len(encoder.columns)))
    assert set(quantile.decode(z)) <= {1, 3}
    assert np.isnan(none.decode(z)).all()
    assert always.decode(z).all()
    assert not never.decode(z).any()


@pytest.fixture
def associated(cohort):
    config = load_config(cohort / "config.yaml")
    rng = np.random.default_rng(123)
    n = 800
    a = rng.normal(size=n)
    missing = a > 0.4
    p = pd.DataFrame(
        {
            "trait": 50 + 10 * a,
            "related": 50 + 8 * a + rng.normal(0, 1, n),
            "marker": np.where(missing, np.nan, 2.0),
            "marker2": np.where(missing, np.nan, 3.0),
            "category": np.where(a > 0, "high", "low"),
        }
    )
    x = np.column_stack([np.exp(a), np.exp(-a), np.where(a > 0, 0.4, 0), np.where(a > 0, 0.6, 0)])
    x /= x.sum(1, keepdims=True)
    config = replace(
        config,
        bmi=None,
        columns={
            "trait": Column("continuous", minimum=0, maximum=100),
            "related": Column("continuous", minimum=0, maximum=100),
            "marker": Column("continuous", True, 0, 10),
            "marker2": Column("continuous", True, 0, 10),
            "category": Column("categorical", categories=("low", "high", "unseen")),
        },
    )
    return config, Dataset(p, pd.DataFrame(x, columns=["a", "b", "c", "d"]), None)


def test_joint_preserves_associations(associated):
    config, data = associated
    joint = fit_engine(data, joint_config(config, rank=16, shrinkage=0.02))
    baseline = fit_engine(data, config)
    p, x = joint.sample(4000, np.random.default_rng(10))
    bp, bx = baseline.sample(4000, np.random.default_rng(10))
    assert p["trait"].corr(p["related"]) > 0.75
    assert abs(bp["trait"].corr(bp["related"])) < 0.1
    assert p["trait"].corr(np.log(x["a"] / x["b"])) > 0.75
    assert abs(bp["trait"].corr(np.log(bx["a"] / bx["b"]))) < 0.1
    assert (x["c"] > 0).corr(x["d"] > 0) > 0.6
    assert p["marker"].isna().corr(p["marker2"].isna()) > 0.6
    assert p.loc[p["category"] == "high", "trait"].mean() > 55
    assert "unseen" not in p["category"].values
    assert np.allclose(x.sum(axis=1), 1)


def test_joint_roundtrip(cohort, tmp_path):
    config = joint_config(load_config(cohort / "config.yaml"), rank=8, batch_size=25)
    a, b = tmp_path / "a", tmp_path / "b"
    generate(config, a, 75, seed=7)
    generate(config, b, 75, seed=7)
    result = load_dataset(load_config(a / "config.yaml"))
    p, x = result.phenotypes, result.microbiome
    assert x.shape == (75, 200)
    assert p.index.equals(x.index)
    assert np.allclose(p["BMI"], p["weight"] / p["height"] ** 2)
    assert (x >= 0).all().all()
    assert np.allclose(x.sum(axis=1), 1)
    assert all(f.read_bytes() == (b / f.name).read_bytes() for f in a.iterdir())


def test_joint_stratified(cohort, tmp_path):
    config = joint_config(load_config(cohort / "config.yaml"), rank=4)
    config = replace(
        config,
        stratification=Stratification(
            columns=("disease",), allocation="balanced", min_stratum_size=1
        ),
    )
    output = tmp_path / "stratified"
    generate(config, output, 60)
    p = load_dataset(load_config(output / "config.yaml")).phenotypes
    assert p["disease"].value_counts().max() - p["disease"].value_counts().min() <= 1


def test_all_missing_constant_and_absent(associated):
    config, data = associated
    data.phenotypes["marker"] = np.nan
    data.microbiome.loc[:, "a"] = 1.0
    data.microbiome.loc[:, ["b", "c", "d"]] = 0.0
    model = JointLatentEngine.fit(data, joint_config(config))
    p, x = model.sample(30, np.random.default_rng(1))
    assert p["marker"].isna().all()
    assert (x["a"] == 1).all()
    assert (x[["b", "c", "d"]] == 0).all().all()
    with pytest.raises(ContractError, match="Positive"):
        model.sample(0, np.random.default_rng(2))


def test_joint_rejection(associated, monkeypatch):
    config, data = associated
    model = JointLatentEngine.fit(data, joint_config(config))
    for gate in model.presence:
        monkeypatch.setattr(gate, "probability", 0.0)
    with pytest.raises(ContractError, match="joint biological"):
        model.sample(2, np.random.default_rng(1))


@pytest.mark.parametrize(
    "change",
    [
        {"name": "unknown"},
        {"rank": 0},
        {"rank": True},
        {"shrinkage": -0.1},
        {"shrinkage": 1.1},
        {"batch_size": 0},
        {"extra": 1},
    ],
)
def test_invalid_engine(cohort, change):
    path = cohort / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["engine"] = {"name": "joint_latent"} | change
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContractError):
        load_config(path)


def test_cli_and_yaml(cohort, tmp_path):
    path = cohort / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["engine"] = {"name": "joint_latent", "rank": 4, "batch_size": 10}
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).engine.rank == 4
    assert (
        main(
            [
                "generate",
                "--config",
                str(path),
                "--output",
                str(tmp_path / "mock"),
                "--n-samples",
                "20",
                "--engine",
                "joint_latent",
                "--latent-rank",
                "5",
            ]
        )
        == 0
    )
