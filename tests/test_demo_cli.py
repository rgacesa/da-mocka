import numpy as np
import pytest

from da_mocka.cli import main
from da_mocka.config import ContractError, load_config
from da_mocka.demo import make_demo
from da_mocka.io import load_dataset


def test_demo_reproducibility(cohort, tmp_path):
    other = tmp_path / "other"
    make_demo(other, n_samples=40, seed=7)
    for path in cohort.iterdir():
        assert path.read_bytes() == (other / path.name).read_bytes()
    result = load_dataset(load_config(cohort / "config.yaml"))
    p = result.phenotypes
    assert np.allclose(p["BMI"], p["weight"] / p["height"] ** 2)
    assert 0.5 < (result.microbiome == 0).to_numpy().mean() < 0.9
    assert p["marker"].isna().any()


def test_no_overwrite(cohort):
    with pytest.raises(ContractError, match="already exists"):
        make_demo(cohort)


@pytest.mark.parametrize(("n", "seed"), [(0, 1), (10, -1)])
def test_bad_demo_arguments(tmp_path, n, seed):
    with pytest.raises(ContractError):
        make_demo(tmp_path / "bad", n, seed)


def test_cli(tmp_path, capsys):
    path = tmp_path / "demo"
    assert main(["demo", "--output", str(path), "--n-samples", "3"]) == 0
    assert main(["check-input", "--config", str(path / "config.yaml")]) == 0
    assert "3 samples" in capsys.readouterr().out
    assert main(["demo", "--output", str(path)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert main(["check-input", "--config", str(path / "missing.yaml")]) == 2


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "0.0.5"
