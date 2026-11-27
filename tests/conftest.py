from pathlib import Path

import pytest

from da_mocka.demo import make_demo


@pytest.fixture
def cohort(tmp_path: Path) -> Path:
    output = tmp_path / "cohort"
    make_demo(output, n_samples=40, seed=7)
    return output
