"""With its defaults, the observed corridor-lattice planner reproduces the pre-L3 planner (50e9ad9)."""
import json
from pathlib import Path

import pytest

from regression_episodes import run

REF = Path(__file__).parent / "data" / "observed_ref_50e9ad9.json"


@pytest.mark.slow
def test_observed_defaults_reproduce_50e9ad9():
    ref = json.loads(REF.read_text())
    seeds = sorted({int(k.split("/")[0]) for k in ref})
    got = run(seeds=seeds)
    assert got == ref
