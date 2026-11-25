"""bench.pot_exact: POT's network simplex on the benchmark instances (CPU)."""

import pytest
import torch

pytest.importorskip("ot")

from sinkslot.bench.bench_forward import _sample_problem, compute_exact_ot_reference  # noqa: E402
from sinkslot.bench.pot_exact import run  # noqa: E402


def test_dense_and_lazy_reach_the_exact_reference():
    n, d, seed = 200, 2, 0
    rows = run([("half_moon", d)], n, [seed], ["dense", "lazy"], torch.device("cpu"))
    x, y, a, b = _sample_problem(n, n, d, torch.device("cpu"), "half_moon", seed)
    ref = compute_exact_ot_reference(x.double(), y.double(), a.double() / a.double().sum(),
                                     b.double() / b.double().sum()).cost
    assert [r["kind"] for r in rows] == ["dense", "lazy"]
    for r in rows:
        assert r["converged"] and r["method"] == "pot_exact"
        assert r["cost"] == pytest.approx(ref, rel=1e-10)
        assert r["total_ms"] == pytest.approx(r["setup_ms"] + r["mean_ms"])
    assert rows[1]["setup_ms"] == 0.0
