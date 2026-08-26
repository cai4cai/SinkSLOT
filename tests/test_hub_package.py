"""Smoke test for the installed `sinkslot` package layout.

Checks that a plain `pip install sinkslot` -- no extras, no Triton, no
FlashSinkhorn -- yields a working public API. Anything needing the `bench`
extra belongs in torch-ext/sinkslot/testing/ instead.
"""

import subprocess
import sys

import sinkslot


def test_public_api_is_importable():
    assert sinkslot.SamplesLoss.__name__ == "SamplesLoss"


def test_importing_sinkslot_does_not_pull_in_flash_sinkhorn():
    """In a fresh interpreter, not this one.

    Checking sys.modules in-process would test the pytest session rather than
    the package: once any other test in the run imports flash_sinkhorn (the
    parity tests do, when the bench extra is installed), it is present in
    sys.modules no matter what importing sinkslot does. A subprocess is the
    only way to ask the question this test is actually about.

    Only flash_sinkhorn is asserted absent. triton is not: sinkslot probes for
    it at import to decide between the fused and pure-torch backends, so on a
    host that has it, importing sinkslot importing triton is correct. That
    Triton is optional rather than required is covered by the CI matrix
    installing neither and running this same suite.
    """
    code = "import sinkslot, sys; print(int('flash_sinkhorn' in sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "0", "importing sinkslot pulled in flash_sinkhorn"
