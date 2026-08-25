"""Smoke test for the installed `sinkslot` package layout.

Deliberately imports nothing but `sinkslot` itself: this is the check that a
plain `pip install sinkslot` -- no extras, no Triton, no FlashSinkhorn --
yields a working public API. Anything needing the `bench` extra belongs in
torch-ext/sinkslot/testing/ instead.
"""

import sinkslot


def test_public_api_is_importable():
    assert sinkslot.SamplesLoss.__name__ == "SamplesLoss"


def test_package_does_not_require_flash_sinkhorn():
    import sys

    assert "flash_sinkhorn" not in sys.modules
