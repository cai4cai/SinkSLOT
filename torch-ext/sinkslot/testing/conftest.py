"""Shared fixtures for the test suite.

`fused` runs a test once per fused backend in `_backends.DEVICE_BACKENDS`
(Triton on CUDA, Metal on MPS), as a `(device, backend)` pair, and skips the
backends this machine can't run. A test that needs the fused kernels takes
this fixture instead of hardcoding `.cuda()`, so it covers every backend
without knowing which ones exist.

`gpu` is the same idea for tests that only need a GPU tensor, not a fused
backend (e.g. the plain-torch plan builders): once per GPU device type,
skipped where that device isn't present.
"""

import pytest
import torch

from sinkslot._backends import DEVICE_BACKENDS, fused_backend


_PRESENT = {"cuda": torch.cuda.is_available, "mps": torch.backends.mps.is_available}


def _runnable(device: str, backend: str) -> bool:
    # fused_backend() only checks the toolchain: Triton can be importable on a
    # host with no CUDA device, so the device itself is checked here.
    return _PRESENT[device]() and fused_backend(device) == backend


FUSED_PARAMS = [
    pytest.param(
        (device, backend), id=backend,
        marks=pytest.mark.skipif(not _runnable(device, backend),
                                 reason=f"needs {backend} on {device}"),
    )
    for device, backend in DEVICE_BACKENDS.items()
]


@pytest.fixture(params=FUSED_PARAMS)
def fused(request):
    """`(device, backend)` of one fused backend runnable on this machine."""
    return request.param


GPU_PARAMS = [
    pytest.param(device, id=device,
                 marks=pytest.mark.skipif(not _PRESENT[device](), reason=f"needs {device}"))
    for device in DEVICE_BACKENDS
]


@pytest.fixture(params=GPU_PARAMS)
def gpu(request):
    """One GPU device type present on this machine ("cuda" or "mps")."""
    return request.param
