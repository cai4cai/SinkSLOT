"""Smoke tests for the noarch Hugging Face Kernels source layout."""

import flash_sinkhorn


def test_public_api_is_importable():
    assert flash_sinkhorn.SamplesLoss.__name__ == "SamplesLoss"
