"""Shared color-transfer utilities: the bundled painting set, unique-RGB
point-cloud loading, and the stop-config dataclass used by
main.py and export_qualitative.py.

Point cloud: the UNIQUE RGB colors of an image, not raw pixels and not a
palette/cluster reduction -- duplicate pixels collapse into one point, and
their counts become that point's mass, via pixels_and_weights() below.
color_transfer/paintings/ ships 12 Monet paintings (public domain; Monet
died 1926) as the default dataset.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

DEFAULT_PAINTINGS_DIR = Path(__file__).parent / "paintings"


@dataclass
class StopCfg:
    """Duck-typed stop config accepted by sinkslot_alternating_triton's
    `stop` argument (mode/max_iter/check_every/tol attributes only)."""
    mode: str = "potential"
    max_iter: int = 8000
    check_every: int = 5
    tol: float = 1e-6


def list_images(root):
    valid_ext = {".jpg", ".jpeg", ".png"}
    return [os.path.join(root, n) for n in sorted(os.listdir(root))
            if os.path.splitext(n)[1].lower() in valid_ext]


def pixels_and_weights(path, device, dtype):
    """Unique RGB pixel values with summed weights -- duplicate pixels combined."""
    img = Image.open(path).convert("RGB")
    raw = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(-1, 3).to(device)
    total = raw.shape[0]
    uniq, counts = torch.unique(raw, dim=0, return_counts=True)
    pixels = uniq.to(dtype) / 255.0
    weights = counts.to(dtype) / total
    return pixels, weights
