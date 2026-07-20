---
library_name: kernels
license: MIT
tags:
  - triton
  - optimal-transport
  - sinkhorn
---

# FlashSinkhorn

FlashSinkhorn provides streaming entropic optimal-transport kernels for PyTorch
and Triton. It computes squared-Euclidean Sinkhorn losses without materializing
the full cost matrix.

Source code, documentation, and issues live in the GitHub repository:
[ot-triton-lab/flash-sinkhorn](https://github.com/ot-triton-lab/flash-sinkhorn).

## Usage

```python
from kernels import get_kernel

flash_sinkhorn = get_kernel("yexf308/flash-sinkhorn", version=1)
loss = flash_sinkhorn.SamplesLoss(loss="sinkhorn", blur=0.1, debias=True)
```

The input tensors must be CUDA tensors and require PyTorch and Triton.

## Links

- **GitHub repository:** https://github.com/ot-triton-lab/flash-sinkhorn
- **Paper:** https://arxiv.org/abs/2602.03067 (FlashSinkhorn, ICML 2026 Oral)
