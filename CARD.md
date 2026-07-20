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

## Usage

```python
from kernels import get_kernel

flash_sinkhorn = get_kernel("ot-triton-lab/flash-sinkhorn", version=1)
loss = flash_sinkhorn.SamplesLoss(loss="sinkhorn", blur=0.1, debias=True)
```

The input tensors must be CUDA tensors and require PyTorch and Triton.
