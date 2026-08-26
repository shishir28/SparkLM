# Smoke Test

## Purpose

Verify the local Python, PyTorch, CUDA, and GPU setup, and confirm deterministic random-number generation.

## What the test checks

- Python version
- PyTorch version
- CUDA availability
- CUDA version
- GPU count and GPU name
- tensor creation on the selected device
- deterministic output from `torch.manual_seed(seed)`

## Observed result

- Python: 3.12.3
- PyTorch: 2.13.0+cu130
- CUDA available: True
- CUDA version: 13.0
- GPU count: 1
- GPU: NVIDIA GB10
- Tensor created successfully on `cuda:0`
- Deterministic match: True
- Exit status: 0

## Notes

- PyTorch printed a NumPy warning because NumPy is not installed yet.
- The smoke test should stay small and fast.
