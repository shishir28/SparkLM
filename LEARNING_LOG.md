# Learning Log

## Entry 1 — Phase 0 start

Date (UTC): 2026-08-26

### Goal

Set up the environment, verify reproducibility, and document project rules before writing model code.

### What I did

- Ran a smoke test for Python, PyTorch, CUDA, GPU detection, and tensor placement.
- Added a determinism check using `torch.manual_seed(42)`.
- Created the Phase 0 documentation files under `essentials/`.

### Observations

- Python: 3.12.3
- PyTorch: 2.13.0+cu130
- CUDA: available
- GPU: NVIDIA GB10
- RAM and disk space are sufficient for the early phases.
- PyTorch warns that NumPy is missing from the virtual environment.

### Conclusion

The environment is usable for Phase 0 and ready for the tokenizer work in Phase 1.
