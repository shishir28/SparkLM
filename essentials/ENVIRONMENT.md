# Environment

Date (UTC): 2026-08-26

## Runtime

- Python: 3.12.3
- PyTorch: 2.13.0+cu130
- CUDA available: True
- CUDA version: 13.0
- Device count: 1
- GPU: NVIDIA GB10
- Platform: Linux-6.17.0-1014-nvidia-aarch64-with-glibc2.39

## Storage

- Filesystem: `/dev/nvme0n1p2`
- Total: 3.7T
- Used: 892G
- Available: 2.7T
- Mount point: `/`

## Memory

- RAM: 121Gi total, 95Gi available
- Swap: 15Gi total, 15Gi available

## Notes

- The smoke test ran successfully on CUDA.
- PyTorch emitted a NumPy warning because NumPy is not installed in the virtual environment yet.
- Keep this file updated if the environment changes.
