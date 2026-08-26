import sys

import torch


def main():
    print("=== SparkLM Environment Smoke Test ===")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        for i in range(device_count):
            print(f"Device {i} name: {torch.cuda.get_device_name(i)}")
        device = torch.device("cuda:0")
    else:
        print("Device name: CPU")
        device = torch.device("cpu")

    print("\n--- Tensor-Device Status ---")
    try:
        # Create a small tensor directly on the selected device.
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print(f"Successfully created a tensor on: {x.device}")
        print(f"Tensor value: {x}")

        print("\n--- Determinism Check ---")
        seed = 42

        # Resetting the random-number generator to the same seed should
        # reproduce the same sequence of values.
        torch.manual_seed(seed)
        first = torch.rand(2, 3, device=device)

        torch.manual_seed(seed)
        second = torch.rand(2, 3, device=device)

        deterministic_match = torch.equal(first, second)
        print(f"Seed: {seed}")
        print(f"First tensor:\n{first}")
        print(f"Second tensor:\n{second}")
        print(f"Deterministic match: {deterministic_match}")

        if not deterministic_match:
            print("\nSmoke test FAILED!")
            return 1
    except Exception as error:
        print(f"\nSmoke test error: {error}")
        print("Smoke test FAILED!")
        return 1

    print("\nSmoke test PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
