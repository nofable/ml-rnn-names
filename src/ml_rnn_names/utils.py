from pathlib import Path
import torch


def get_project_root():
    """Get project root (where pyproject.toml is)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


def torch_device_setup():
    """Check if CUDA is available and return device"""
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    torch.set_default_device(device)
    print(f"Using device = {torch.get_default_device()}")
    return device
