import pytest
import torch

from sparklm.device import (
    assert_model_on_device,
    is_on_device,
    move_model_to_device,
    select_device,
)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


def make_model_with_buffer():
    model = torch.nn.Linear(2, 2)
    model.register_buffer("mask", torch.ones(2))
    return model


def test_select_device_prefers_first_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert select_device() == torch.device("cuda:0")


def test_select_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert select_device() == torch.device("cpu")


@pytest.mark.parametrize(
    ("tensor_device", "selected_device", "expected"),
    [
        ("cpu", "cpu", True),
        ("cuda:0", "cuda:0", True),
        ("cuda:0", "cuda", True),
        ("cuda:1", "cuda", True),
        ("cuda:1", "cuda:0", False),
        ("cpu", "cuda", False),
        ("cuda:0", "cpu", False),
    ],
)
def test_is_on_device(tensor_device, selected_device, expected):
    assert (
        is_on_device(torch.device(tensor_device), torch.device(selected_device))
        is expected
    )


def test_move_model_to_cpu_returns_cpu_device():
    model = make_model_with_buffer()

    selected_device = move_model_to_device(model, "cpu")

    assert selected_device == torch.device("cpu")
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(buffer.device.type == "cpu" for buffer in model.buffers())


def test_move_model_uses_selected_device_by_default(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = make_model_with_buffer()

    assert move_model_to_device(model) == torch.device("cpu")


@requires_cuda
@pytest.mark.parametrize("device", ["cuda", "cuda:0", torch.device("cuda")])
def test_move_model_to_cuda_accepts_indexed_and_unindexed_devices(device):
    model = make_model_with_buffer()

    move_model_to_device(model, device)

    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    assert all(buffer.device.type == "cuda" for buffer in model.buffers())


def test_assert_model_on_device_accepts_matching_model():
    assert_model_on_device(make_model_with_buffer(), torch.device("cpu"))


def test_assert_model_on_device_rejects_mismatched_parameter():
    model = torch.nn.Linear(2, 2)

    with pytest.raises(ValueError, match="Parameter weight"):
        assert_model_on_device(model, torch.device("meta"))


def test_assert_model_on_device_rejects_mismatched_buffer():
    model = torch.nn.Module()
    model.register_buffer("mask", torch.ones(2))

    with pytest.raises(ValueError, match="Buffer mask"):
        assert_model_on_device(model, torch.device("meta"))


@requires_cuda
def test_assert_model_on_device_rejects_cpu_model_for_unindexed_cuda():
    model = torch.nn.Linear(2, 2)

    with pytest.raises(ValueError, match="expected cuda"):
        assert_model_on_device(model, torch.device("cuda"))
