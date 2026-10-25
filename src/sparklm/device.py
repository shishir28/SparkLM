import torch


def select_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_model_to_device(model, device=None):
    selected_device = select_device() if device is None else torch.device(device)
    model.to(selected_device)
    assert_model_on_device(model, selected_device)
    return selected_device


def is_on_device(tensor_device, selected_device):
    if tensor_device.type != selected_device.type:
        return False
    return selected_device.index is None or tensor_device.index == selected_device.index


def assert_model_on_device(model, selected_device):
    for name, parameter in model.named_parameters():
        if not is_on_device(parameter.device, selected_device):
            raise ValueError(
                f"Parameter {name} is on {parameter.device}, "
                f"expected {selected_device}"
            )
    for name, buffer in model.named_buffers():
        if not is_on_device(buffer.device, selected_device):
            raise ValueError(
                f"Buffer {name} is on {buffer.device}, "
                f"expected {selected_device}"
            )
