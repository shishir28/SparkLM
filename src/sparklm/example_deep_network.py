from torch import nn

from sparklm.layers import GELU


class ExampleDeepNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(layer_sizes[index], layer_sizes[index + 1]),
                GELU(),
            )
            for index in range(len(layer_sizes) - 1)
        )

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
