from networks.layers import NoisyLayer
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, layers):
        super(Network, self).__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()