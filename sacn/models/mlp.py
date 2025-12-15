from typing import List, Callable, Tuple, Union

import torch as th

from utils import normc_initializers, glorot_initializers


def build_mlp_network(input_size: int, layers_sizes: Tuple[int] = (256, 256), activation: str = 'tanh',
                      initializer: Union[Callable, str] = None, device: str = None) \
        -> th.nn.Module:
    """Builds Multilayer Perceptron neural network

    Args:
        layers_sizes: sizes of hidden layers
        activation: activation function name
        initializer: callable to the weight initializer function

    Returns:
        created network
    """
    if activation is None:
        activation = 'tanh'
    if initializer is None and activation == 'tanh':
        initializer, bias_initializer = normc_initializers()
    elif isinstance(initializer, str):
        initializer, bias_initializer = {
            'normc': normc_initializers(),
            'glorot': glorot_initializers(),
            'default': (None, None)
        }[initializer]
    layers = []
    for i, out_size in enumerate(layers_sizes):
        in_size = input_size if i == 0 else layers_sizes[i - 1]
        layer = th.nn.Linear(in_size, out_size, device=device)
        if initializer is not None:
            with th.no_grad():
                layer.weight.copy_(initializer(layer.weight.shape, layer.weight.dtype, layer.weight.device))
                layer.bias.copy_(bias_initializer(layer.bias.shape, layer.bias.dtype, layer.bias.device))
        layers.append(layer)
        if i + 1 != len(layers_sizes):
            if activation == 'tanh':
                layers.append(th.nn.Tanh())
            elif activation == 'relu':
                layers.append(th.nn.ReLU())
            else:
                raise ValueError()
    model = th.nn.Sequential(
        *layers
    )
    return model
