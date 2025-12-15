from typing import List, Callable, Tuple, Union

import torch as th

from utils import normc_initializers


def build_cnn_network(input_size: Union[Tuple[int, int], Tuple[int, int, int]],
                      filters: tuple = (32, 64, 64), kernels: tuple = (8, 4, 3),
                      strides: tuple = ((4, 4), (2, 2), (1, 1)), activation: str = 'relu',
                      initializer: Callable = None) \
        -> th.nn.Module:
    """Builds predefined CNN neural network

    Args:
        filters: tuple with filters to be used
        kernels: tuple with kernels to be used
        strides: tuple with strides to be used
        activation: activation function to be used in each layer
        initializer: callable to the weight initializer function

    Returns:
        created network
    """
    assert len(filters) == len(kernels) == len(strides), "Layers specifications must have the same lengths"
    raise NotImplementedError("cnn network is not yet implemented")
