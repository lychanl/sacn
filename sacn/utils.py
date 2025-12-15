import argparse
import pickle
from functools import partial, wraps
from time import time
from typing import Tuple, List, Union, Dict
import gym
import numpy as np
import torch as th


def normc_initializers():
    """Normalized column initializer from the OpenAI baselines"""
    def _initializer(shape, dtype=None, device=None):
        out = np.random.randn(*shape)
        out *= 1 / np.sqrt(np.square(out).sum(axis=1, keepdims=True))
        return th.as_tensor(out, dtype=dtype, device=device)
    def _bias_initializer(shape, dtype=None, device=None):
        return th.zeros(shape, dtype=dtype, device=device)
    return _initializer, _bias_initializer


def glorot_initializers():
    def _initializer(shape, dtype=None, device=None):
        assert len(shape) == 2
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        out = np.random.uniform(-limit, limit, shape)
        return th.as_tensor(out, dtype=dtype, device=device)
    def _bias_initializer(shape, dtype=None, device=None):
        return th.zeros(shape, dtype=dtype, device=device)
    return _initializer, _bias_initializer


class _DTLinearIncrease(argparse.Action):
    def __init__(self, *args, dt, **kwargs) -> None:
        self.dt = dt
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values * self.dt)

def DTLinearIncrease(dt):
    return partial(_DTLinearIncrease, dt=dt)


class _DTRootIncrease(argparse.Action):
    def __init__(self, *args, dt, **kwargs) -> None:
        self.dt = dt
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values ** (1 / self.dt))

def DTRootIncrease(dt):
    return partial(_DTRootIncrease, dt=dt)

