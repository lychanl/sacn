from abc import ABC, abstractmethod
from argparse import ArgumentError
from algos.common.automodel import AutoModel, AutoModelComponent
from pathlib import Path
from typing import Any, Iterable, Tuple, Union, List, Optional, Dict

import gym
import numpy as np
import torch as th
from distributions import DISTRIBUTIONS

import utils
from models.cnn import build_cnn_network
from models.mlp import build_mlp_network
from replay_buffer import MultiReplayBuffer, BufferFieldSpec, ReplayBuffer
from utils import DTLinearIncrease, DTRootIncrease


class BaseModel(AutoModelComponent, th.nn.Module):
    @staticmethod
    def get_args(preproc_params, component):
        return {
            'layers': (int, [256, 256], {'nargs': '*'}),
            'activation': (str, 'tanh'),
            'initializer': (str, None)
        }

    @staticmethod
    def preprocess_args(args, preproc_params):
        pass

    def __init__(
            self, observation_space: gym.spaces.Space, layers: Optional[Tuple[int]], output_dim: int, extra_models: Tuple[Iterable[int]] = (),
            activation=None, *args, initializer=None, device=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.initializer = initializer

        self.optimizer_cls = th.optim.Adam
        self.extra_models = extra_models

        if not isinstance(observation_space, gym.spaces.Discrete):
            self.model = self._build_layers(observation_space.shape, layers, output_dim, activation, device=device)
            self.extras_models = [
                self._build_layers(observation_space.shape, elayers, outs, activation, device=device) for *elayers, outs in extra_models
            ]
            self._input_shape_len = len(observation_space.shape)
            self._forward = self._nn_forward
            self._extras_forward = self._extras_nn_forward
        else:
            self.array = th.nn.Parameter(th.zeros((observation_space.n, output_dim), device=device, dtype=th.float32))
            self.extras_arrays = th.nn.ParameterList([th.zeros((observation_space.n, out), device=device, dtype=th.float32) for out in extra_models])
            self._forward = self._arr_forward
            self._extras_forward = self._extras_arr_forward

        self._output_dim = output_dim

        self.optimizer = None

    def _build_layers(
        self, input_shape: int, layers: Optional[Tuple[int]], output_dim: int, activation=None, device=None
    ) -> th.nn.Module:
        if len(input_shape) > 1:
            # build_cnn_network()
            raise NotImplementedError
        
        layers_sizes = list(layers) + [output_dim]

        return build_mlp_network(
            input_size=input_shape[-1], layers_sizes=layers_sizes,
            initializer=self.initializer, activation=activation, device=device)

    def _nn_forward(self, input: th.Tensor) -> th.Tensor:
        return self.model(input)

    def _extras_nn_forward(self, input: th.Tensor) -> Tuple[th.Tensor]:
        return tuple(model(input) for model in self.extra_models)

    def _arr_forward(self, input: th.Tensor) -> th.Tensor:
        return self.array[input]

    def _extras_arr_forward(self, input: th.Tensor) -> Tuple[th.Tensor]:
        tuple([extra_arr[input] for extra_arr in self.extras_arrays])

    def call(self, input) -> th.Tensor:
        return self._forward(input)

    def optimize(self, **loss_kwargs) -> th.Tensor:
        loss = self._loss(**loss_kwargs)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def init_optimizer(self, *args, **kwargs) -> th.optim.Optimizer:
        self.optimizer = self.optimizer_cls(self.parameters(), *args, **kwargs)
        return self.optimizer

    def _loss(self) -> th.Tensor:
        raise NotImplementedError


class BaseActor(BaseModel):
    @staticmethod
    def get_args(preproc_params, component):
        args = BaseModel.get_args(preproc_params, component)
        args['b'] = (float, 3)
        args['ignore_above_b'] = (bool, False, {'action': 'store_true'})
        args['ignore_outside_b'] = (bool, False, {'action': 'store_true'})
        args['no_truncate'] = (bool, False, {'action': 'store_true'})
        args['clip_b'] = (bool, False, {'action': 'store_true'})
        return args

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, th_time_step: th.Tensor, *args, no_truncate: bool = False,
                 ignore_above_b: bool = False, ignore_outside_b: bool = False, b: float = 3, clip_b: bool = False, additional_outputs=0,
                 **kwargs):
        """Base abstract Actor class

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layers sizes, eg: for neural network with two layers with 10 and 20 hidden units
                pass: [10, 20]
            beta_penalty: penalty coefficient. In discrete case, BaseActor is penalized for too executing too
                confident actions (no exploration), in the continuous case it is penalized for making actions
                that are out of the allowed bounds
            th_time_step: time step as Torch variable, required for TensorBoard summaries
        """
        if type(actions_space) == gym.spaces.discrete.Discrete:
            actions_dim = actions_space.n
        else:
            actions_dim = actions_space.shape[0]

        super().__init__(observations_space, layers, actions_dim + additional_outputs, *args, **kwargs)

        self.obs_shape = observations_space.shape
        self.actions_dim = actions_dim
        self.beta_penalty = beta_penalty
        self._th_time_step = th_time_step

        self._clip_b = clip_b
        self._truncate = not no_truncate and not ignore_above_b and not ignore_outside_b and not clip_b
        self._ignore_above_b = ignore_above_b
        self._ignore_outside_b = ignore_outside_b
        self._b = b

        self.register_method('policies', self.policy, {'observations': 'obs', 'actions': 'actions'})
        self.register_method('actions', self.act_deterministic, {'observations': 'obs'})
        self.register_method(
            'density', self._calculate_density, {
                'policies': 'actor.policies',
                'old_policies': 'policies',
                'mask': 'base.mask'
            }
        )

        ignore_b_mask_args = {'mask': 'self.ignore_above_b_mask'} \
            if self._ignore_above_b else {'mask': 'self.ignore_outside_b_mask'} \
            if self._ignore_outside_b else {}
        self.register_method(
            'ignore_above_b_mask', self._ignore_above_b_mask, {
                'density': 'actor.density',
                'priorities': 'priorities',
        })
        self.register_method(
            'ignore_outside_b_mask', self._ignore_outside_b_mask, {
                'density': 'actor.density',
                'priorities': 'priorities',
        })
        self.register_method(
            'sample_weights', self._calculate_truncated_weights, {
                'density': 'actor.density',
                'priorities': 'priorities',
                **ignore_b_mask_args
            }
        )
        self.register_method(
            'truncated_density', self._calculate_truncated_density, {
                'density': 'actor.density',
                **ignore_b_mask_args
            }
        )

        self.register_method('optimize', self.optimize, {
            'observations': 'base.first_obs',
            'actions': 'base.first_actions',
            'd': 'base.weighted_td'
        })
        self.targets = ['optimize']

    def policy(self, observations, actions) -> th.Tensor:
        return self.prob(observations, actions)[0]

    @property
    @abstractmethod
    def action_dtype(self):
        """Returns data type of the BaseActor's actions (Torch)"""

    @property
    @abstractmethod
    def action_dtype_np(self):
        """Returns data type of the BaseActor's actions (Numpy)"""

    @abstractmethod
    def prob(self, observations: np.array, actions: np.array) -> th.Tensor:
        """Computes probabilities (or probability densities in continuous case) and logarithms of it

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            actions: batch [batch_size, actions_dim] of actions vectors

        Returns:
             Tensor [batch_size, actions_dim, 2] with computed probabilities (densities) and logarithms
        """

    @abstractmethod
    def act(self, observations: np.array, **kwargs) -> Tuple[th.Tensor, th.Tensor]:
        """Samples actions and computes their probabilities (or probability densities in continuous case)

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors

        Returns:
            tuple with two Tensors:
                * actions [batch_size, actions_dim]
                * probabilities/densities [batch_size, 1]
        """

    @abstractmethod
    def act_deterministic(self, observations: np.array, **kwargs) -> th.Tensor:
        """Samples actions without exploration noise.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors

        Returns:
            Tensor of actions [batch_size, actions_dim]
        """

    def _calculate_density(self, policies, old_policies, mask) -> th.Tensor:
        with th.no_grad():
            policies_masked = policies * mask + (1 - mask) * th.ones_like(policies)
            old_policies_masked = old_policies * mask + (1 - mask) * th.ones_like(old_policies)

            policies_ratio = policies_masked / old_policies_masked
            policies_ratio_prod = th.cumprod(policies_ratio, dim=-1)

            return policies_ratio_prod

    def _ignore_above_b_mask(self, density, priorities):
        weights = density / th.reshape(priorities, (-1, 1))
        return (weights <= self._b).to(th.float32)

    def _ignore_outside_b_mask(self, density, priorities):
        weights = density / th.reshape(priorities, (-1, 1))
        return ((weights <= self._b) * (weights >= 1 / self._b)).to(th.float32)

    def _calculate_truncated_density(self, density, mask=None):
        with th.no_grad():
            if mask is not None:
                density = th.nan_to_num(density * mask, 0., 0., 0.)
            if self._clip_b:
                density = th.minimum(density, self._b)
            elif self._truncate:
                density = th.tanh(density / self._b) * self._b

            return density

    def _calculate_truncated_weights(self, density, priorities, mask=None):
        with th.no_grad():
            weights = density / th.reshape(priorities, (-1, 1))

            if mask is not None:
                weights = weights * mask
                weights = th.nan_to_num(weights, 0., 0., 0.)
            if self._clip_b:
                weights = th.minimum(weights, self._b)
                weights = th.nan_to_num(weights, 0., 0., 0.)
            elif self._truncate:
                weights = th.tanh(weights / self._b) * self._b
                weights = th.nan_to_num(weights, 0., 0., 0.)

            return weights

    def update_ends(self, ends):
        pass

class BaseCritic(BaseModel):
    @staticmethod
    def get_args(preproc_params, component):
        return BaseModel.get_args(preproc_params, component)

    def __init__(self, observations_space: gym.Space, layers: Optional[Tuple[int]],
                 th_time_step: th.Tensor, use_additional_input: bool = False, *args, nouts: int = 1, **kwargs):
        """Value function approximation as MLP network neural network.

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layers sizes, eg: for neural network with two layers with 10 and 20 hidden units
                pass: [10, 20]
            additional_input_shape: shape of additional input variables
        """
        if len(observations_space.shape) > 1:
            assert not use_additional_input

        super().__init__(observations_space, layers, nouts, *args, **kwargs)

        self.obs_shape = observations_space.shape
        self._th_time_step = th_time_step
        self._use_additional_input = use_additional_input

        self.register_method('value', self.value, {'observations': 'obs'})
        self.register_method('value_next', self.value, {'observations': 'obs_next'})

        self.register_method('optimize', self.optimize, {
            'value': 'self.value',
            'd': 'base.weighted_td'
        })
        self.targets = ['optimize']

    def call(self, inputs, training=None, mask=None, additional_input=None):
        return self.value(inputs, additional_input=additional_input)

    def value(self, observations: th.Tensor, additional_input: th.Tensor=None, **kwargs) -> th.Tensor:
        """Calculates value function given observations batch

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors


        Returns:
            Tensor [batch_size, 1] with value function estimations

        """
        if self._use_additional_input:
            x = th.concat([observations, additional_input], axis=-1)
        else:
            x = observations
        return self._forward(x)


class Critic(BaseCritic):
    @staticmethod
    def get_args(preproc_params, component):
        return BaseCritic.get_args(preproc_params, component)

    def __init__(self, observations_space: gym.Space, layers: Optional[Tuple[int]], th_time_step: th.Tensor,
                 use_additional_input: bool = False, *args, nouts=1, **kwargs):
        """Basic Critic that outputs single value"""
        super().__init__(observations_space, layers, th_time_step, *args, use_additional_input=use_additional_input, nouts=nouts, **kwargs)

    def _loss(self, value: th.Tensor, d: th.Tensor) -> th.Tensor:
        """Computes Critic's loss.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            d: batch [batch_size, 1] of gradient update coefficient (summation term in the Equation (9)) from
                the paper (1))
        """

        value = value[:, 0, :]
        loss = th.mean(-value * d.detach())

        return loss


class CategoricalActor(BaseActor):
    @staticmethod
    def get_args(preproc_params, component):
        args = BaseActor.get_args(preproc_params, component)
        args['entropy_coeff'] = (float, 0.)

        return args

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 *args, entropy_coeff=0., **kwargs):
        """BaseActor for discrete actions spaces. Uses Categorical Distribution"""
        super().__init__(observations_space, actions_space, layers, *args, **kwargs)
        self._entropy_coeff = entropy_coeff
        self.n = actions_space.n

    @property
    def action_dtype(self):
        return th.int32

    @property
    def action_dtype_np(self):
        return np.int32

    def _loss(self, observations: th.Tensor, actions: th.Tensor, d: th.Tensor) -> th.Tensor:
        probs, log_probs, action_probs, action_log_probs = self._prob(observations, actions)

        total_loss = th.mean(-th.unsqueeze(action_log_probs, 1) * d.detach())  # + penalty)

        # entropy maximization penalty
        entropy = -th.sum(probs * log_probs, axis=1)
        # penalty = self.beta_penalty * (-tf.reduce_sum(tf.math.multiply(probs, log_probs), axis=1))

        return total_loss - entropy * self._entropy_coeff

    def prob(self, observations: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        _, _, action_probs, action_log_probs = self._prob(observations, actions)
        return action_probs, action_log_probs

    def _prob(self, observations: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        logits = self._forward(observations)  # tf.divide(self._forward(observations) , 10)

        probs = th.softmax(logits)
        log_probs = th.log_softmax(logits) 

        action_probs = th.gather(probs, -1, actions)
        action_log_probs = th.gather(log_probs, -1, actions)

        return probs, log_probs, action_probs, action_log_probs

    def act(self, observations: th.Tensor, **kwargs) -> Tuple[th.Tensor, th.Tensor]:

        # TODO: remove hardcoded '10' and '20'
        logits = self._forward(observations)  # tf.divide(self._forward(observations) , 10)
        probs = th.softmax(logits)

        dist = th.distributions.Categorical(probs=probs)

        actions = dist.sample()
        actions = th.clip(actions, 0, self.n - 1)  # there was some weird error

        action_probs = th.gather(probs, -1, actions)

        return actions, action_probs

    def act_deterministic(self, observations: th.Tensor, **kwargs) -> th.Tensor:
        """Performs most probable action"""
        logits = self._forward(observations)  # tf.divide(self._forward(observations) , 10)

        actions = th.argmax(logits, axis=-1)
        return actions


class GaussianActor(BaseActor):
    @staticmethod
    def get_args(preproc_params, component):
        args = BaseActor.get_args(preproc_params, component)
        args['entropy_coeff'] = (float, 0.)
        args['beta_penalty'] = (float, 0.1)
        args['clip_mean'] = (float, None)
        args['distribution'] = (str, 'normal', {'choices': DISTRIBUTIONS.keys()})
        args['act_policy_epsilon'] = (float, None)
        args['std'] = (float, None)

        return args

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, actions_bound: float = None, std: float = None, device: str = None,
                 *args, distribution: str = 'normal', clip_mean: float = None, act_policy_epsilon: float = None, **kwargs):
        """BaseActor for continuous actions space. Uses MultiVariate Gaussian Distribution as policy distribution.

        TODO: introduce [a, b] intervals as allowed actions bounds

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layer sizes
            beta_penalty: penalty for too confident actions coefficient
            actions_bound: upper (lower == '-actions_bound') bound for allowed actions,
             required in case of continuous actions, deprecated (now taken as actions_space.high)
        """
        super().__init__(observations_space, actions_space, layers, beta_penalty, *args, device=device, **kwargs)

        self._actions_bound = th.as_tensor(actions_space.high, device=device)
        self._clip_mean = clip_mean
        self._act_policy_epsilon = act_policy_epsilon
        self._k = actions_space.shape[0]

        self.distribution = DISTRIBUTIONS[distribution]

        if std:
            std = th.as_tensor(std, device=device)
        else:
            std = 0.4 * self._actions_bound

        self.log_std = th.ones(self._k, device=device) * th.log(std)

        self.register_method('mean_and_std', self.mean_and_std, {'observation': 'obs'})
        self.register_method('prev_mean_and_std', self.mean_and_std, {'observation': 'prev_obs'})
        self.register_method('next_mean_and_std', self.mean_and_std, {'observation': 'obs_next'})

    def mean_and_std(self, observation):
        mean, std = self._forward(observation), th.exp(self.log_std)
        if self._clip_mean is not None:
            mean = th.clip(mean, -self._clip_mean, self._clip_mean)
        return mean, std

    @property
    def action_dtype(self):
        return th.float32

    @property
    def action_dtype_np(self):
        return np.float32

    def _loss(self, observations: th.Tensor, actions: th.Tensor, d: th.Tensor) -> th.Tensor:
        mean, std = self.mean_and_std(observations)
        dist = self.distribution(
            loc=mean,
            scale_diag=std.detach()
        )

        return self._loss_from_dist(dist, actions, d)

    def _calculate_beta_penalty(self, dist) -> th.Tensor:
        mean = dist.mode()

        bounds_penalty = th.sum(
            self.beta_penalty * th.maximum(th.abs(mean) - self._actions_bound, th.zeros_like(mean)) ** 2,
            dim=1,
            keepdim=True
        )

        return bounds_penalty

    def _loss_from_dist(self, dist, actions: th.Tensor, d: th.Tensor) -> th.Tensor:
        action_log_probs = th.unsqueeze(dist.log_prob(actions), dim=1)

        bounds_penalty = self._calculate_beta_penalty(dist)

        total_loss = th.mean(-action_log_probs * d.detach() + bounds_penalty)

        return total_loss

    def _dist(self, observations: th.Tensor) -> th.Tensor:
        mean, std = self.mean_and_std(observations)
        dist = self.distribution(
            loc=mean,
            scale_diag=std
        )
        return dist

    def prob(self, observations: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        dist = self._dist(observations)
        return self._prob(dist, actions)

    def _prob(self, dist, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return dist.prob(actions), dist.log_prob(actions)

    def act(self, observations: th.Tensor, **kwargs) -> Tuple[th.Tensor, th.Tensor]:
        dist = self._dist(observations)
        actions, probs = self._act(dist)
        if self._act_policy_epsilon is not None:
            probs = th.maximum(probs, self._act_policy_epsilon)
        return actions, probs

    def _act(self, dist: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = dist.sample()
        actions_probs = dist.prob(actions)

        return actions, actions_probs

    def act_deterministic(self, observations: th.Tensor, **kwargs) -> th.Tensor:
        """Returns mean of the Gaussian"""
        dist = self._dist(observations)
        return dist.mode()


class BaseACERAgent(AutoModelComponent, AutoModel):
    @staticmethod
    def get_args(preproc_params, component):
        adapt_gamma = '.'.join(component + ['gamma']) not in preproc_params.get('keep', ())
        adapt_learning_starts = '.'.join(component + ['learning_starts']) not in preproc_params.get('keep', ())
        return {
            'gamma': (
                float, 0.99 ** (1 / preproc_params.get('timesteps_increase', 1)),
                {'action': DTRootIncrease(preproc_params.get('timesteps_increase', 1))} if adapt_gamma else {}),
            'actor_lr': (float, 3e-5),
            'actor_adam_epsilon': (float, None),
            'actor_adam_beta1': (float, 0.9),
            'actor_adam_beta2': (float, 0.999),
            'critic_lr': (float, 1e-4),
            'critic_adam_epsilon': (float, None),
            'critic_adam_beta1': (float, 0.9),
            'critic_adam_beta2': (float, 0.999),
            'c': (int, 1),
            'c0': (float, 1),
            'batches_per_env': (int, 256),
            'learning_starts': (
                int, 10000 * preproc_params.get('timesteps_increase', 1),
                {'action': DTLinearIncrease(preproc_params.get('timesteps_increase', 1))} if adapt_learning_starts else {}),
        }

    """Base ACER abstract class"""
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, gamma: int = 0.99,
                 memory_size: int = 1e6, num_parallel_envs: int = 10,
                 batches_per_env: int = 5, c: int = 10, c0: float = 0.3, actor_lr: float = 0.001,
                 actor_adam_beta1: float = 0.9, actor_adam_beta2: float = 0.999, actor_adam_epsilon: float = 1e-7,
                 critic_lr: float = 0.001, critic_adam_beta1: float = 0.9, critic_adam_beta2: float = 0.999,
                 critic_adam_epsilon: float = 1e-7, time_step: int = 1, learning_starts: int = 1000,
                 policy_spec: BufferFieldSpec = None, summary_writer = None, additional_buffer_types = (),
                 device: str = None, **kwargs):

        super().__init__()

        self._th_time_step = th.zeros((), dtype=th.int64) + time_step
        self._observations_space = observations_space
        self._actions_space = actions_space
        self._c = c
        self._c0 = c0
        self._learning_starts = learning_starts
        self._gamma = th.as_tensor(gamma, device=device)
        self._batches_per_env = batches_per_env
        self._time_step = 0
        self._num_parallel_envs = num_parallel_envs
        self._batch_size = self._num_parallel_envs * self._batches_per_env

        self.device = device
        self.summary_writer = summary_writer

        self._is_obs_discrete = type(observations_space) == gym.spaces.Discrete

        if type(actions_space) == gym.spaces.Discrete:
            self._is_discrete = True
            self._actions_bound = 0
        else:
            self._is_discrete = False
            self._actions_bound = actions_space.high

        self._actor = self._init_actor()
        self._critic = self._init_critic()

        self._init_replay_buffer(memory_size, policy_spec)

        self.register_method("first_obs", self._first_obs, {"obs": "obs"})
        self.register_method("first_actions", self._first_actions, {"actions": "actions"})

        self._init_automodel()

        self._init_data_loader(additional_buffer_types)

        self._actor.init_optimizer(
            lr=actor_lr,
            betas=(actor_adam_beta1, actor_adam_beta2),
            eps=actor_adam_epsilon
        )

        self._critic.init_optimizer(
            lr=critic_lr,
            betas=(critic_adam_beta1, critic_adam_beta2),
            eps=critic_adam_epsilon
        )

    def as_tensors(self, data: dict, ignore: tuple = ()):
        return {k: th.as_tensor(v, device=self.device) if k not in ignore else v for k, v in data.items()}

    def _first_obs(self, obs):
        return obs[:, 0]

    def _first_actions(self, actions):
        return actions[:, 0]

    def _init_automodel(self) -> None:
        pass  # call for automodel-compatibile classes (FastACER and subseq.) before immediatly before initializing data loader

    def _init_data_loader(self, additional_buffer_types) -> None:
        raise NotImplementedError

    def _init_replay_buffer(self, memory_size: int, policy_spec: BufferFieldSpec = None):
        if type(self._actions_space) == gym.spaces.Discrete:
            actions_shape = (1, )
        else:
            actions_shape = self._actions_space.shape

        self._memory = MultiReplayBuffer(
            buffer_class=ReplayBuffer,
            action_spec=BufferFieldSpec(shape=actions_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=self._observations_space.shape, dtype=self._observations_space.dtype),
            policy_spec=policy_spec,
            max_size=memory_size,
            num_buffers=self._num_parallel_envs
        )

    def save_experience(self, steps: List[
        Tuple[Union[int, float, list], np.array, float, np.array, bool, bool]
    ]):
        """Stores gathered experiences in a replay buffer. Accepts list of steps.

        Args:
            steps: List of steps, see ReplayBuffer.put() for a detailed format description
        """
        self._time_step += len(steps)
        self._th_time_step += len(steps)
        self._memory.put(steps)

        self._actor.update_ends(np.array([[step[5]] for step in steps]))

    @th.compile(fullgraph=True)
    def predict_action(self, observations: np.array, is_deterministic: bool = False) \
            -> Tuple[np.array, Optional[np.array], Any]:
        """Predicts actions for given observations. Performs forward pass with BaseActor network.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            is_deterministic: True if mean actions (without exploration noise) should be returned

        Returns:
            Tuple of sampled actions and corresponding probabilities (probability densities) if action was sampled
                from the distribution, None otherwise
        """
        with th.no_grad():
            if is_deterministic:
                return self._actor.act_deterministic(th.as_tensor(observations, device=self.device)).detach().cpu().numpy(), None, None
            else:
                actions, policies = self._actor.act(th.as_tensor(observations, device=self.device))
                return actions.detach().cpu().numpy(), policies.detach().cpu().numpy(), None

    @abstractmethod
    def _fetch_offline_batch(self) -> List[Tuple[Dict[str, Union[np.array, list]], int]]:
        ...

    @abstractmethod
    def learn(self):
        ...

    @abstractmethod
    def _init_actor(self) -> BaseActor:
        ...

    @abstractmethod
    def _init_critic(self) -> BaseCritic:
        ...

    def save(self, path: Path, **kwargs):
        actor_path = str(path / 'actor.weights.h5')
        critic_path = str(path / 'critic.weights.h5')
        buffer_path = str(path / 'buffer.pkl')

        th.save(self._actor, actor_path)
        th.save(self._critic, critic_path)

        self._memory.save(buffer_path)
