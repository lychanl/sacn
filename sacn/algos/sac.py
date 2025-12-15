import gym
import numpy as np
import torch as th
from typing import Optional, Tuple

from algos.common.automodel import AutoModelComponent
from algos.base import BaseModel
from algos.acer_q import TwinQDelayedCritic
from algos.varsigmaactors import VarSigmaActor
from replay_buffer import MultiReplayBuffer, VecReplayBuffer
from algos.base_nextgen_acer import BaseNextGenACERAgent


class TwinQDelayedSoftCritic(TwinQDelayedCritic):

    def __init__(
            self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.register_method('td_q_est', self.td_q_est, {
            'next_min_target_qs': 'self.target_q_next',
            'dones': 'dones',
            'rewards': 'rewards',
            'discount': 'base.gamma',
            'entropy_coef': 'actor.entropy_coef',
            'next_log_prob': 'actor.next_log_prob'
        })

    def td_q_est(self, next_min_target_qs, dones, rewards, discount, entropy_coef, next_log_prob):
        target = rewards + th.unsqueeze(
            discount * (next_min_target_qs - entropy_coef * next_log_prob) * (1 - dones.to(th.float32)),
            -1
        ).detach()
        return target

    def init_optimizer(self, *args, **kwargs):
        self._q1.init_optimizer(*args, **kwargs)
        self._q2.init_optimizer(*args, **kwargs)


class GaussianSoftActor(VarSigmaActor):
    @staticmethod
    def get_args(preproc_params, component):
        args = VarSigmaActor.get_args(preproc_params, component)
        args['nn_std'] = (bool, True)
        args['target_entropy'] = (float, None)
        args['clip_mean'] = (float, None)
        return args

    def __init__(self, obs_space, action_space, *args, target_entropy=None, nn_std=True, clip_mean=None, device, **kwargs):
        super().__init__(obs_space, action_space, *args, custom_optimization=True, nn_std=nn_std, clip_mean=clip_mean, device=device, **kwargs)
        if target_entropy is None:
            target_entropy = -np.prod(action_space.shape)
        self._target_entropy = target_entropy
        self.log_entropy_coef = th.nn.Parameter(th.zeros((), dtype=th.float32, device=device, requires_grad=True))

        self.register_method('entropy_coef', self.entropy_coef, {})

        self.register_method('optimize', self.optimize, {
            'obs': 'base.first_obs', 'entropy_coef': 'self.entropy_coef'
        })

        self.register_method('next_actions_with_log_prob', self.sample_inplace_with_log_prob, {
            'obs': 'obs_next'
        })

        self.register_method('next_actions', self.sample_inplace, {
            'act_with_log_prob': 'self.next_actions_with_log_prob'
        })

        self.register_method('next_log_prob', self.expected_action_log_prob, {
            'act_with_log_prob': 'self.next_actions_with_log_prob'
        })

        self.register_method(
            'sample_weights', self._calculate_truncated_weights, {
                'priorities': 'priorities'
            }
        )

    def _calculate_truncated_weights(self, priorities):
        return th.reshape(priorities, (-1, 1))

    def sample_inplace_with_log_prob(self, obs):
        dist = self._dist(obs)
        return dist.sample_with_log_prob()

    def sample_inplace(self, act_with_log_prob):
        act, _ = act_with_log_prob
        return act
    
    def expected_action_log_prob(self, act_with_log_prob):
        _, lp = act_with_log_prob
        return lp

    def entropy_coef(self):
        return th.exp(self.log_entropy_coef)

    #    # def mean_and_std(self, observations):
    #     mean, std = super().mean_and_std(observations)
    #     return mean, tf.ones_like(std) * 0.25

    def _loss(self, obs, entropy_coef):
        mean, std = self.mean_and_std(obs)
        dist = self.distribution(
            loc=mean,
            scale_diag=std
        )

        actions, log_probs = dist.sample_with_log_prob()

        qs = self.call_now('critic.qs', {'obs': obs, 'actions': actions})
        min_q = th.min(qs, dim=-1)[0]

        entropy_loss = -th.mean(self.log_entropy_coef * (log_probs + self._target_entropy).detach())
        actor_loss = th.mean(entropy_coef.detach() * log_probs - min_q)
        return entropy_loss + actor_loss


class SAC(BaseNextGenACERAgent):
    ACTORS = {'simple': {False: GaussianSoftActor}}
    CRITICS = {'simple': TwinQDelayedSoftCritic}
    BUFFERS = {
        'simple': (MultiReplayBuffer, {'buffer_class': VecReplayBuffer}),
    }

    def __init__(self, *args, **kwargs):
        self._validate(kwargs)

        super().__init__(*args, **kwargs)

    def _validate(self, kwargs):
        if 'buffer.n' in kwargs:
            assert kwargs['buffer.n'] == 1
        else:
            kwargs['buffer.n'] = 1

    def _init_automodel(self, skip=()):
        self.register_method('weighted_td', self._calculate_weighted_td, {
            "td_q_est": "critic.td_q_est",
            "qs": "critic.qs",
            "weights": "actor.sample_weights",
        })

        super()._init_automodel(skip=skip)


    def _init_critic(self):
        # if self._is_obs_discrete:
        #     return TabularCritic(self._observations_space, None, self._tf_time_step)
        # else:
        return self.CRITICS[self._critic_type](
            self._observations_space, self._actions_space, device=self.device,
            th_time_step=self._th_time_step, **self._critic_args
        )

    def _calculate_weighted_td(self, td_q_est, qs, weights):
        return (th.mean(qs, dim=-1) - td_q_est) * weights
