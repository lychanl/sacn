import gym
import numpy as np
import torch as th
from typing import Optional, Tuple

from algos.common.automodel import AutoModelComponent
from algos.base import BaseModel, GaussianActor
from algos.varsigmaactors import VarSigmaActor
from replay_buffer import MultiReplayBuffer, VecReplayBuffer
from algos.base_nextgen_acer import BaseNextGenACERAgent
from utils import DTLinearIncrease



class TwinQDelayedCritic(AutoModelComponent, th.nn.Module):
    @staticmethod
    def get_args(preproc_params, component):
        adapt_update_delay = '.'.join(component + ['update_delay']) not in preproc_params.get('keep', ())
        args = BaseModel.get_args(preproc_params, component)
        args['update_delay'] = (
            int, preproc_params.get('timesteps_increase', 1),
            {'action': DTLinearIncrease(preproc_params.get('timesteps_increase', 1))} if adapt_update_delay else {})
        args['tau'] = (float, 0.005)
        return args

    def __init__(
            self, observations_space: gym.Space, action_space: gym.Space, layers,
            *args, update_delay=1, tau=0.005, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self._update_delay = update_delay
        self._tau = tau
        q_input_space, q_outs = self._prepare_q_dims(observations_space, action_space)
        self._q1 = BaseModel(q_input_space, layers, q_outs, *args, **kwargs)
        self._q2 = BaseModel(q_input_space, layers, q_outs, *args, **kwargs)
        self._target_q1 = BaseModel(q_input_space, layers, q_outs, *args, **kwargs)
        self._target_q2 = BaseModel(q_input_space, layers, q_outs, *args, **kwargs)

        # if not isinstance(observations_space, gym.spaces.Discrete):
        #     sample_input = np.zeros((1, *q_input_space.shape))
        #     # build models
        #     self._q1(sample_input)
        #     self._q2(sample_input)
        #     self._target_q1(sample_input)
        #     self._target_q2(sample_input)

        self._target_q1.load_state_dict(self._q1.state_dict())
        self._target_q2.load_state_dict(self._q2.state_dict())

        for prefix in '', 'target_':
            for ob in '', '_next':
                self.register_method(f'{prefix}qs{ob}', self.target_qs if prefix else self.qs, {
                    'obs': f'obs{ob}',
                    'actions': 'actor.next_actions' if ob else 'actions'
                })
                self.register_method(f'{prefix}q{ob}', self.min_q, {
                    'qs': f'self.{prefix}qs{ob}',
                })
        self.register_method('td_q_est', self.td_q_est, {
            'next_min_target_qs': 'self.target_q_next',
            'dones': 'dones',
            'rewards': 'rewards',
            'discount': 'base.gamma',
        })
        self.register_method('optimize', self.optimize, {
            'qs': 'self.qs',
            'td_q_est': 'self.td_q_est',
            'timestep': 'base.time_step',
        })
        self.targets = ['optimize']

    def _prepare_q_dims(self, observations_space, action_space):
        if isinstance(observations_space, gym.spaces.Discrete):
            assert isinstance(action_space, gym.spaces.Discrete)
            q_input_space = observations_space
        else:
            q_input_space_size = observations_space.shape[0]
            low = observations_space.low
            high = observations_space.high
            if not isinstance(action_space, gym.spaces.Discrete):
                q_input_space_size += action_space.shape[0]
                low = np.concatenate([low, action_space.low])
                high = np.concatenate([high, action_space.high])
            q_input_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        if isinstance(action_space, gym.spaces.Discrete):
            q_outs = action_space.n
            self.discrete_actions = True
        else:
            q_outs = 1
            self.discrete_actions = False
        return q_input_space, q_outs


    def qs(self, obs, actions):
        if not self.discrete_actions:
            input = th.concat([obs, actions], dim=-1)
        else:
            input = obs

        out1 = self._q1._forward(input)
        out2 = self._q2._forward(input)

        if self.discrete_actions:
            out1 = th.gather(out1, -1, actions)
            out2 = th.gather(out2, -1, actions)

        return th.concat([out1, out2], dim=-1)

    def target_qs(self, obs, actions):
        if not self.discrete_actions:
            input = th.concat([obs, actions], dim=-1)
        else:
            input = obs

        out1 = self._target_q1._forward(input)
        out2 = self._target_q2._forward(input)

        if self.discrete_actions:
            out1 = th.gather(out1, -1, actions)
            out2 = th.gather(out2, -1, actions)

        return th.concat([out1, out2], dim=-1)

    def min_q(self, qs):
        return th.min(qs, dim=-1)[0]

    def td_q_est(self, next_min_target_qs, dones, rewards, discount):
        target = rewards + th.unsqueeze(
            discount * next_min_target_qs * (1 - dones.to(th.float32)),
            -1
        )
        return target

    def _loss(self, qs, td_q_est):
        return th.mean((qs[:, 0, :] - td_q_est.detach()) ** 2)

    def optimize(self, timestep, **loss_kwargs):
        loss_qs = self._loss(**loss_kwargs)

        self._q1.optimizer.zero_grad(set_to_none=True)
        self._q2.optimizer.zero_grad(set_to_none=True)
        loss_qs.backward()
        self._q1.optimizer.step()
        self._q2.optimizer.step()

        with th.no_grad():
            assign_mask = ((timestep - 1) % self._update_delay == 0).to(th.float32)
            for q, target_q in (
                (self._q1, self._target_q1),
                (self._q2, self._target_q2),
            ):
                for w1, w2 in zip(q.parameters(), target_q.parameters()):
                    assert w1.data.shape == w2.data.shape
                    th.add((w2.data * (1 - self._tau) + w1.data * self._tau) * assign_mask, w2.data * (1 - assign_mask), out=w2.data)

        return loss_qs

    def init_optimizer(self, *args, **kwargs):
        self._q1.init_optimizer(*args, **kwargs)
        self._q2.init_optimizer(*args, **kwargs)
        self._target_q1.init_optimizer(*args, **kwargs)
        self._target_q2.init_optimizer(*args, **kwargs)


class GaussianQActor(GaussianActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_method('next_actions_with_log_prob', self.sample_inplace_with_log_prob, {
            'obs': 'obs_next'
        })

        self.register_method('next_actions', self.sample_inplace, {
            'act_with_log_prob': 'self.next_actions_with_log_prob'
        })

        self.register_method('next_log_prob', self.expected_action_log_prob, {
            'act_with_log_prob': 'self.next_actions_with_log_prob'
        })

        self.register_method('optimize', self.optimize, {
            'obs': 'base.first_obs',
        })

    def sample_inplace_with_log_prob(self, obs):
        dist = self._dist(obs)
        return dist.sample_with_log_prob()

    def sample_inplace(self, act_with_log_prob):
        act, _ = act_with_log_prob
        return act
    
    def expected_action_log_prob(self, act_with_log_prob):
        _, lp = act_with_log_prob
        return lp

    def _loss(self, obs, **kwargs):
        mean, std = self.mean_and_std(obs)
        dist = self.distribution(
            loc=mean,
            scale_diag=std
        )

        actions, log_probs = dist.sample_with_log_prob()

        qs = self.call_now('critic.qs', {'obs': obs, 'actions': actions})
        min_q = th.min(qs, dim=-1)[0]

        actor_loss = th.mean(-min_q)
        bounds_penalty = self._calculate_beta_penalty(dist)

        return actor_loss + bounds_penalty
    


class ACER_Q(BaseNextGenACERAgent):
    ACTORS = {'simple': {False: GaussianQActor}}
    CRITICS = {'simple': TwinQDelayedCritic}
    BUFFERS = {
        'simple': (MultiReplayBuffer, {'buffer_class': VecReplayBuffer}),
    }

    def __init__(self, *args, **kwargs):
        if 'buffer.n' in kwargs:
            assert kwargs['buffer.n'] == 1
        else:
            kwargs['buffer.n'] = 1

        super().__init__(*args, **kwargs)


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
