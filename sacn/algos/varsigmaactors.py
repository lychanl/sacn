import numpy as np
from algos.base import GaussianActor
import torch as th


class VarSigmaActor(GaussianActor):
    @staticmethod
    def get_args(preproc_params, component):
        args = GaussianActor.get_args(preproc_params, component)
        args['entropy_bonus'] = (float, 0)
        args['single_std'] = (bool, False, {'action': 'store_true'})
        args['nn_std'] = (bool, False, {'action': 'store_true'})
        args['separate_nn_std'] = (int, None, {'nargs': '*'})
        args['initial_log_std'] = (float, 0)
        args['clip_log_std'] = (float, None, {'nargs': 2})
        args['use_soft_plus'] = (bool, None, {'action': 'store_true'})
        return args

    def __init__(
            self, obs_space, action_space,
            *args, entropy_bonus=0, dist_std_gradient=True,
            single_std=False, nn_std=False, separate_nn_std=None,
            initial_log_std=0, std_loss_args=None, custom_optimization=False,
            clip_log_std=None, use_soft_plus=False, device,
            **kwargs):

        self.entropy_bonus = entropy_bonus
        self.dist_std_gradient = dist_std_gradient
        self.separate_nn_std = separate_nn_std
        self.single_std = single_std
        self.nn_std = nn_std
        self.initial_log_std = initial_log_std
        self.clip_log_std = clip_log_std if not use_soft_plus else None
        self.use_soft_plus = use_soft_plus

        if separate_nn_std is not None:
            additional_outputs = 0
            if single_std:
                extra_models = ([*separate_nn_std, 1],)
            else:
                extra_models = ([*separate_nn_std, action_space.shape[0]],)
        elif nn_std:
            extra_models = ()
            if single_std:
                additional_outputs = 1
            else:
                additional_outputs = action_space.shape[0]
        else:
            additional_outputs = 0
            extra_models = ()

        GaussianActor.__init__(self, obs_space, action_space, *args, **kwargs, additional_outputs=additional_outputs, extra_models=extra_models, device=device)

        if not separate_nn_std and not nn_std:
            if single_std:
                self.var_log_std = th.nn.Parameter(th.full((), initial_log_std, dtype=th.float32, requires_grad=True, device=device))
            else:
                self.var_log_std = th.nn.Parameter(th.full(action_space.shape, initial_log_std, dtype=th.float32, requires_grad=True, device=device))

        self.register_method('std', self.std, {'observations': 'obs'})

        if not custom_optimization:
            if std_loss_args is None:
                std_loss_args = {
                    'observations': 'base.first_obs',
                    'actions': 'base.first_actions',
                    'd': 'base.weighted_td'
                }

            for k, v in std_loss_args.items():
                self.methods['optimize'][1][k] = v

    @property
    def mean_trainable_variables(self):
        return self._hidden_layers.trainable_variables

    @property
    def std_trainable_variables(self):
        if self.separate_nn_std:
            return self._extra_hidden_layers.trainable_variables
        else:
            return [self.var_log_std]

    def mean_and_log_std(self, observation):
        out = self._forward(observation)
        if self.separate_nn_std:
            mean = out
            log_std, = self._extras_forward(observation)
            log_std += self.initial_log_std
        elif self.nn_std:
            mean = out[..., :self._k]
            log_std = out[..., self._k:] + self.initial_log_std
        else:
            mean = out
            log_std = th.ones_like(mean[...,:1]) * th.unsqueeze(self.var_log_std, 0)
        if self.single_std:
            log_std = th.repeat_interleave(log_std, self._k, dim=-1)
        
        if self._clip_mean is not None:
            mean = th.clip(mean, -self._clip_mean, self._clip_mean)
        if self.clip_log_std is not None:
            log_std = th.clip(log_std, self.clip_log_std[0], self.clip_log_std[1])

        return mean, log_std

    def mean_and_std(self, observation):
        mean, log_std = self.mean_and_log_std(observation)
        if self.use_soft_plus:
            std = th.nn.functional.softplus(log_std) + 1e-5
        std = th.exp(log_std)
        return mean, std if self.dist_std_gradient else std.detach()

    def std(self, observations):
        return self.mean_and_std(observations)[1]

    def _loss(self, **kwargs) -> th.Tensor:
        loss = self.mean_loss(**kwargs) + self.std_loss(**kwargs)
        return loss

    def mean_loss(self, observations: np.array, actions: np.array, d: np.array, **kwargs) -> th.Tensor:
        mean, std = self.mean_and_std(observations)

        dist = self.distribution(
            loc=mean,
            scale_diag=std.detach()
        )

        return self._loss_from_dist(dist, actions, d)

    def std_loss(self, observations: np.array, actions: np.array, d: np.array, **kwargs):
        mean, std = self.mean_and_std(observations)

        dist = self.distribution(
            loc=mean.detach(),
            scale_diag=std
        )

        entropy_bonus = self.entropy_bonus * th.mean(dist.entropy())

        return self._loss_from_dist(dist, actions, d) - entropy_bonus
