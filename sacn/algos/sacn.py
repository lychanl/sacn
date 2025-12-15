from algos.base import GaussianActor
from algos.sac import SAC, GaussianSoftActor, TwinQDelayedSoftCritic

import torch as th


class TwinQDelayedSoftNCritic(TwinQDelayedSoftCritic):
    @staticmethod
    def get_args(preproc_params, component):
        args = TwinQDelayedSoftCritic.get_args(preproc_params, component)
        args['sample_n_limit'] = (bool, None, {'action': 'store_true'})
        args['sample_n_limit_geom'] = (bool, None, {'action': 'store_true'})
        return args

    def __init__(self, *args, sample_n_limit=False, sample_n_limit_geom=False, **kwargs):
        self.sample_n_limit = sample_n_limit
        self.sample_n_limit_geom = sample_n_limit_geom
        super().__init__(*args, **kwargs)

        self.register_method('target_qs_sampled', self.target_qs, {
            'obs': 'obs',
            'actions': 'actor.sampled_actions'
        })

        self.register_method('target_q_sampled', self.min_q, {
            'qs': 'self.target_qs_sampled'
        })

        self.register_method('td_q_est', self.td_q_est, {
            'min_target_qs': 'self.target_q_sampled',
            'next_min_target_qs': 'self.target_q_next',
            'dones': 'dones',
            'rewards': 'rewards',
            'discount': 'base.gamma',
            'entropy_coef': 'actor.entropy_coef',
            'log_prob': 'actor.sampled_n_log_prob',
            'next_log_prob': 'actor.next_sampled_n_log_prob',
            'lengths': 'lengths',
            'n': 'n'
        })

        self.register_method('optimize', self.optimize, {
            'qs': 'self.qs',
            'td_q_est': 'self.td_q_est',
            'timestep': 'base.time_step',
            'mask': 'base.mask',
            'sample_weights': 'actor.sample_weights'
        })

    def td_q_est(self, min_target_qs, next_min_target_qs, dones, rewards, discount, entropy_coef, log_prob, next_log_prob, lengths, n):
        dones_mask = 1 - (
            (th.unsqueeze(th.arange(1, n + 1, device=lengths.device), 0) == th.unsqueeze(lengths, 1)) & th.unsqueeze(dones, 1)
        ).to(th.float32)
        discounts = th.unsqueeze(discount ** th.arange(0, n, device=rewards.device), 0)
        discounts_next = dones_mask * discounts
        log_probs = th.concatenate((log_prob[:, 1:], th.unsqueeze(next_log_prob, axis=-2)), axis=1)
        if self.sample_n_limit or self.sample_n_limit_geom:
            if self.sample_n_limit:
                limit = th.arange(1, n + 1, device=log_prob.device)
            elif self.sample_n_limit_geom:
                limit = th.round((1 - discount ** (2 * th.arange(0, n, device=log_prob.device))) / (1 - discount ** 2))
            limit = th.reshape(limit, (1, -1, 1))
            lp_mask = th.cumsum(th.ones_like(log_probs), dim=-1) <= th.reshape(th.arange(1, n + 1, device=log_prob.device), (1, -1, 1))
            mean_log_probs = (th.sum(th.unsqueeze(lp_mask, 1) * th.unsqueeze(log_probs, 2), dim=3) / th.unsqueeze(th.sum(lp_mask, dim=2), 1))
            discounted_mean_log_probs = th.unsqueeze(discounts_next, dim=2) * entropy_coef * mean_log_probs
            cum_discounted_mean_log_probs = th.cumsum(discounted_mean_log_probs, dim=1)
            discounted_log_probs = th.diagonal(cum_discounted_mean_log_probs, dim1=1, dim2=2)
        else:
            log_probs = th.mean(log_probs, -1)
            discounted_log_probs = th.cumsum(discounts_next * entropy_coef * log_probs, dim = 1)
        target_qs = th.concatenate((min_target_qs[:, 1:], th.unsqueeze(next_min_target_qs, axis=-1)), axis=1)

        target = th.cumsum(discounts * rewards, dim=1) + discount * (
            discounts_next * target_qs - discounted_log_probs
        )
        return target

    def _loss(self, qs, td_q_est, mask, sample_weights):
        q_sample_weights = th.concatenate((th.ones_like(sample_weights[:, :1]), sample_weights[:, :-1]), dim=1).detach()
        return th.mean((qs[:, :1, :] - th.unsqueeze(td_q_est, -1).detach()) ** 2 * th.unsqueeze(mask * q_sample_weights, dim=-1))


class GaussianSoftNActor(GaussianSoftActor):
    @staticmethod
    def get_args(preproc_params, component):
        args = GaussianSoftActor.get_args(preproc_params, component)
        args['sample_n'] = (int, None)
        args['fixed2_qb'] = (bool, None, {'action': 'store_true'})
        args['qb'] = (bool, None, {'action': 'store_true'})
        args['qb_min1'] = (bool, None, {'action': 'store_true'})
        args['qb_limit'] = (float, None)
        args['abs_qb'] = (bool, None, {'action': 'store_true'})
        args['scale_weights'] = (bool, None, {'action': 'store_true'})
        args['scale_weights_mean'] = (bool, None, {'action': 'store_true'})
        args['scale_weights_median'] = (bool, None, {'action': 'store_true'})
        return args

    def __init__(
            self, *args, sample_n=1, qb=False, abs_qb=False, only_last_n=False,
            scale_weights=False, scale_weights_mean=False, scale_weights_median=False,
            qb_min1=False, fixed2_qb=False, qb_limit=None, **kwargs):
        self.sample_n = sample_n
        self.only_last_n = only_last_n
        self.qb = qb
        self.qb_min1 = qb_min1
        self.qb_limit = qb_limit
        self.fixed2_qb = fixed2_qb
        self.abs_qb = abs_qb
        self.scale_weights = scale_weights
        self.scale_weights_mean = scale_weights_mean
        self.scale_weights_median = scale_weights_median
        super().__init__(*args, **kwargs)

        self.register_method('sampled_n_actions_with_log_prob', self.sample_n_inplace_with_log_prob, {
            'obs': 'obs'
        })

        self.register_method('sampled_actions', self.sample_n_inplace, {
            'act_with_log_prob': 'self.sampled_n_actions_with_log_prob'
        })

        self.register_method('sampled_n_log_prob', self.expected_n_action_log_prob, {
            'act_with_log_prob': 'self.sampled_n_actions_with_log_prob'
        })

        self.register_method('next_sampled_n_actions_with_log_prob', self.sample_n_inplace_with_log_prob, {
            'obs': 'obs_next'
        })

        self.register_method('next_actions', self.sample_n_inplace, {
            'act_with_log_prob': 'self.next_sampled_n_actions_with_log_prob'
        })

        self.register_method('next_sampled_n_log_prob', self.expected_n_action_log_prob, {
            'act_with_log_prob': 'self.next_sampled_n_actions_with_log_prob'
        })

        if 'mask' in self.methods['truncated_density'][1]:
            mask_kwargs = {'mask': self.methods['truncated_density'][1]['mask']}
        else:
            mask_kwargs = {}

        if self.qb:
            self.register_method('b', self._calc_qb, {
                'density': 'actor.density',
            })
            self.register_method(
                'ignore_above_b_mask', self._ignore_above_qb_mask, {
                    'density': 'actor.density',
                    'priorities': 'priorities',
                    'b': 'self.b'
            })
            self.register_method(
                'ignore_outside_b_mask', self._ignore_outside_qb_mask, {
                    'density': 'actor.density',
                    'priorities': 'priorities',
                    'b': 'self.b'
            })
            b_kwargs = {'b': 'self.b'}
        else:
            b_kwargs = {}

        self.register_method(
            'sample_weights', self._calculate_truncated_weights, {
                'density': 'actor.density',
                'priorities': 'priorities',
                'discount': 'base.gamma',
                **b_kwargs,
                **mask_kwargs
            }
        )
        self.register_method(
            'truncated_density', self._calculate_truncated_density, {
                'density': 'actor.density',
                **b_kwargs,
                **mask_kwargs
            }
        )

    def _calc_qb(self, density):
        vals = th.concat((th.ones_like(density[:, :1]), density[:, 1:]), dim=1)
        if self.abs_qb:
            vals = th.abs(th.log(vals))
        if self.fixed2_qb:
            vals = th.nan_to_num(vals, 0)
        q = th.quantile(vals, self._b)
        if self.abs_qb:
            q = th.exp(q)
        if self.qb_min1:
            q = th.maximum(q, th.ones_like(q))
        if self.qb_limit is not None:
            q = th.minimum(q, th.ones_like(q) * self.qb_limit)
        return q

    def _ignore_above_qb_mask(self, density, priorities, b):
        weights = density / th.reshape(priorities, (-1, 1))
        return (weights <= b).to(th.float32)

    def _ignore_outside_qb_mask(self, density, priorities, b):
        weights = density / th.reshape(priorities, (-1, 1))
        return ((weights <= b) * (weights >= 1 / b)).to(th.float32)

    def _calculate_truncated_weights(self, density, priorities, b=None, mask=None, discount=None):
        if b is None:
            b = self._b
        with th.no_grad():
            weights = density / th.reshape(priorities, (-1, 1))

            if mask is not None:
                weights = weights * mask
                weights = th.nan_to_num(weights, 0., 0., 0.)
            if self._clip_b:
                weights = th.minimum(weights, b)
                weights = th.nan_to_num(weights, 0., 0., 0.)
            elif self._truncate:
                weights = th.tanh(weights / b) * b
                weights = th.nan_to_num(weights, 0., 0., 0.)

            if self.only_last_n:
                assert mask is not None
                # find last non-0 mask element
                non_0_mask = th.cumprod(mask, dim=1)
                cum_non_0_mask = th.cumsum(non_0_mask, dim=1)
                last_n = th.max(cum_non_0_mask, dim=1)[0]
                last_n_mask = cum_non_0_mask == last_n

                first_and_last_mask = th.concat((th.ones_like(last_n_mask[:, :1]), last_n_mask[:, 1:]), dim=1)

                weights = weights * first_and_last_mask

            if self.scale_weights:
                weights = th.nan_to_num(weights / th.max(weights, dim=0, keepdim=True)[0])
            elif self.scale_weights_mean:
                weights = th.nan_to_num(weights / th.mean(weights, dim=0, keepdim=True))
            elif self.scale_weights_median:
                weights = th.nan_to_num(weights / th.median(weights, dim=0, keepdim=True))

            return weights

    def _calculate_truncated_density(self, density, b=None, mask=None):
        if b is None:
            b = self._b
        with th.no_grad():
            if mask is not None:
                density = th.nan_to_num(density * mask, 0., 0., 0.)
            if self._clip_b:
                density = th.minimum(density, b)
            elif self._truncate:
                density = th.tanh(density / b) * b

            return density

    def sample_n_inplace_with_log_prob(self, obs):
        mean, std = self.mean_and_std(obs)
        dist = self.distribution(
            loc=th.repeat_interleave(th.unsqueeze(mean, dim=-2), self.sample_n, dim=-2),
            scale_diag=th.repeat_interleave(th.unsqueeze(std, dim=-2), self.sample_n, dim=-2)
        )

        return dist.sample_with_log_prob()

    def sample_n_inplace(self, act_with_log_prob):
        act, _ = act_with_log_prob
        return act[..., 0, :]
    
    def expected_n_action_log_prob(self, act_with_log_prob):
        _, lp = act_with_log_prob
        return lp


class SACN(SAC):
    ACTORS = {'simple': {False: GaussianSoftNActor}}
    CRITICS = {'simple': TwinQDelayedSoftNCritic}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _validate(self, kwargs):
        pass


class TwinQDelayedFastSoftNCritic(TwinQDelayedSoftNCritic):
    @staticmethod
    def get_args(preproc_params, component):
        args = TwinQDelayedSoftNCritic.get_args(preproc_params, component)
        args['sum_loss'] = (bool, None, {'action': 'store_true'})
        args['lam'] = (float, None)
        return args

    def __init__(self, *args, sum_loss=False, lam=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sum_loss = sum_loss
        self.lam = lam

        self.register_method('target_qs_sampled_ns', self.target_qs, {
            'obs': 'actor.obs_next_ns',
            'actions': 'actor.sampled_next_actions_ns'
        })

        self.register_method('target_q_sampled_ns', self.min_q, {
            'qs': 'self.target_qs_sampled_ns'
        })

        self.register_method('td_q_est', self.td_q_est, {
            'target_qs': 'self.target_q_sampled_ns',
            'dones': 'actor.dones_ns',
            'rewards': 'rewards',
            'discount': 'base.gamma',
            'entropy_coef': 'actor.entropy_coef',
            'log_prob': 'actor.sampled_n_log_prob',
            'next_log_prob': 'actor.next_sampled_n_log_prob',
            'n': 'n',
            'ns_with_mask': 'actor.ns_with_mask'
        })

        self.register_method('optimize', self.optimize, {
            'qs': 'self.qs',
            'td_q_est': 'self.td_q_est',
            'timestep': 'base.time_step',
            'sample_weights': 'actor.sample_weights_ns',
            'ns_with_mask': 'actor.ns_with_mask',
        })


    def td_q_est(self, target_qs, dones, rewards, discount, entropy_coef, log_prob, next_log_prob, n, ns_with_mask):
        ns, ns_mask = ns_with_mask
        discounts = th.unsqueeze(discount ** th.arange(0, n, device=rewards.device), 0)
        log_probs = th.concatenate((log_prob[:, 1:], th.unsqueeze(next_log_prob, axis=-2)), axis=1)
        if self.sample_n_limit or self.sample_n_limit_geom:
            if self.sample_n_limit:
                limit = th.arange(1, n + 1, device=log_prob.device)
            elif self.sample_n_limit_geom:
                limit = th.round((1 - discount ** (2 * th.arange(0, n, device=log_prob.device))) / (1 - discount ** 2))
            lp_mask = th.cumsum(th.ones_like(log_probs), dim=-1) <= th.reshape(limit, (1, -1, 1))
            mean_log_probs = (th.sum(th.unsqueeze(lp_mask, 1) * th.unsqueeze(log_probs, 2), dim=3) / th.unsqueeze(th.sum(lp_mask, dim=2), 1))
            discounted_mean_log_probs = th.unsqueeze(discounts, dim=2) * entropy_coef * mean_log_probs
            cum_discounted_mean_log_probs = th.cumsum(discounted_mean_log_probs, dim=1)
            discounted_log_probs = th.diagonal(cum_discounted_mean_log_probs, dim1=1, dim2=2)
        else:
            log_probs = th.mean(log_probs, -1)
            discounted_log_probs = th.cumsum(discounts * entropy_coef * log_probs, dim = 1)

        cum_discounted_rewards_masked = th.sum(th.unsqueeze(th.cumsum(discounts * rewards, dim=1), 2) * ns_mask, 1)
        cum_discounted_lp_masked = th.sum(th.unsqueeze(discounted_log_probs, 2) * ns_mask, 1)

        discounts_target = discount ** ns

        target = cum_discounted_rewards_masked + ~dones * discount * (discounts_target * target_qs - cum_discounted_lp_masked)
        return target

    def _loss(self, qs, td_q_est, sample_weights, ns_with_mask):
        partial = (qs[:, :1, :] - th.unsqueeze(td_q_est, -1).detach()) ** 2 * th.unsqueeze(sample_weights.detach(), -1)
        if self.lam is not None:
            ns, _ = ns_with_mask
            lams = self.lam ** ns
            lams = th.unsqueeze(lams, -1).detach()
            partial = partial * lams
        if self.sum_loss:
            partial = th.sum(partial, dim=1)
        elif self.lam is not None:
            partial = th.sum(partial, dim=1) / th.sum(lams, 1)
        return th.mean(partial)


class GaussianFastSoftNActor(GaussianSoftNActor):
    @staticmethod
    def get_args(preproc_params, component):
        args = GaussianSoftNActor.get_args(preproc_params, component)
        args['randn'] = (bool, None, {'action': 'store_true'})
        return args

    def __init__(self, *args, randn=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_n_weights = self.scale_weights
        self.scale_weights = False
        self.scale_n_weights_mean = self.scale_weights_mean
        self.scale_weights_mean = False
        self.scale_n_weights_median = self.scale_weights_median
        self.scale_weights_median = False
        self.randn = randn

        if self._ignore_above_b or self._ignore_outside_b:
            self.register_method('last_n', self.last_n, {
                'mask': self.methods['truncated_density'][1]['mask'],
                'lengths': 'lengths',
                'n': 'n'
            })
        else:
            self.register_method('last_n', self.last_n_nomask, {
                'lengths': 'lengths',
                'n': 'n'
            })

        self.register_method('ns_with_mask', self.ns_with_mask, {
            'last_n': 'self.last_n',
            'n': 'n'
        })

        self.register_method('obs_next_ns', self.obs_next_ns, {
            'obs': 'obs',
            'obs_next': 'obs_next',
            'ns_with_mask': 'self.ns_with_mask'
        })

        self.register_method('sampled_next_actions_ns', self.acts_next_ns, {
            'actions': 'self.sampled_actions',
            'actions_next': 'self.next_actions',
            'ns_with_mask': 'self.ns_with_mask'
        })

        self.register_method('dones_ns', self.dones_ns, {
            'dones': 'dones',
            'lengths': 'lengths',
            'last_n': 'self.last_n'
        })

        self.register_method('sample_weights_ns', self.weights_ns, {
            'weights': 'self.sample_weights',
            'ns_with_mask': 'self.ns_with_mask'
        })

    def last_n(self, mask, lengths, n):
        with th.no_grad():
            mask = th.concat((th.ones_like(mask[:, :1]), mask[:, :-1]), axis=1)
            n_vals = th.unsqueeze(th.arange(0, n, dtype=th.int32, device=mask.device), 0)
            lens_mask = (n_vals < th.unsqueeze(lengths, 1)).to(th.float32)
            cummask = th.cumsum(th.cumprod(mask * lens_mask, dim=1), dim=1)
            last = th.argmax(cummask, dim=1)

            if self.randn:
                nmult = n
                for f in range(2, n):
                    nmult *= f
                return th.randint_like(last, 1, nmult) % (last + 1)
            return last

    def last_n_nomask(self, lengths, n):
        with th.no_grad():
            n_vals = th.unsqueeze(th.arange(0, n, dtype=th.int32, device=lengths.device), 0)
            lens_mask = (n_vals < th.unsqueeze(lengths, 1)).to(th.float32)
            cummask = th.cumsum(th.cumprod(lens_mask, dim=1), dim=1)
            last = th.argmax(cummask, dim=1)

            if self.randn:
                nmult = n
                for f in range(2, n):
                    nmult *= f
                return th.randint_like(last, 1, f) % (last + 1)
            return last

    def ns_with_mask(self, last_n, n):
        zeros = th.zeros_like(last_n)
        n_vals = th.unsqueeze(th.arange(0, n, dtype=th.int32, device=last_n.device), 0)
        ns = th.stack((zeros, last_n), dim=1)
        mask = th.stack((n_vals == th.unsqueeze(zeros, 1), n_vals == th.unsqueeze(last_n, 1)), dim=2)
        return ns, mask
    
    def obs_ns(self, obs, ns_with_mask):
        ns_mask = ns_with_mask[1]
        return th.sum(th.unsqueeze(obs, 2) * th.unsqueeze(ns_mask, -1), 1)

    def obs_next_ns(self, obs, obs_next, ns_with_mask):
        ns_mask = ns_with_mask[1]
        return th.sum(th.unsqueeze(th.concat((obs[:, 1:], th.unsqueeze(obs_next, 1)), 1), 2) * th.unsqueeze(ns_mask, -1), 1)

    def acts_next_ns(self, actions, actions_next, ns_with_mask):
        ns_mask = ns_with_mask[1]
        return th.sum(th.unsqueeze(th.concat((actions[:, 1:], th.unsqueeze(actions_next, 1)), 1), 2) * th.unsqueeze(ns_mask, -1), 1)

    def dones_ns(self, dones, lengths, last_n):
        dones_1 = (lengths == 1) * dones
        dones_n = (lengths == last_n) * dones
        return th.stack((dones_1, dones_n), 1)

    def weights_ns(self, weights, ns_with_mask):
        weights = th.concat((th.ones_like(weights[:, :1]), weights[:, :-1]), dim=1)

        if self.scale_n_weights:
            weights = th.nan_to_num(weights / th.max(weights, dim=0, keepdim=True)[0])
        if self.scale_n_weights_mean:
            weights = th.nan_to_num(weights / th.mean(weights, dim=0, keepdim=True))
        if self.scale_n_weights_median:
            weights = th.nan_to_num(weights / th.median(weights, dim=0, keepdim=True))

        ns_mask = ns_with_mask[1]
        return th.sum(th.unsqueeze(weights, 2) * ns_mask, 1)

class FastSACN(SACN):
    ACTORS = {'simple': {False: GaussianFastSoftNActor}}
    CRITICS = {'simple': TwinQDelayedFastSoftNCritic}
