import torch as th
import torch.distributions as dists


class Distribution:
    def __init__(self, *args, **kwargs):
        pass

    def prob(self, val):
        pass

    def log_prob(self, val):
        pass

    def sample(self):
        pass

    def mode(self):
        pass

    def entropy(self):
        pass

    def sample_with_log_prob(self):
        sample = self.sample()
        return sample, self.log_prob(sample)


class MultivariateNormalDiag(Distribution):
    def __init__(self, loc, scale_diag) -> None:
        self._distr = dists.Normal(loc, scale_diag)
        self._scale_diag = scale_diag
        self._loc = loc

    def _sample_normal(self):
        return self._distr.rsample()

    def prob(self, val):
        return th.prod(th.exp(self._distr.log_prob(val)), dim=-1)

    def log_prob(self, val):
        return th.sum(self._distr.log_prob(val), dim=-1)

    def sample(self):
        return self._sample_normal()

    def mode(self):
        return self._loc

    def entropy(self):
        return self._distr.entropy()


class SquashedMultivariateNormalDiag(MultivariateNormalDiag):
    epsilon = 1e-6

    def prob(self, val):
        gaussian = th.atanh(val)
        return MultivariateNormalDiag.prob(self, gaussian) / th.prod(1 - val ** 2 + self.epsilon, dim=-1)

    def log_prob(self, val):
        gaussian = th.atanh(val)
        return self._calc_log_prob(val, gaussian)

    def _calc_log_prob(self, val, gaussian):
        return MultivariateNormalDiag.log_prob(self, gaussian) - th.sum(th.log(1 - val ** 2 + self.epsilon), dim=-1)

    def _sample(self):
        gaussian = self._sample_normal()
        sample = th.tanh(gaussian)
        return gaussian, sample

    def sample(self):
        return self._sample()[1]

    def mode(self):
        return th.tanh(self._loc)

    def sample_with_log_prob(self):
        gaussian, sample = self._sample()
        log_prob = self._calc_log_prob(sample, gaussian)
        return sample, log_prob


DISTRIBUTIONS = {
    'normal': MultivariateNormalDiag,
    'squashed': SquashedMultivariateNormalDiag,
}
