# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
# Modified from https://github.com/SudeepDasari/one_shot_transformers/blob/master/hem/models/discrete_logistic.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_one_hot(tensor, n, fill_with=1.0):
    # we perform one hot encoding with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


class DiscreteMixLogistic(torch.distributions.Distribution):
    def __init__(self, mean, log_scale, logit_probs, num_classes=256, log_scale_min=-7.0):
        assert (
            mean.device == log_scale.device and mean.device == logit_probs.device
        ), "all tensors must lie on same device!"
        batch_shape = log_scale.shape[:-1]
        event_shape = mean.shape[len(batch_shape) + 1 :]
        super().__init__(batch_shape, event_shape, None)
        self._mean = mean  # BxTxAxN
        self._log_scale = log_scale  # 1x1xAx1
        self._logit_probs = logit_probs  # BxTx1xN
        self._num_classes = num_classes
        self._log_scale_min = log_scale_min

    def log_prob(self, value):

        # reshape value to match convention
        B, n_mix = value.shape[0], self._log_scale.shape[-1]

        # unpack parameters. (B, T, num_mixtures) x 3
        logit_probs = self._logit_probs.reshape((self._log_scale.shape[0], -1, n_mix))
        means = self._mean.reshape((self._mean.shape[0], -1, n_mix))
        log_scales = torch.clamp(
            self._log_scale.reshape((self._log_scale.shape[0], -1, n_mix)), min=self._log_scale_min
        )

        # B x T x 1 -> B x T x num_mixtures
        y = value.reshape((B, -1, 1))

        centered_y = y - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_y + 1.0 / (self._num_classes - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_y - 1.0 / (self._num_classes - 1))
        cdf_min = torch.sigmoid(min_in)

        # log probability for edge case of 0 (before scaling)
        # equivalent: torch.log(torch.sigmoid(plus_in))
        log_cdf_plus = plus_in - F.softplus(plus_in)

        # log probability for edge case of 255 (before scaling)
        # equivalent: (1 - torch.sigmoid(min_in)).log()
        log_one_minus_cdf_min = -F.softplus(min_in)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        mid_in = inv_stdv * centered_y
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in our code)
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

        # tf equivalent
        """
        log_probs = tf.where(x < -0.999, log_cdf_plus,
                            tf.where(x > 0.999, log_one_minus_cdf_min,
                                    tf.where(cdf_delta > 1e-5,
                                            tf.log(tf.maximum(cdf_delta, 1e-12)),
                                            log_pdf_mid - np.log(127.5))))
        """
        # for num_classes=65536 case? 1e-7? not sure..
        inner_inner_cond = (cdf_delta > 1e-5).float()

        inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (
            1.0 - inner_inner_cond
        ) * (log_pdf_mid - np.log((self._num_classes - 1) / 2))
        inner_cond = (y > 0.999).float()
        inner_out = inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
        cond = (y < -0.999).float()
        log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out

        log_probs = log_probs + F.log_softmax(logit_probs, -1)
        return torch.logsumexp(log_probs, axis=-1).reshape(value.shape)

    def sample(self):

        n_mix = self._log_scale.shape[-1]
        logit_probs = self._logit_probs.reshape((self._log_scale.shape[0], -1, n_mix))

        # sample mixture indicator from softmax
        temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = logit_probs.data - torch.log(-torch.log(temp))
        _, argmax = temp.max(dim=-1)

        # (B, T) -> (B, T, nr_mix)
        one_hot = (
            to_one_hot(argmax, n_mix)
            .unsqueeze(2)
            .repeat(1, 1, self._mean.shape[2], 1)
            .reshape((self._mean.shape[0], -1, n_mix))
        )
        # select logistic parameters
        means = self._mean.reshape((self._mean.shape[0], -1, n_mix))
        means = torch.sum(means * one_hot, dim=-1)
        log_scales = self._log_scale.reshape((self._log_scale.shape[0], -1, n_mix))
        log_scales = torch.sum(log_scales * one_hot, dim=-1)

        # sample from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))

        x = torch.clamp(torch.clamp(x, min=-1.0), max=1.0)

        return x.reshape(self._mean.shape[:-1])

    @property
    def mean(self):
        alphas = F.softmax(self._logit_probs, dim=-1)
        return torch.sum(self._mean * alphas, -1)

    def expand(self, batch_shape, _instance=None):
        """
        Returns a new distribution instance (or populates an existing instance
        provided by a derived class) with batch dimensions expanded to
        `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
        the distribution's parameters. As such, this does not allocate new
        memory for the expanded distribution instance. Additionally,
        this does not repeat any args checking or parameter broadcasting in
        `__init__.py`, when an instance is first created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance: new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            New distribution instance with batch dimensions expanded to
            `batch_size`.
        """
        pass

    @property
    def arg_constraints(self):
        """
        Returns a dictionary from argument names to
        :class:`~torch.distributions.constraints.Constraint` objects that
        should be satisfied by each argument of this distribution. Args that
        are not tensors need not appear in this dict.
        """
        return dict()

    @property
    def support(self):
        """
        Returns a :class:`~torch.distributions.constraints.Constraint` object
        representing this distribution's support.
        """
        raise None

    @property
    def mode(self):
        """
        Returns the mode of the distribution.
        """
        pass

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        pass

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        pass

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        pass

    def cdf(self, value):
        """
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        pass

    def icdf(self, value):
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        pass

    def enumerate_support(self, expand=True):
        """
        Returns tensor containing all values supported by a discrete
        distribution. The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Returns:
            Tensor iterating over dimension 0.
        """
        pass

    def entropy(self):
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        pass

    def perplexity(self):
        """
        Returns perplexity of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        return torch.exp(self.entropy())

    def _extended_shape(self, sample_shape=torch.Size()):
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape (torch.Size): the size of the sample to be drawn.
        """
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return sample_shape + self._batch_shape + self._event_shape
