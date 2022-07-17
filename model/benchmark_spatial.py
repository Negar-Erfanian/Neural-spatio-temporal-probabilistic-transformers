# Copyright (c) Facebook, Inc. and its affiliates.

import tensorflow as tf
import math


import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

class GaussianMixtureSpatialModel(tf.keras.Model):

    def __init__(self):
        super(GaussianMixtureSpatialModel, self).__init__()
        self.mu0 = tf.Variable(0.0)
        self.logstd0 = tf.Variable(0.0)
        self.coeff_decay = tf.Variable(0.1)
        self.spatial_logstd = tf.Variable(0.1)

    def call(self, input_time, input_loc, aux_state=None):
        """
        Args:
            event_times: (N, T, 1)
            spatial_locations: (N, T, D)

        Returns:
            logprob: (N,)
        """

        # Set distribution of first sample to be standard Normal.
        s0 = input_loc[:, 0]
        loglik0 = tf.reduce_sum(gaussian_loglik(s0, self.mu0, self.logstd0), -1)  # (N,)

        # Pair-wise time deltas.
        N, T, _ = input_time.shape
        dt = input_time - tf.reshape(input_time, (N, 1, T))  # (N, T, T)
        locs = input_loc[:,:,tf.newaxis,:]   # (N, T, 1, D)
        means = input_loc[:,tf.newaxis,:,:]  # (N, 1, T, D)

        pairwise_logliks = tf.reduce_sum(gaussian_loglik(locs, means, self.spatial_logstd), -1)  # (N, T, T)
        pairwise_logliks = fill_triu(pairwise_logliks, -1e20)

        dt_logdecay = -dt / tf.nn.softplus(self.coeff_decay)
        dt_logdecay = fill_triu(dt_logdecay, -1e20)

        # Normalize time-decay coefficients.
        dt_logdecay = dt_logdecay - tf.reduce_logsumexp(dt_logdecay, axis=-1, keepdims=True)  # (N, T, 1)
        loglik = tf.reduce_logsumexp(pairwise_logliks + dt_logdecay, axis=-1)  # (N, T)
        return tf.concat([loglik0[..., None], loglik[:, 1:]], axis=1)  # (N, T)

    def spatial_conditional_logprob_fn(self, t, input_time, input_loc):
        print(f'shapes are {t.shape}, {input_time.shape}, {input_loc.shape}')
        """
        Args:
            t: scalar
            input_time: (T,)
            input_loc: (T, D)

        Returns a function that takes locations (N, D) and returns (N,) the logprob at time t.
        """
        if input_loc is None:
            return lambda s: tf.reduce_sum(gaussian_loglik(s, self.mu0[None], self.logstd0[None]), -1)

        dt = t - input_time  #(T,)
        logweights = tf.nn.log_softmax(-dt / tf.nn.softplus(self.coeff_decay), axis=0)

        def loglikelihood_fn(s):
            loglik = tf.reduce_sum(gaussian_loglik(s[:, None], input_loc[None], self.spatial_logstd), -1)
            return tf.reduce_logsumexp(loglik + logweights[None], axis=1)

        return loglikelihood_fn



class ConditionalGMM(tf.keras.Model):


    def __init__(self, dim=2, hidden_dims=[64, 64, 64], aux_dim=0, n_mixtures=5, actfn="softplus"):
        super(ConditionalGMM, self).__init__()
        assert aux_dim, "ConditionalGMM requires aux_dim > 0"
        self.dim = dim
        self.n_mixtures = n_mixtures
        self.aux_dim = aux_dim * 2  # Since SharedHiddenStateSpatiotemporalModel splits the hidden state.
        self.gmm_params = mlp(aux_dim * 2, hidden_dims, out_dim=dim * n_mixtures * 3, actfn=actfn)

    def call(self, event_times, spatial_locations, aux_state=None):
        return self._cond_logliks(event_times, spatial_locations,  aux_state)

    def _cond_logliks(self, input_time, input_loc, aux_state=None):
        """
        Args:
            input_time: (N, T, 1)
            input_loc: (N, T, D)
            aux_state: (N, T, D_a)

        Returns:
            A tensor of shape (N, T) containing the conditional log probabilities.
        """


        N, T, _ = input_time.shape

        aux_state = aux_state[:, :, -self.aux_dim:].reshape(N * T, self.aux_dim)
        params = self.gmm_params(aux_state)
        logpx = tf.reduce_sum(gmm_loglik(input_loc, params), -1)  # (N, T)
        return logpx

    def sample_spatial(self, nsamples, input_time, input_loc, aux_state=None):
        """
        Args:
            nsamples: int
            input_time: (N, T, 1)
            input_loc: (N, T, D)
            aux_state: (N, T, D_a)

        Returns:
            Samples from the spatial distribution at event times, of shape (nsamples, N, T, D).
        """


        N, T, _ = input_time.shape
        D = input_loc.shape[-1]

        aux_state = aux_state[:, :, -self.aux_dim:].reshape(N * T, self.aux_dim)
        params = self.gmm_params(aux_state).reshape(-1, self.dim, 3, self.n_mixtures)
        params = tf.broadcast_to(params[None], (nsamples, *params.shape))
        samples = gmm_sample(params).reshape(nsamples, N, T, D)
        return samples

    def spatial_conditional_logprob_fn(self, t, input_time, input_loc, aux_state=None):
        """
        Args:
            t: scalar
            input_time: (T,)
            input_loc: (T, D)
            aux_state: (T + 1, D_a)

        Returns a function that takes locations (N, D) and returns (N,) the logprob at time t.
        """
        N, T, D = input_loc.shape
        input_time = tf.squeeze(input_time, -1)

        def loglikelihood_fn(s):
            bsz = s.shape[0]
            bsz_event_times = tf.reshape(input_time, (N, bsz, T))
            bsz_event_times = tf.concat([bsz_event_times, tf.cast(tf.ones(N, bsz, 1), bsz_event_times.dtype) * t], axis=1)
            bsz_spatial_locations = tf.broadcast_to(input_loc[tf.newaxis,...], (N, bsz, T, D))
            bsz_spatial_locations = tf.concat([bsz_spatial_locations, s.reshape(bsz, 1, D)[tf.newaxis, ...]], axis=2)

            if aux_state is not None:
                bsz_aux_state = tf.broadcast_to(aux_state.reshape(N, 1, T + 1, -1), (N, bsz, -1, -1))
            else:
                bsz_aux_state = None

            return tf.reduce_sum(self.logprob(bsz_event_times, bsz_spatial_locations, aux_state=bsz_aux_state), 2)  #(N, bsz,1,D)

        return loglikelihood_fn


def gmm_loglik(z, params):
    params = params.reshape(*z.shape, 3, -1)
    mix_logits, means, logstds = params[..., 0, :], params[..., 1, :], params[..., 2, :]
    mix_logprobs = mix_logits - tf.reduce_logsumexp(mix_logits, axis=-1, keepdims=True)
    logprobs = gaussian_loglik(z[..., None], means, logstds)
    return tf.reduce_logsumexp.logsumexp(mix_logprobs + logprobs, axis=-1)


def gmm_sample(params):
    """ params is (-1, 3, n_mixtures) """
    n_mixtures = params.shape[-1]
    params = params.reshape(-1, 3, n_mixtures)
    mix_logits, means, logstds = params[..., 0, :], params[..., 1, :], params[..., 2, :]
    mix_logprobs = mix_logits - tf.reduce_logsumexp(mix_logits, axis=-1, keepdims=True)
    samples_for_all_clusters = gaussian_sample(means, logstds)    # (-1, n_mixtures)
    cluster_idx = tfp.distributions.Multinomial(tf.math.exp(mix_logprobs), 1).reshape(-1)  # (-1,)
    cluster_idx = tf.one_hot(cluster_idx, num_classes=n_mixtures)  # (-1, n_mixtures)
    select_sample = tf.reduce_sum(samples_for_all_clusters * cluster_idx.to(samples_for_all_clusters), dim=-1)
    return select_sample




def gaussian_sample(mean, log_std):
    mean = mean + tf.Variable(0.)
    log_std = log_std + tf.Variable(0.)
    z = tf.random.normal(mean) * tf.math.exp(log_std) + mean
    return z


ACTFNS = {
    "softplus": tf.nn.softplus,
    "relu": tf.nn.relu,
    "elu": tf.nn.elu,
}


def mlp(dim=2, hidden_dims=[64, 64, 64], out_dim=None, actfn="softplus"):
    out_dim = out_dim or dim
    if hidden_dims:
        dims = [dim] + list(hidden_dims)
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(tfk.layers.linear(d_in, d_out))
            layers.append(ACTFNS[actfn]())
        layers.append(tfk.layers.Linear(hidden_dims[-1], out_dim))
    else:
        layers = [tfk.layers.Linear(dim, out_dim)]

    return tfk.models.Sequential(*layers)



def lowtri(A):
    return tf.experimental.numpy.tril(A, k=-1)


def fill_triu(A, value):
    A = lowtri(A)
    A = A + tf.experimental.numpy.triu(tf.ones_like(A)) * value
    return A


def gaussian_loglik(z, mean, log_std):
    mean = mean + tf.Variable(0.)
    log_std = log_std + tf.Variable(0.)
    c = tf.cast(tf.Variable([tf.math.log(2 * math.pi)]), z.dtype)
    inv_sigma = tf.math.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)
