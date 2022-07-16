# Copyright (c) Facebook, Inc. and its affiliates.

import tensorflow as tf


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

    def call(self, inputs):
        """
        Args:
            event_times: (N, T, 1)
            spatial_locations: (N, T, D)

        Returns:
            logprob: (N,)
        """

        input_time, input_loc, input_mag, input_timediff = inputs
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
        dt_logdecay = dt_logdecay - tf.reduce_logsumexp(dt_logdecay, dim=-1, keepdim=True)  # (N, T, 1)
        loglik = tf.reduce_logsumexp(pairwise_logliks + dt_logdecay, dim=-1)  # (N, T)

        return tf.concat([loglik0[..., None], loglik[:, 1:]], dim=1)  # (N, T)

    def spatial_conditional_logprob_fn(self, t, inputs):
        """
        Args:
            t: scalar
            event_times: (T,)
            spatial_locations: (T, D)

        Returns a function that takes locations (N, D) and returns (N,) the logprob at time t.
        """
        input_time, input_loc, input_mag, input_timediff = inputs
        if input_loc is None:
            return lambda s: tf.reduce_sum(gaussian_loglik(s, self.mu0[None], self.logstd0[None]), -1)

        dt = t - tf.squeeze(input_time,-1)
        logweights = tf.nn.log_softmax(-dt / tf.nn.softplus(self.coeff_decay), dim=0)

        def loglikelihood_fn(s):
            loglik = tf.reduce_sum(gaussian_loglik(s[:, None], input_loc[None], self.spatial_logstd), -1)
            return tf.reduce_logsumexp(loglik + logweights[None], dim=1)

        return loglikelihood_fn


def lowtri(A):
    return tf.experimental.numpy.tril(A, diagonal=-1)


def fill_triu(A, value):
    A = lowtri(A)
    A = A + tf.experimental.numpy.triu(tf.ones_like(A)) * value
    return A


def gaussian_loglik(z, mean, log_std):
    mean = mean + tf.Variable(0.)
    log_std = log_std + tf.Variable(0.)
    c = tf.cast(tf.Variable([tf.math.log(2 * tf.math.pi)]), z.dtype)
    inv_sigma = tf.math.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)
