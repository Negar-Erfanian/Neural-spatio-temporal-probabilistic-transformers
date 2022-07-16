import tensorflow as tf
import random


import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


class HomogeneousPoissonPointProcess(tf.keras.Model):

    def __init__(self):
        super(HomogeneousPoissonPointProcess, self).__init__()
        self.lamb = tf.Variable(tf.random.normal(1) * 0.2)

    def call(self, input_time):
        lamb = tf.nn.softplus(self.lamb)
        N, T, _ = input_time.shape
        input_time = tf.reshape(input_time, (N, T))
        compensator = (input_time[:, -1] - input_time[:, 0]) * lamb
        dist = lamb*tf.math.exp(-compensator)
        loglik = tf.math.log(lamb + 1e-20) - compensator
        return loglik, dist, lamb  # (N)

    def predict(self, input_time, output_time):
        _,_, lamb = self.call(input_time)
        num_output = output_time.shape[1]
        input_time = tf.squeeze(input_time, -1)
        output_time = tf.squeeze(output_time, -1)
        expected = []
        for i in range(num_output):
            t_range = tf.linspace(input_time[:,-1], output_time[:,-1], 1000, axis=1)
            predicted = tf.reduce_sum(t_range * lamb, -1)
            expected.append(predicted)
            input_time = tf.concat([input_time, predicted], -1)
        return expected










class HawkesPointProcess(tf.keras.Model):

    def __init__(self):
        super(HawkesPointProcess, self).__init__()

        self.mu = tf.Variable(tf.random.normal(1) * 0.5 - 2.0)
        self.alpha = tf.Variable(tf.random.normal(1) * 0.5 - 3.0)
        self.beta = tf.Variable(tf.random.normal(1) * 0.5)

    def call(self, input_time):
        mu = tf.nn.softplus(self.mu)
        alpha = tf.nn.softplus(self.alpha)
        beta = tf.nn.softplus(self.beta)


        N, T, _ = input_time.shape
        dt = input_time - tf.reshape(input_time, (N, 1, T))  # (N, T, T)
        input_time = tf.reshape(input_time, (N,T))
        dt = fill_triu(-dt * beta, -1e20)
        lamb = tf.math.exp(tf.math.reduce_logsumexp(dt, dim=-1)) * alpha + mu  # (N, T)
        loglik = tf.reduce_sum(tf.math.log(lamb + 1e-8), -1)  # (N,)

        log_kernel = -beta * (input_time[:, -1] - input_time)

        compensator = (input_time[:, -1] - input_time[:, 0]) * mu
        compensator = compensator - alpha / beta * (tf.math.exp(tf.math.reduce_logsumexp(log_kernel, dim=-1)))
        dist = lamb*tf.math.exp(-compensator)
        return (loglik - compensator), dist, lamb  # (N,)

    def predict(self, input_time, output_time):
        return predict_hawkes(t_range, self.beta, self.alpha, self.mu)

        num_output = output_time.shape[1]
        input_time = tf.squeeze(input_time, -1)
        output_time = tf.squeeze(output_time, -1)
        expected = []
        for i in range(num_output):
            t_range = tf.linspace(input_time[:, -1], output_time[:, -1], 1000, axis=1)
            predicted = predict_hawkes(t_range, self.beta, self.alpha, self.mu)
            expected.append(predicted)
            input_time = tf.concat([input_time, predicted], -1)
        return expected


class SelfCorrectingPointProcess(tf.keras.Model):

    def __init__(self):
        super(SelfCorrectingPointProcess, self).__init__()

        self.mu = tf.Variable(tf.random.normal(1) * 0.5 - 2.0)
        self.beta = tf.Variable(tf.random.normal(1) * 0.5)

    def call(self, inputs):
        input_time, input_loc, input_mag, input_timediff = inputs

        N, T, _ = input_time.shape
        input_time = tf.reshape(input_time, (N,T))
        mu = tf.nn.softplus(self.mu)
        beta = tf.nn.softplus(self.beta)

        betaN = beta * tf.cast(tf.broadcast_to(tf.experimental.numpy.arange(T).reshape(1, T), (N, T)),
                               beta.dtype)  # (N, T)

        loglik = mu * input_time - betaN  # (N, T)
        lamb = tf.math.exp(loglik)
        loglik = tf.reduce_sum(loglik, -1)  # (N,)

        t0_i = input_time[:, 0]
        N_i = tf.cast(tf.zeros(N), input_time.dtype)
        compensator = tf.cast(tf.zeros(N), input_time.dtype)
        for i in range(1, T):
            t1_i = input_time[:, i]
            compensator = compensator + tf.math.exp(-beta * N_i) / mu * (
                        tf.math.exp(mu * t1_i) - tf.math.exp(mu * t0_i))

            t0_i = input_time[:, i]
            N_i += tf.ones_like(input_time)[:, i]
        compensator = compensator + tf.math.exp(-beta * N_i) / mu * (tf.math.exp(mu * t1) - tf.math.exp(mu * t0_i))
        dist = lamb*tf.math.exp(-compensator)
        return (loglik - compensator), dist, lamb  # (N,)


def lowtri(A):
    return tf.experimental.numpy.tril(A, diagonal=-1)


def fill_triu(A, value):
    A = lowtri(A)
    A = A + tf.experimental.numpy.triu(tf.ones_like(A)) * value
    return A

def predict_hawkes(t_range, beta, alpha, mu):

    dt = t_range[..., tf.newaxis] - t_range[:,tf.newaxis,:]  # (N, T, T)
    dt = fill_triu(-dt * beta, -1e20)
    lamb = tf.math.exp(tf.math.reduce_logsumexp(dt, dim=-1)) * alpha + mu  # (N, T)
    exptected_time = tf.reduce_sum(tf.math.multiply(t_range, lamb), -1)

    return exptected_time