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
        self.lamb = tf.Variable(tf.random.normal([1]) * 0.2)

    def call(self, input_time):
        lamb = tf.nn.softplus(self.lamb)
        #print(f'lamb is {lamb}')
        N, T, _ = input_time.shape
        input_time = tf.reshape(input_time, (N, T))
        compensator = (input_time[:, -1] - input_time[:, 0]) * lamb
        loglik = tf.math.log(lamb + 1e-20) - compensator
        return loglik, lamb  # (N)

    def predict(self, input_time, output_time):
        _, lamb = self.call(input_time)
        num_output = output_time.shape[1]
        output_time = tf.squeeze(output_time, -1)
        expected = []
        initial = tf.broadcast_to([0.], [output_time.shape[0]])[..., tf.newaxis]
        for i in range(num_output):
            t_range = tf.linspace(initial[:, -1], output_time[:, -1], 1000, axis=-1)
            predicted = tf.reduce_sum(t_range * lamb, -1)

            expected.append(predicted)
            initial = tf.concat([initial, predicted[..., tf.newaxis]], -1)
        expected = tf.convert_to_tensor(expected)
        return expected










class HawkesPointProcess(tf.keras.Model):

    def __init__(self):
        super(HawkesPointProcess, self).__init__()

        self.mu = tf.Variable(tf.random.normal([1]) * 0.5)
        self.alpha = tf.Variable(tf.random.normal([1]) * 0.5)
        self.beta = tf.Variable(tf.random.normal([1]) * 0.5)

    def call(self, input_time):
        mu = tf.nn.softplus(self.mu)
        alpha = tf.nn.softplus(self.alpha)
        beta = tf.nn.softplus(self.beta)


        N, T, _ = input_time.shape
        dt = input_time - tf.squeeze(input_time[:, tf.newaxis, :], -1) # (N, T, T)

        input_time = tf.reshape(input_time, (N,T))
        dt = fill_triu(-dt * beta, -1e20)
        # print(f'dt is {dt[0]}')
        lamb = tf.nn.softplus(tf.math.exp(tf.math.reduce_logsumexp(dt, axis=-1)) * alpha * beta) + mu # (N, T)
        lamb = tf.divide(
            tf.subtract(lamb, tf.math.reduce_min(lamb, axis=-1, keepdims=True)),
            tf.subtract(tf.math.reduce_max(lamb, axis=-1, keepdims=True),
                        tf.math.reduce_min(lamb, axis=-1, keepdims=True))
        )
        loglik = tf.reduce_sum(tf.math.log(lamb + 1e-8), -1)  # (N,)
        log_kernel = -beta * (input_time[:, -1][...,tf.newaxis] - input_time) # (N,T)

        compensator = (input_time[:, -1] - input_time[:, 0]) * mu
        compensator = compensator + alpha *beta * (tf.math.exp(tf.math.reduce_logsumexp(log_kernel, axis=-1)))
        #print(f'compensator is {compensator} and {loglik}')
        return (loglik - compensator),  lamb  # (N,)

    def predict(self, input_time, output_time):

        num_output = output_time.shape[1]
        input_time = tf.squeeze(input_time, -1)
        output_time = tf.squeeze(output_time, -1)
        expected = []
        initial = tf.broadcast_to([0.], [output_time.shape[0]])[...,tf.newaxis]
        for i in range(num_output):
            t_range = tf.linspace(initial[:, -1], output_time[:, -1], 1000, axis=-1)
            predicted = predict_hawkes(t_range, self.beta, self.alpha, self.mu)
            expected.append(predicted)
            initial = tf.concat([initial, predicted[...,tf.newaxis]], -1)
        expected = tf.convert_to_tensor(expected)
        return expected


class SelfCorrectingPointProcess(tf.keras.Model):

    def __init__(self):
        super(SelfCorrectingPointProcess, self).__init__()

        self.mu = tf.Variable(tf.random.normal([1]) * 0.5 )
        self.beta = tf.Variable(tf.random.normal([1]) * 0.5)

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
    return tf.experimental.numpy.tril(A, k=-1)


def fill_triu(A, value):
    A = lowtri(A)
    A = A + tf.experimental.numpy.triu(tf.ones_like(A)) * value
    return A

def predict_hawkes(t_range, beta, alpha, mu):
    dt = t_range[..., tf.newaxis] - t_range[:,tf.newaxis,:]  # (N, T, T)
    dt = fill_triu(-dt * beta, -1e20)
    #print(f'dt os {dt[0]}')
    lamb = tf.nn.softplus(tf.math.exp(tf.math.reduce_logsumexp(dt, axis=-1)) * alpha *beta) + mu  # (N, T)
    lamb = tf.divide(
        tf.subtract(lamb, tf.math.reduce_min(lamb, axis=-1, keepdims=True)),
        tf.subtract(tf.math.reduce_max(lamb, axis=-1, keepdims=True),
                    tf.math.reduce_min(lamb, axis=-1, keepdims=True))
    )
    exptected_time = tf.reduce_sum(tf.math.multiply(t_range, lamb), -1)

    return exptected_time