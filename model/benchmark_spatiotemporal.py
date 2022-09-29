import tensorflow as tf
import numpy as np


import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


from model.benchmark_temporal import HomogeneousPoissonPointProcess, HawkesPointProcess, SelfCorrectingPointProcess
from model.benchmark_spatial import GaussianMixtureSpatialModel, ConditionalGMM


class Spatiotemporal(tf.keras.Model):

    def __init__(self, temporal_model, spatial_model, dim):
        super(Spatiotemporal, self).__init__()

        if temporal_model == 'Homoppp':
            self.temporal_model = HomogeneousPoissonPointProcess()
        elif temporal_model == 'Hawkesppp':
            self.temporal_model = HawkesPointProcess()
        elif temporal_model == 'Selfppp':
            self.temporal_model = SelfCorrectingPointProcess()
        if spatial_model =='gmm':
            self.spatial_model = GaussianMixtureSpatialModel()
        elif spatial_model == 'condgmm':
            self.spatial_model = ConditionalGMM(dim=dim, hidden_dims=[64, 64, 64], aux_dim=1)
        self.dim = dim
    def call(self, inputs, outputs, training = True):

        input_time, input_loc, input_mag, input_timediff = inputs
        output_time, output_loc, output_mag, output_timediff = outputs
        aux_state = input_time
        #temporal_loglik, lamb = self.temporal_model.call(input_time)
        temporal_loglik, lamb = self.temporal_model.call(output_time)

        spatial_loglik = self.spatial_model.call(input_time, input_loc, aux_state)


        loss_time = -tf.reduce_mean(temporal_loglik)
        loss_space = -tf.reduce_mean(spatial_loglik)
        expected_times = self.temporal_model.predict(input_time, output_time)
        temporal_loglik, lamb = self.temporal_model.call(output_time)
        #expected_loc_func = self.spatial_modelspatial_conditional_logprob_fn(self, true_time, inputs)
        spatial_model = self.spatial_model
        spatial_loglik, expected_locs = spatial_prediction(output_time, input_time, input_loc, output_loc, input_mag, output_mag, spatial_model, self.dim, training = training)
        temporal_loglik, _= self.temporal_model.call(expected_times[..., tf.newaxis])
        expected_locs = tf.convert_to_tensor(expected_locs, dtype=tf.float32)
        expected_locs = tf.reshape(expected_locs, (expected_locs.shape[1], expected_locs.shape[0], -1))
        return loss_time, loss_space, expected_times, expected_locs, temporal_loglik ,spatial_loglik



def spatial_prediction(curr_time, input_time, history_data, expected_data, aux_state_in, aux_state_out, model, dim, training = True):
    N = 25
    if dim ==3:
        N = 10
    plts =  expected_data.shape[1]
    predicted_list = []
    history = []
    for i in range(plts):
        current_time = curr_time[:,i]
        expected_loc = expected_data[:,i,:]
        aux_state = tf.concat([aux_state_in, aux_state_out[:, i, :][..., tf.newaxis]], axis=1)
        aux_state = tf.cast(aux_state, dtype = tf.float32)
        normalized_data = tf.concat([history_data, expected_loc[:,tf.newaxis,:]], axis=1)
        minx, maxx = tf.math.reduce_min(normalized_data[:, 0]), tf.math.reduce_max(normalized_data[:, 0])
        miny, maxy = tf.math.reduce_min(normalized_data[:, 1]), tf.math.reduce_max(normalized_data[:, 1])

        xlim = np.linspace(minx, maxx, N)
        ylim = np.linspace(miny, maxy, N)
        X1, X2 = np.meshgrid(xlim, ylim)
        arr = tf.stack([X1.reshape(-1), X2.reshape(-1)], axis=1)
        if dim ==3:
            minz, maxz = tf.math.reduce_min(normalized_data[:, 2]), tf.math.reduce_max(normalized_data[:, 2])
            zlim = np.linspace(minz, maxz, N)
            X1, X2, X3 = np.meshgrid(xlim, ylim, zlim)
            arr = tf.stack([X1.reshape(-1), X2.reshape(-1), X3.reshape(-1)], axis=1)


        arr = tf.broadcast_to(arr[None], (expected_data.shape[0], *arr.shape))
        arr = arr.reshape(-1, arr.shape[-1])
        loglikelihood_fn = model.spatial_conditional_logprob_fn(current_time, input_time, history_data, aux_state)
        loglikelihood = loglikelihood_fn(arr)
        #norm_loglik = (loglikelihood - np.min(loglikelihood)) / (np.max(loglikelihood) - np.min(loglikelihood))
        if dim == 2:
            predicted_arg = tf.math.argmin(loglikelihood.reshape(expected_data.shape[0], N * N), -1)
            arr_reshaped = arr.reshape(expected_data.shape[0], N * N , -1)
            predicted = []
            for j in range(expected_data.shape[0]):
                predicted.append(arr_reshaped[j, predicted_arg[j], :])
            predicted = tf.concat([predicted], axis = 0)
        elif dim == 3:
            predicted_arg = tf.math.argmin(loglikelihood.reshape(expected_data.shape[0], N * N * N), -1)
            arr_reshaped = arr.reshape(expected_data.shape[0], N * N *N, -1)
            predicted = []
            for j in range(expected_data.shape[0]):
                predicted.append(arr_reshaped[j, predicted_arg[j], :])
            predicted = tf.concat([predicted], axis=0)
        predicted_list.append(predicted)

        history_data = tf.concat([history_data, tf.cast(predicted[:,tf.newaxis, :], dtype = tf.float32)], axis=1)
        history_data = tf.cast(history_data, dtype = tf.float32)
        input_time = tf.concat([input_time, current_time[...,None]], axis=1)
        input_time = tf.cast(input_time, dtype = tf.float32)
        aux_state_in = aux_state
    spatial_loglik = model.call(input_time, history_data, aux_state)
    if not training:
        #history_data = tf.stack(predicted_list, axis=1)
        history_data = expected_data
        history_data = tf.cast(history_data, dtype=tf.float32)
        spatial_loglik = model.call(curr_time, history_data, aux_state)


    return spatial_loglik, predicted_list

