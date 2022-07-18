import tensorflow as tf


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

    def __init__(self,temporal_model, spatial_model, dim):
        super(Spatiotemporal, self).__init__()

        if temporal_model == 'Homoppp':
            self.temporal_model = HomogeneousPoissonPointProcess()
        elif temporal_model == 'Hawkesppp':
            self.temporal_model = HawkesPointProcess()
        elif temporal_model == 'Selfppp':
            self.temporal_model = SelfCorrectingPointProcess()
        if spatial_model =='gmm':
            self.spatial_model = GaussianMixtureSpatialModel()
        if spatial_model == 'condgmm':
            self.spatial_model = ConditionalGMM(dim=dim, hidden_dims=[64, 64, 64], aux_dim=1)
    def call(self, inputs, outputs):

        input_time, input_loc, input_mag, input_timediff = inputs
        output_time, output_loc, output_mag, output_timediff = outputs
        aux_state = input_mag
        temporal_loglik, lamb = self.temporal_model.call(input_time)

        spatial_loglik = self.spatial_model.call(input_time, input_loc, aux_state)


        loss = -tf.reduce_mean(temporal_loglik)-tf.reduce_mean(spatial_loglik)
        expected_times = self.temporal_model.predict(input_time, output_time)
        #expected_loc_func = self.spatial_modelspatial_conditional_logprob_fn(self, true_time, inputs)
        return loss, expected_times#, expected_loc_func