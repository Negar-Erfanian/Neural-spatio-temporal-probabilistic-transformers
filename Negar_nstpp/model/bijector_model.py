import tensorflow as tf
import numpy as np
try:
    from tensorflow.keras.models import Model, Sequential
except ImportError:
    from keras.models import Model, Sequential
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras

from model.functions import get_trainable_dist, get_base_distribution


class RealNVP(tfb.Bijector):
    
    def __init__(self, num_coupling_layers, ):
        super(RealNVP, self).__init__()
        
        self.num_coupling_layers = num_coupling_layers
        
        
    def _get_mask(self):
        masks = np.array(
            [[0, 1, 0], [1, 0, 1]] * (self.num_coupling_layers // 2), dtype="float32"
        )

        return tf.convert_to_tensor(masks)
    
    
    def _forward(self, x):
        mask = self._get_mask()
        pass
        
        
        
        
    def _inverse(self,z):
        pass
        
        
        
    def _forward_log_det_jacobian(self, x):
        pass
        
        
    def _inverse_log_det_jacobian(self, z):
        pass


class BijectorModel(Model):

    def __init__(self, event_shape, bijector_type = 'MAR'):
        super(BijectorModel, self).__init__()
        
        self.base = get_base_distribution(event_shape = event_shape)
        self.dist = get_trainable_dist(base_distribution = self.base , event_shape = event_shape, bijector_type = bijector_type)
        self._variables = self.dist.trainable_variables
    
    def call(self, input_data):
        
        Log_prob = self.dist.log_prob(input_data)
        #print(Log_prob)
        
        
        return Log_prob, self.dist, self.base
    