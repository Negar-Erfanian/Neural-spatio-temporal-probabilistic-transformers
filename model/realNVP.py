import tensorflow as tf
import tensorflow_probability as tfp
try:
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Conv2D, BatchNormalization
    from tensorflow.keras.optimizers import Adam
except ImportError:
    from keras import Model, Input
    from keras.layers import Conv2D, BatchNormalization
    from keras.optimizers import Adam

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras

from model.functions import get_base_distribution


class Realnvp(tf.keras.Model):

    def __init__(self, nets, nett, masks, event_shape):
        super(Realnvp, self).__init__()
        
        self.base = get_base_distribution(event_shape = event_shape)
        
        self.mask = tf.constant(masks)
        self.t = [nett() for _ in range(len(masks))]
        self.s = [nets() for _ in range(len(masks))]

        
    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * tf.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = tf.zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * tf.exp(-s) + z_
            log_det_J -= tf.reduce_sum(s, dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.base.log_prob(z) + logp
        
    def sample(self, batchSize): 
        z = self.base.sample((batchSize, 1))
        logp = self.base.log_prob(z)
        x = self.g(z)
        return x