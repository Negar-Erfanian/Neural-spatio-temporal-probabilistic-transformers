import tensorflow as tf

try:
    from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization, \
        CategoryEncoding
    from tensorflow.keras.models import Model, Sequential
except ImportError:
    from keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization, CategoryEncoding
    from keras.models import Model, Sequential
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras

from model.layers import Encoder, Decoder, RealNVP, Problayer, SoftSignModel

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

initializer = tf.keras.initializers.HeNormal()
class Transformer(Model):
    """
    Complete transformer with an Encoder and a Decoder
    """

    def __init__(self, num_layers, embedding_dim_enc, embedding_dim_dec, num_heads, fc_dim,
                 dim_out_time, dim_out_loc, max_positional_encoding_input,
                 max_positional_encoding_target, time_layer_prob,
                 dropout_rate=0.3, layernorm_eps=1e-6):
        super(Transformer, self).__init__()

        self.fc_dim = fc_dim
        self.max_positional_encoding_input = max_positional_encoding_input

        self.max_positional_encoding_target = max_positional_encoding_target

        self.encoder = Encoder(num_layers=num_layers,
                               embedding_dim=embedding_dim_enc,
                               num_heads=num_heads,
                               fc_dim=fc_dim,
                               dim_out_time=dim_out_time,
                               dim_out_loc=dim_out_loc,
                               maximum_position_encoding=max_positional_encoding_input,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.decoder = Decoder(num_layers=num_layers,
                               embedding_dim=embedding_dim_dec,
                               num_heads=num_heads,
                               fc_dim=fc_dim,
                               dim_out_time=dim_out_time,
                               dim_out_loc=dim_out_loc,
                               maximum_position_encoding=max_positional_encoding_target,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)
        if time_layer_prob == 'exp':
            self.problayer_time = Problayer(event_shape=dim_out_time,
                                            input_shape=(max_positional_encoding_target, dim_out_time,),
                                            variable='time')  # 'time' = exponential, 'loc' = gauss
        else:
            self.problayer_time = Problayer(event_shape=dim_out_time,
                                            input_shape=(max_positional_encoding_target, dim_out_time,),
                                            variable='loc')  # 'time' = exponential, 'loc' = gauss

        self.problayer_loc = Problayer(event_shape=dim_out_loc,
                                       input_shape=(max_positional_encoding_target, dim_out_loc,),
                                       variable='loc')

        self.ffn1 = Dense(fc_dim, activation='elu',kernel_initializer=initializer)
        self.ffn2 = Dense(embedding_dim_dec,kernel_initializer=initializer)
        self.dropout_time1 = Dropout(dropout_rate)
        self.dropout_time2 = Dropout(dropout_rate)
        self.bij_time = SoftSignModel()

        self.ffn3 = Dense(fc_dim, activation='elu',kernel_initializer=initializer)
        self.ffn4 = Dense(embedding_dim_dec,kernel_initializer=initializer)
        self.dropout_loc1 = Dropout(dropout_rate)
        self.dropout_loc2 = Dropout(dropout_rate)

        # self.bij_loc = RealNVPbij()
        self.bij_loc = RealNVP(num_coupling_layers=6,
                               input_shape=(max_positional_encoding_target, dim_out_loc,),
                               dim = dim_out_loc)

    def call(self, inputs, outputs, training, look_ahead_mask_in, look_ahead_mask_out):
        scale = 2.
        input_time, input_loc, input_mag, input_timediff = inputs
        output_time, output_loc, output_mag, output_timediff = outputs

        enc_output_time, enc_output_loc, attention_weights_enc = \
            self.encoder(inputs, training, tf.cast(look_ahead_mask_in, tf.bool))  # final dim for enc_output_time is #batch * #seq_len * 1

        #################################
        # time

        enc_output_time += scale
        enc_output_time += input_timediff
        enc_output_time = self.ffn2(self.ffn1(enc_output_time))
        enc_output_time = self.dropout_time1(enc_output_time, training=training)

        #################################
        # space

        enc_output_loc += scale
        # enc_output_loc += input_time
        enc_output_loc += input_timediff
        enc_output_loc = self.ffn4(self.ffn3(enc_output_loc))
        enc_output_loc = self.dropout_loc1(enc_output_loc, training=training)

        #################################

        output_time_zero = tf.ones_like(output_time) * 1e-9
        output_loc_zero = tf.ones_like(output_loc) * 1e-9
        output_mag_zero = tf.ones_like(output_mag) * 1e-9
        output_timediff_zero = tf.ones_like(output_timediff) * 1e-9
        outputs_zero = [output_time_zero, output_loc_zero, output_mag_zero, output_timediff_zero]
        enc_output = [enc_output_time, enc_output_loc]

        if training:

            dec_output_time, dec_output_loc, attention_weights_dec = \
                self.decoder(outputs, enc_output, training, tf.cast(look_ahead_mask_out, tf.bool))

            #################################

            # time
            dec_output_time += scale
            dec_output_time += output_timediff
            dec_output_time = self.dropout_time2(dec_output_time, training=training)
            probl2_dist_time = self.problayer_time(dec_output_time)
            # print(f'probl2_dist_time is {probl2_dist_time}')
            probl2_bij_time = self.bij_time(output_timediff, probl2_dist_time)
            # print(f'probl2_bij_time is {probl2_bij_time}')
            probl2_output_time = tf.math.abs(tf.reduce_mean(self.bij_time.sample(100, probl2_dist_time), axis=0))

            #################################

            # space

            dec_output_loc += scale
            dec_output_loc += output_timediff
            dec_output_loc = self.dropout_loc2(dec_output_loc, training=training)
            probl2_dist_loc = self.problayer_loc(dec_output_loc)
            _, probl2_bij_loc = self.bij_loc(output_loc, probl2_dist_loc, training=training)
            probl2_output_loc = self.bij_loc.sample(probl2_dist_loc, 100)

        else:

            dec_output_time, dec_output_loc, attention_weights_dec = \
                self.decoder(outputs_zero, enc_output, training, tf.cast(look_ahead_mask_out, tf.bool))

            #################################

            # time
            dec_output_time += scale
            dec_output_time = self.dropout_time2(dec_output_time, training=training)
            probl2_dist_time = self.problayer_time(dec_output_time)
            probl2_output_time = tf.math.abs(tf.reduce_mean(self.bij_time.sample(100, probl2_dist_time), axis=0))
            probl2_bij_time = self.bij_time(None, probl2_dist_time)

            #################################

            # space
            dec_output_loc += scale
            dec_output_loc += output_timediff
            dec_output_loc = self.dropout_loc2(dec_output_loc, training=training)
            probl2_dist_loc = self.problayer_loc(dec_output_loc)
            _, probl2_bij_loc = self.bij_loc(output_loc, probl2_dist_loc, training=True)
            probl2_output_loc = self.bij_loc.sample(probl2_dist_loc, 100)

        return probl2_dist_time, probl2_output_time, probl2_bij_time, probl2_dist_loc, \
               probl2_output_loc, probl2_bij_loc, attention_weights_dec, attention_weights_enc
