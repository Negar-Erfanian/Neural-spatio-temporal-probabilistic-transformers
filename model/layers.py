import tensorflow as tf
import numpy as np
try:
    from tensorflow.keras.layers import MultiHeadAttention, Dense, Dropout, LayerNormalization, Layer
except ImportError:
    from keras.layers import MultiHeadAttention, Dense, Dropout, LayerNormalization, Layer
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras

from model.functions import positional_encoding, problayer_dec_time, problayer_dec_loc, Coupling
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
initializer = tf.keras.initializers.HeNormal()


class EncoderLayer(Layer):

    def __init__(self, embedding_dim, num_heads, fc_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim,
                                      dropout=dropout_rate)

        self.ffn1 = Dense(fc_dim, activation='elu',kernel_initializer=initializer)  # (batch_size, seq_len, dff)
        self.ffn2 = Dense(embedding_dim,kernel_initializer=initializer)  # (batch_size, seq_len, d_model)

        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, x, training, look_ahead_mask):
        if look_ahead_mask != None:
            attn_output, attn_scores = self.mha(x, x, x, attention_mask=look_ahead_mask, training=training,
                                                return_attention_scores=True)
        else:
            attn_output, attn_scores = self.mha(x, x, x, training=training, return_attention_scores=True)
        Q = self.layernorm1(attn_output + x)

        ffn_output = self.ffn2(self.ffn1(Q))

        out = self.dropout_ffn(ffn_output, training=training)

        encoder_layer_out = self.layernorm2(out + Q)

        return encoder_layer_out, attn_scores


class Encoder(Layer):

    def __init__(self, num_layers, embedding_dim, num_heads, fc_dim, dim_out_time, dim_out_loc,
                 maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding, embedding_dim)

        self.enc_layers_time = [EncoderLayer(embedding_dim=embedding_dim,
                                             num_heads=num_heads,
                                             fc_dim=fc_dim,
                                             dropout_rate=dropout_rate,
                                             layernorm_eps=layernorm_eps)
                                for _ in range(self.num_layers)]

        self.enc_layers_loc = [EncoderLayer(embedding_dim=embedding_dim,
                                            num_heads=num_heads,
                                            fc_dim=fc_dim,
                                            dropout_rate=dropout_rate,
                                            layernorm_eps=layernorm_eps)
                               for _ in range(self.num_layers)]

        self.dense_time = Dense(self.embedding_dim,kernel_initializer=initializer)
        self.dense_loc = Dense(self.embedding_dim,kernel_initializer=initializer)
        self.dense_mag = Dense(self.embedding_dim,kernel_initializer=initializer)

        # time
        self.ffn1 = Dense(fc_dim, activation='elu',kernel_initializer=initializer)  # (batch_size, seq_len, dff)
        self.ffn2 = Dense(dim_out_time,kernel_initializer=initializer)  # (batch_size, seq_len, d_model)
        self.dropout_time = Dropout(dropout_rate)

        # space
        self.ffn3 = Dense(fc_dim, activation='elu',kernel_initializer=initializer)  # (batch_size, seq_len, dff)
        self.ffn4 = Dense(dim_out_loc,kernel_initializer=initializer)  # (batch_size, seq_len, d_model)
        self.dropout_loc = Dropout(dropout_rate)

    def call(self, x, training, look_ahead_mask):

        t, loc, mag, t_diff = x
        seq_len = tf.shape(t)[1]
        t_emb = self.dense_time(t)
        mag_emb = self.dense_mag(mag)
        loc_emb = self.dense_loc(loc)
        t_emb_copy = t_emb
        loc_emb_copy = loc_emb

        # for time

        t_emb *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        t_emb += self.pos_encoding[:, :seq_len, :]
        t_emb += loc_emb_copy
        t_emb += mag_emb
        t_emb = self.dropout_time(t_emb, training=training)
        attention_weights = {}
        for i in range(self.num_layers):
            t_emb, block_time = self.enc_layers_time[i](t_emb, look_ahead_mask=look_ahead_mask, training=training)
            attention_weights['encoder_layer{}_block_time1_att'.format(i + 1)] = block_time

        out_enc_time = self.ffn2(self.ffn1(t_emb))  # final dim is #batch * #seq_len* 1

        # for space

        loc_emb *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        loc_emb += self.pos_encoding[:, :seq_len, :]
        loc_emb += t_emb_copy
        loc_emb += mag_emb
        loc_emb = self.dropout_loc(loc_emb, training=training)

        for i in range(self.num_layers):
            loc_emb, block_time = self.enc_layers_loc[i](loc_emb, look_ahead_mask=look_ahead_mask, training=training)
            attention_weights['encoder_layer{}_block_loc1_att'.format(i + 1)] = block_time

        out_enc_loc = self.ffn4(self.ffn3(loc_emb))  # final dim is #batch * #seq_len* 1

        return out_enc_time, out_enc_loc, attention_weights


class DecoderLayer(Layer):
    """
    The decoder layer is composed by two multi-head attention blocks, 
    one that takes the new input and uses self-attention, and the other 
    one that combines it with the output of the encoder, followed by a
    fully connected block. 
    """

    def __init__(self, embedding_dim, num_heads, fc_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(num_heads=num_heads,
                                       key_dim=embedding_dim,
                                       dropout=dropout_rate)

        self.mha2 = MultiHeadAttention(num_heads=num_heads,
                                       key_dim=embedding_dim,
                                       dropout=dropout_rate)

        self.ffn1 = Dense(fc_dim, activation='elu',kernel_initializer=initializer)  # (batch_size, seq_len, dff)
        self.ffn2 = Dense(embedding_dim,kernel_initializer=initializer)

        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask):
        if look_ahead_mask!= None:
            mult_attn_out1, attn_weights_block1 = \
                self.mha1(x, x, x, training=training, attention_mask=look_ahead_mask, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)  #look_ahead_mask
        else:
            mult_attn_out1, attn_weights_block1 = \
                self.mha1(x, x, x, training=training, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)  #look_ahead_mask

        Q1 = self.layernorm1(mult_attn_out1 + x)

        mult_attn_out2, attn_weights_block2 = \
            self.mha2(Q1, enc_output, enc_output, training=training, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)
        mult_attn_out2 = self.layernorm2(mult_attn_out2 + Q1)  # (batch_size, target_seq_len, fully_connected_dim)

        ffn_output = self.ffn2(self.ffn1(mult_attn_out2))  # (batch_size, target_seq_len, fc_dim)

        ffn_output = self.dropout_ffn(ffn_output, training=training)

        out = self.layernorm3(ffn_output + mult_attn_out2)  # (batch_size, target_seq_len, fully_connected_dim)

        return out, attn_weights_block1, attn_weights_block2


class Decoder(Layer):
    """
    The entire Encoder is starts by passing the target input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    decoder Layers
        
    """

    def __init__(self, num_layers, embedding_dim, num_heads, fc_dim, dim_out_time, dim_out_loc,
                 maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding, embedding_dim)

        self.dec_layers_time = [DecoderLayer(embedding_dim=embedding_dim,
                                             num_heads=num_heads,
                                             fc_dim=fc_dim,
                                             dropout_rate=dropout_rate,
                                             layernorm_eps=layernorm_eps)
                                for _ in range(self.num_layers)]

        self.dec_layers_loc = [DecoderLayer(embedding_dim=embedding_dim,
                                            num_heads=num_heads,
                                            fc_dim=fc_dim,
                                            dropout_rate=dropout_rate,
                                            layernorm_eps=layernorm_eps)
                               for _ in range(self.num_layers)]
        # time
        self.dropout_time = Dropout(dropout_rate)
        self.ffn1 = Dense(fc_dim, activation='elu',kernel_initializer=initializer)  # (batch_size, seq_len, dff)
        self.ffn2 = Dense(dim_out_time,kernel_initializer=initializer)  # for time dim_out should be 4 and for space should be 3

        # space
        self.dropout_loc = Dropout(dropout_rate)
        self.ffn3 = Dense(fc_dim, activation='elu',kernel_initializer=initializer)  # (batch_size, seq_len, dff)
        self.ffn4 = Dense(dim_out_loc,kernel_initializer=initializer)  # for time dim_out should be 4 and for space should be 3

        self.dense_time = Dense(embedding_dim,kernel_initializer=initializer)
        self.dense_mag = Dense(embedding_dim,kernel_initializer=initializer)
        self.dense_loc = Dense(embedding_dim,kernel_initializer=initializer)

    def call(self, x, enc_output, training, look_ahead_mask):
        enc_output_time, enc_output_loc = enc_output
        t, loc, mag, t_diff = x
        seq_len = tf.shape(t)[1]
        t_emb = self.dense_time(t)
        mag_emb = self.dense_mag(mag)
        loc_emb = self.dense_loc(loc)
        t_emb_copy = t_emb
        loc_emb_copy = loc_emb

        # for time

        t_emb *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        t_emb += self.pos_encoding[:, :seq_len, :]
        t_emb += loc_emb_copy
        t_emb += mag_emb
        t_emb = self.dropout_time(t_emb, training=training)
        attention_weights = {}

        for i in range(self.num_layers):
            t_emb, block1, block2 = self.dec_layers_time[i](t_emb, enc_output_time, training,
                                                            look_ahead_mask)

            attention_weights['decoder_layer{}_block1_self_att_time'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2_decenc_att_time'.format(i + 1)] = block2

        out_dec_time = self.ffn2(self.ffn1(t_emb))

        # for space

        loc_emb *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        loc_emb += self.pos_encoding[:, :seq_len, :]
        loc_emb += t_emb_copy
        loc_emb += mag_emb
        loc_emb = self.dropout_loc(loc_emb, training=training)

        for i in range(self.num_layers):
            loc_emb, block1, block2 = self.dec_layers_loc[i](loc_emb, enc_output_loc, training,
                                                             look_ahead_mask)

            attention_weights['decoder_layer{}_block1_self_att_loc'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2_decenc_att_loc'.format(i + 1)] = block2

        out_dec_loc = self.ffn4(self.ffn3(loc_emb))

        return out_dec_time, out_dec_loc, attention_weights


class RealNVP(Layer):

    def __init__(self, num_coupling_layers, input_shape=(None, 3,), dim = 3):
        super(RealNVP, self).__init__()
        self.num_coupling_layers = num_coupling_layers

        if dim == 3:
            self.masks = np.array(
                [[0, 1, 0], [1, 0, 1]] * (num_coupling_layers // 2), dtype="float32"
            )
        elif dim ==2 :
            self.masks = np.array(
                [[0, 1], [1, 0]] * (num_coupling_layers // 2), dtype="float32"
            )
        self.layers_list = [Coupling(input_shape=input_shape) for i in range(num_coupling_layers)]

    def call(self, x, base_dist, training=True, test=False):
        log_det_inv = 0
        direction = 1

        if training:
            direction = -1

        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]

            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                    reversed_mask
                    * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                    + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [-1])

        if test:
            x = x[:, np.newaxis, :, :]
            log_det_inv = log_det_inv[:, np.newaxis, :]

        log_likelihood = base_dist.log_prob(x) + log_det_inv

        return x, log_likelihood

    def sample(self, base_dist, n_sample):
        x = tf.reduce_mean(base_dist.sample(n_sample), 0)
        out, _ = self.call(x, base_dist, training=False)

        return out


class RealNVPbij(Layer):

    def __init__(self, **kwargs):
        super(RealNVPbij, self).__init__(**kwargs)
        self.bijector = tfb.RealNVP(
            num_masked=2,
            shift_and_log_scale_fn=tfb.real_nvp_default_template(hidden_layers=[32, 32])
        )

    def call(self, inputs, base_dist):
        return tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=self.bijector,
        )

    def sample(self, batch_size, base_dist):
        sample = base_dist.sample(batch_size)
        return self.bijector.inverse(sample)


class SoftSignModel(Layer):

    def __init__(self, **kwargs):
        super(SoftSignModel, self).__init__(**kwargs)
        self.bijector1 = tfb.Softsign()

    def call(self, inputs, base_dist):
        return tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=tfb.Invert(self.bijector1),
        )

    def sample(self, batch_size, base_dist):
        sample = base_dist.sample(batch_size)
        return self.bijector1.inverse(sample)


class Problayer(Layer):
    def __init__(self, event_shape, input_shape, variable='time'):
        super(Problayer, self).__init__()

        if variable == 'time':
            self.problayer = problayer_dec_time(event_shape=event_shape, input_shape=input_shape)

        elif variable == 'loc':
            self.problayer = problayer_dec_loc(event_shape=event_shape, input_shape=input_shape)
        else:
            print('unknown variable')

    def call(self, x):
        return self.problayer(x)
