import tensorflow as tf
import numpy as np
from random import shuffle
try:
    from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, CategoryEncoding
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras import regularizers
except ImportError:
    print(0)
    from keras.layers import Dense
    from keras.models import Model, Sequential
    from keras import regularizers
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

initializer = tf.keras.initializers.HeUniform(seed = 1)


def get_angles(pos, k, d):
    """
    Get the angles for the positional encoding
    
    Arguments:
        pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
        k --   Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
        d(integer) -- Encoding size
    
    Returns:
        angles -- (pos, d) numpy array 
    """
    i = tf.experimental.numpy.floor(k//2)
    angles = pos/(1e4)**(2*i/d)
    return angles


def positional_encoding(positions, d):
    """
    Precomputes a matrix with all the positional encodings 
    
    Arguments:
        positions (int) -- Maximum number of positions to be encoded 
        d (int) -- Encoding size 
    
    Returns:
        pos_encoding -- (1, position, d_model) A matrix with the positional encodings
    """
    #print('positions are',positions)
    angle_rads = get_angles(tf.experimental.numpy.arange(positions)[:, tf.newaxis],
                            tf.experimental.numpy.arange(d)[tf.newaxis, :],d)
    
    angle_rads_np=angle_rads.numpy()
    np_slice1 = np.sin(angle_rads_np[:, 0::2])
    np_slice2 = np.cos(angle_rads_np[:, 1::2])

    #some calculations with np_slice

    angle_rads_np[:, 0::2] = np_slice1
    angle_rads_np[:, 1::2] = np_slice2

    #convert back to tensor
    angle_rads=tf.convert_to_tensor(angle_rads_np, dtype=tf.float32)
    pos_encoding = angle_rads[tf.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)



#not used
def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        decoder_token_ids -- (n, m) matrix
    
    Returns:
        mask -- (n, 1, 1, m) binary tensor
    """    
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
    return seq[:, tf.newaxis, :] 


def create_look_ahead_mask(sequence_length):
    """
    Returns an upper triangular matrix filled with ones
    
    Arguments:
        sequence_length -- matrix size
    
    Returns:
        mask -- (size, size) tensor
    """
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask 

#not used
def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead) 
      but it must be broadcastable for addition.

    Arguments:
        q -- query shape == (..., seq_len_q, depth)
        k -- key shape == (..., seq_len_k, depth)
        v -- value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output -- attention_weights
    """
   
    matmul_qk = tf.matmul(q, tf.transpose(k))  # (..., seq_len_q, seq_len_k)

    dk = k.shape[-1]
    scaled_attention_logits = tf.divide(matmul_qk,tf.math.sqrt(dk))

    if mask is not None: # Don't replace this None
        scaled_attention_logits += (1- mask)*-1e9 

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights




def problayer_dec_time(event_shape, input_shape):    
    model = Sequential([
        Dense(units=tfpl.IndependentPoisson.params_size(event_shape), input_shape=input_shape),
        tfpl.DistributionLambda( lambda t: tfd.Independent(
            tfd.Exponential(rate=tf.math.softplus(t)),
            reinterpreted_batch_ndims=1) ,convert_to_tensor_fn=lambda s: s.sample(100)),   #Exponential
    ])
    #print('model.variables in problayer_dec_time is',model.variables)
    
    return model



def problayer_dec_loc(event_shape, input_shape):  
    model = Sequential([
        Dense(units=tfpl.IndependentNormal.params_size(event_shape), input_shape=input_shape),
        tfpl.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc = t[...,:event_shape], scale =tf.math.exp(t[...,event_shape:])),
            reinterpreted_batch_ndims=1),convert_to_tensor_fn=lambda s: s.sample(100)),
    ])

    return model


    

def Coupling(input_shape, output_dim = 256, reg = 0.01):
    input1 = tf.keras.Input(shape=input_shape)

    t_layer_1 = Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(input1)
    
    t_layer_2 = Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(t_layer_1)
    t_layer_3 = Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(t_layer_2)
    t_layer_4 = Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(t_layer_3)
    t_layer_5 = Dense(input_shape[-1], activation="linear", kernel_regularizer=regularizers.l2(reg))(t_layer_4)

    s_layer_1 = Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(input1)
    s_layer_2 = Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(s_layer_1)
    s_layer_3 = Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(s_layer_2)
    s_layer_4 = Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(s_layer_3)
    s_layer_5 = Dense(input_shape[-1], activation="tanh", kernel_regularizer=regularizers.l2(reg))(s_layer_4)

    return Model(inputs=input1, outputs=[s_layer_5, t_layer_5])


def Coupling_list(num_coupling_layers, input_shape):
    return [Coupling(input_shape =input_shape) for i in range(num_coupling_layers)]


#########################not used in this work yet ##########################




    
    
def get_base_distribution(event_shape):
    loc = tf.zeros(event_shape)
    scale_diag = tf.ones(event_shape)
    return tfd.MultivariateNormalDiag(loc = loc
                                      ,scale_diag=  scale_diag)

def make_masked_autoregressive_flow(event_shape, hidden_units = [64, 64], activation = 'elu', conditional = False):
    made = tfb.AutoregressiveNetwork(params = 2, event_shape=[event_shape], hidden_units=hidden_units, activation=activation,)
    return tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)

def get_bijector(event_shape, num_bijectors = 6, activation = 'elu'):
    bijectors = []
    perm = tf.experimental.numpy.arange(event_shape)
    shuffle(perm)
    for i in range(num_bijectors):
        masked_auto_i = make_masked_autoregressive_flow(event_shape = event_shape,hidden_units=[256, 256], activation = activation, conditional = False)
        bijectors.append(masked_auto_i)
        bijectors.append(tfb.Permute(permutation = perm))

    flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
    return flow_bijector

def get_trainable_dist(base_distribution, event_shape, bijector_type = 'MAR', activation = 'elu'):
    
    
    if bijector_type == 'MAR':
    
        bijectors = []
        print(f'event shape in trainable dist is {event_shape}')
        perm = tf.experimental.numpy.arange(event_shape)

        print(perm)
        perm = tf.random.shuffle(perm)
        print(perm)
        for i in range(6):
            masked_auto_i = tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                    params=2, hidden_units=[64,64], activation='elu'
                )
            )
            bijectors.append(masked_auto_i)
            bijectors.append(tfb.Permute(permutation = perm))
        flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        
    elif bijector_type == 'realnvp':
        
        bijectors = []
        perm = tf.experimental.numpy.arange(event_shape)

        print(perm)
        perm = tf.random.shuffle(perm)
        print(perm)
        for i in range(6):
            masked_auto_i = tfb.RealNVP(num_masked=1,
                                    shift_and_log_scale_fn=
                                    tfb.real_nvp_default_template(hidden_layers=[512, 512]))

            bijectors.append(masked_auto_i)
            bijectors.append(tfb.Permute(permutation = perm))
        flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))

    
    trainable_distribution = tfd.TransformedDistribution(base_distribution, flow_bijector)
    return trainable_distribution
