# Copyright (c) Facebook, Inc. and its affiliates.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress INFO logs

import tensorflow as tf
import math


import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
initializer = tf.keras.initializers.HeNormal()

ACTFNS = {
    "softplus": tf.nn.softplus,
    "relu": tf.nn.relu,
    "elu": tf.nn.elu,
}

class GaussianMixtureSpatialModel(tf.keras.Model):
    def __init__(self):
        super(GaussianMixtureSpatialModel, self).__init__()
        self.mu0 = tf.Variable(0.0)
        self.logstd0 = tf.Variable(0.0)
        self.coeff_decay = tf.Variable(0.1)
        self.spatial_logstd = tf.Variable(0.1)

    def call(self, input_time, input_loc, aux_state=None):
        """
        Args:
            input_time: (N, T, 1)
            input_loc: (N, T, D)

        Returns:
            logprob: (N,)
        """
        # Distribution for the first sample (standard Normal).
        s0 = input_loc[:, 0]
        loglik0 = tf.reduce_sum(gaussian_loglik(s0, self.mu0, self.logstd0), axis=-1)  # (N,)

        # Pairwise time deltas (N, T, T) and spatial differences (N, T, T, D)
        N, T, _ = input_time.shape
        dt = input_time - tf.reshape(input_time, (N, 1, T))  # (N, T, T)
        locs = input_loc[:, :, tf.newaxis, :]  # (N, T, 1, D)
        means = input_loc[:, tf.newaxis, :, :]  # (N, 1, T, D)

        # Compute pairwise log-likelihoods
        pairwise_logliks = tf.reduce_sum(gaussian_loglik(locs, means, self.spatial_logstd), axis=-1)  # (N, T, T)
        pairwise_logliks = fill_triu(pairwise_logliks, -1e20)

        # Time-decay coefficients
        dt_logdecay = -dt / tf.nn.softplus(self.coeff_decay)
        dt_logdecay = fill_triu(dt_logdecay, -1e20)

        # Normalize time-decay coefficients
        dt_logdecay = dt_logdecay - tf.reduce_logsumexp(dt_logdecay, axis=-1, keepdims=True)  # (N, T, 1)
        
        # Compute final log-likelihood
        loglik = tf.reduce_logsumexp(pairwise_logliks + dt_logdecay, axis=-1)  # (N, T)
        return tf.concat([loglik0[..., None], loglik[:, 1:]], axis=1)  # (N, T)

    def spatial_conditional_logprob_fn(self, t, input_time, input_loc, aux_state=None):
        """
        Args:
            t: scalar
            input_time: (T,)
            input_loc: (T, D)

        Returns:
            A function that takes locations (N, D) and returns (N,) the logprob at time t.
        """
        if input_loc is None:
            return lambda s: tf.reduce_sum(gaussian_loglik(s, self.mu0[None], self.logstd0[None]), axis=-1)

        dt = t[:, tf.newaxis] - input_time  
        dt = dt.reshape(-1) # (T,)
        logweights = tf.nn.log_softmax(-dt / tf.nn.softplus(self.coeff_decay), axis=0)

        def loglikelihood_fn(s):
            s = tf.cast(s, tf.float32)
            s_tiled = tf.tile(s[:, tf.newaxis, :], [1, input_loc.shape[0], 1])
            loglik = tf.reduce_sum(
                gaussian_loglik(s_tiled.reshape(-1, input_loc.shape[0], input_loc.shape[-1]), input_loc[tf.newaxis,...], self.spatial_logstd), axis=-1)
            return tf.reduce_logsumexp(loglik + logweights[None], axis=-1).reshape(-1)

        return loglikelihood_fn


def gmm_example_usage():
    # Initialize the Gaussian Mixture Spatial Model
    model = GaussianMixtureSpatialModel()

    # Example input data
    N, T, D = 10, 5, 2  # N=batch size, T=time steps, D=dimension of spatial locations
    input_time = tf.linspace(0.0, 10.0, T)  # (T,)
    input_time = tf.reshape(input_time, (1, T, 1))  # (1, T, 1)
    input_time = tf.tile(input_time, [N, 1, 1])  # (N, T, 1)
    
    input_loc = tf.random.normal((N, T, D))  # Random spatial locations (N, T, D)

    # Compute the log-likelihood of the spatial locations given the event times
    logprobs = model(input_time, input_loc)  # (N, T)
    print(f"Log-probabilities of spatial locations: {logprobs.numpy()} and shape is {logprobs.numpy().shape}")

    # Set a future time point `t` and compute the conditional log-probability function
    t = tf.constant([11.0])  # Future time (scalar)
    input_time_future = tf.linspace(0.0, 10.0, T)  # Existing time sequence (T,)
    input_loc_future = tf.random.normal((T, D))  # Random spatial locations (T, D)

    logprob_fn = model.spatial_conditional_logprob_fn(t, input_time_future, input_loc_future)

    # Example new spatial points to evaluate
    new_spatial_points = tf.random.normal((3, D))  # New spatial locations (N=3, D)
    
    
    # Compute the log-probabilities at time `t` for the new spatial points
    logprob_values = logprob_fn(new_spatial_points)  # (N,)
    print(f"Log-probabilities at time {t} for new spatial points: {logprob_values.numpy()}")



class ConditionalGMM(tf.keras.Model):
    def __init__(self, dim=2, hidden_dims=[64, 64, 64], aux_dim=0, n_mixtures=5, actfn="softplus"):
        """
        Initialize the Conditional Gaussian Mixture Model.

        Args:
            dim: Dimensionality of the output (e.g., spatial dimensions).
            hidden_dims: List of integers defining the hidden layer sizes.
            aux_dim: Dimensionality of the auxiliary input. Must be greater than 0.
            n_mixtures: Number of Gaussian mixtures.
            actfn: Activation function to use in the MLP.
        """
        super(ConditionalGMM, self).__init__()
        assert aux_dim > 0, "ConditionalGMM requires aux_dim > 0"
        self.dim = dim
        self.n_mixtures = n_mixtures
        self.aux_dim = aux_dim
        self.gmm_params = mlp(aux_dim, hidden_dims, out_dim=dim * n_mixtures * 3, actfn=actfn)

    def call(self, event_times, spatial_locations, aux_state=None):
        return self._cond_logliks(event_times, spatial_locations, aux_state)

    def _cond_logliks(self, input_time, input_loc, aux_state=None):
        """
        Compute conditional log likelihoods.

        Args:
            input_time: Tensor of shape (N, T, 1).
            input_loc: Tensor of shape (N, T, D).
            aux_state: Tensor of shape (N, T, D_a).

        Returns:
            A tensor of shape (N, T) containing the conditional log probabilities.
        """
        N, T, _ = input_time.shape

        aux_state = tf.reshape(aux_state, (N * T, self.aux_dim))
        params = self.gmm_params(aux_state)
        logpx = tf.reduce_sum(gmm_loglik(input_loc, params), axis=-1)  # (N, T)
        return logpx

    def sample_spatial(self, nsamples, input_time, input_loc, aux_state=None):
        """
        Sample from the spatial distribution.

        Args:
            nsamples: Number of samples to generate.
            input_time: Tensor of shape (N, T, 1).
            input_loc: Tensor of shape (N, T, D).
            aux_state: Tensor of shape (N, T, D_a).

        Returns:
            Samples from the spatial distribution of shape (nsamples, N, T, D).
        """
        N, T, D = input_loc.shape

        aux_state = tf.reshape(aux_state[:, :, -self.aux_dim:], (N * T, self.aux_dim))
        params = self.gmm_params(aux_state)
        params = tf.reshape(params, (-1, self.dim, 3, self.n_mixtures))
        params = tf.broadcast_to(params[None], (nsamples, *params.shape))
        
        samples = gmm_sample(params)
        return tf.reshape(samples, (nsamples, N, T, D))
    
    def spatial_conditional_logprob_fn(self, t, input_time, input_loc, aux_state=None):
        """
        Create a function that computes log probabilities at a specific time.

        Args:
            t: Scalar time.
            input_time: Tensor of shape (T,).
            input_loc: Tensor of shape (T, D).
            aux_state: Tensor of shape (T + 1, D_a).

        Returns:
            A function that takes locations (N, D) and returns (N,) the log probability at time t.
        """
        B, T, D = input_loc.shape

        def loglikelihood_fn(s):
            s = tf.cast(s, tf.float32)

            bsz = s.shape[0]
            bsz_t = bsz // T
            bsz_event_times = tf.broadcast_to(input_time[None], (bsz_t, *input_time.shape))
            bsz_event_times = tf.reshape(bsz_event_times, (bsz, T, 1))

            # Concatenate event times with the new time t
            bsz_event_times = tf.concat([bsz_event_times, tf.fill((bsz, 1, 1), t)], axis=1)
            bsz_spatial_locations = tf.broadcast_to(input_loc[None], (bsz_t, *input_loc.shape))
            bsz_spatial_locations = tf.reshape(bsz_spatial_locations, (bsz, T, D))
            bsz_spatial_locations = tf.concat([bsz_spatial_locations, tf.reshape(s, (bsz, 1, D))], axis=1)

            bsz_aux_state = None
            if aux_state is not None:
                bsz_aux_state = tf.broadcast_to(aux_state[None], (bsz_t, aux_state.shape[0], T + 1, 1))
                bsz_aux_state = tf.reshape(bsz_aux_state, (bsz, T + 1, -1))

            return tf.reduce_sum(self.call(bsz_event_times, bsz_spatial_locations, aux_state=bsz_aux_state), axis=-1)

        return loglikelihood_fn




def gmm_loglik(z, params):
    """
    Compute the log-likelihood of data points under a Gaussian Mixture Model (GMM).

    Args:
        z: A tensor of shape (..., d), where d is the dimension of the data points.
        params: A tensor of shape (..., 3, n_mixtures), where:
            - The first slice (params[..., 0, :]) represents the mixture logits.
            - The second slice (params[..., 1, :]) represents the means of the Gaussians.
            - The third slice (params[..., 2, :]) represents the log standard deviations.

    Returns:
        A tensor containing the log-likelihoods of the data points under the GMM.
    """
    # Reshape params to match the shape of z
    params = tf.reshape(params, (*z.shape, 3, -1))
    
    # Extract mixture logits, means, and log standard deviations
    mix_logits, means, logstds = params[..., 0, :], params[..., 1, :], params[..., 2, :]
    
    # Calculate mixture log probabilities
    mix_logprobs = mix_logits - tf.reduce_logsumexp(mix_logits, axis=-1, keepdims=True)

    # Calculate log probabilities of data points given the mixture parameters
    logprobs = gaussian_loglik(z[..., None], means, logstds)
    
    # Return the log-sum-exp of mixture log probabilities and log probabilities
    return tf.reduce_logsumexp(mix_logprobs + logprobs, axis=-1)


def gmm_sample(params):
    """
    Sample from a Gaussian Mixture Model (GMM).

    Args:
        params: A tensor of shape (-1, 3, n_mixtures), where:
            - The first slice (params[..., 0, :]) represents the mixture logits.
            - The second slice (params[..., 1, :]) represents the means of the Gaussians.
            - The third slice (params[..., 2, :]) represents the log standard deviations.

    Returns:
        A tensor containing the selected samples from the GMM.
    """
    n_mixtures = params.shape[-1]
    params = tf.reshape(params, (-1, 3, n_mixtures))  # Reshape to extract parameters

    # Extract mixture logits, means, and log standard deviations
    mix_logits, means, logstds = params[..., 0, :], params[..., 1, :], params[..., 2, :]

    # Calculate mixture log probabilities
    mix_logprobs = mix_logits - tf.reduce_logsumexp(mix_logits, axis=-1, keepdims=True)

    # Sample from Gaussian distributions
    samples_for_all_clusters = gaussian_sample(means, logstds)  # (-1, n_mixtures)

    ## Sample cluster indices
    # Ensure the number of samples aligns with samples_for_all_clusters
    batch_size = tf.shape(samples_for_all_clusters)[0]
    cluster_samples = tf.random.categorical(tf.nn.softmax(mix_logits), num_samples=1)  # shape: (batch_size, 1)
    cluster_idx = tf.squeeze(cluster_samples, axis=-1)  # shape: (batch_size,)


    # Adjust shapes if necessary
    if samples_for_all_clusters.shape[0] != cluster_idx.shape[0]:
        # Here, you can perform any necessary reshaping
        raise ValueError("Mismatch in shapes: samples_for_all_clusters and cluster_idx must align.")


    # One-hot encode the cluster indices
    cluster_idx = tf.one_hot(tf.cast(cluster_idx, tf.int32), depth=n_mixtures)  # (-1, n_mixtures)

    # Select the samples based on the cluster indices
    selected_sample = tf.reduce_sum(samples_for_all_clusters * tf.cast(cluster_idx, samples_for_all_clusters.dtype), axis=-1)

    return selected_sample




def gaussian_sample(mean, log_std):
    """
    Sample from a Gaussian distribution with the specified mean and log standard deviation.

    Args:
        mean: A tensor representing the mean of the distribution.
        log_std: A tensor representing the log standard deviation of the distribution.

    Returns:
        A tensor containing samples drawn from the Gaussian distribution.
    """
    # Convert inputs to tensors if they are not already
    mean = tf.convert_to_tensor(mean)
    log_std = tf.convert_to_tensor(log_std)
    
    # Calculate standard deviation from log standard deviation
    std_dev = tf.exp(log_std)
    
    # Generate samples from the Gaussian distribution
    z = tf.random.normal(mean.shape) * std_dev + mean
    return z




def mlp(dim=2, hidden_dims=(), out_dim=None, actfn='softplus'):
    """
    Create a multi-layer perceptron (MLP) model.

    Args:
        dim: The input dimension.
        hidden_dims: A tuple of integers representing the number of units in each hidden layer.
        out_dim: The output dimension. Defaults to input dimension if not specified.
        actfn: The activation function to use for hidden layers.
        initializer: The initializer for the weights.

    Returns:
        A Sequential Keras model representing the MLP.
    """
    out_dim = out_dim or dim  # Set output dimension
    layers = []

    # Add hidden layers if specified
    if hidden_dims:
        dims = [dim] + list(hidden_dims)
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(tfk.layers.Dense(d_out, activation=actfn, kernel_initializer=initializer))
    
    # Add output layer
    layers.append(tfk.layers.Dense(out_dim, kernel_initializer=initializer))

    # Return the constructed Sequential model
    return tfk.models.Sequential(layers)




def lowtri(A):
    """
    Extract the lower triangular part of a matrix A, excluding the diagonal.
    
    Args:
        A: A 2D tensor.
    
    Returns:
        A: A tensor containing the lower triangular part of A.
    """
    return tf.experimental.numpy.tril(A, k=-1)



def fill_triu(A, value):
    """
    Fill the upper triangular part of a matrix A with a specified value.

    Args:
        A: A 2D tensor.
        value: The value to fill in the upper triangular part.

    Returns:
        A: The modified tensor with the upper triangular part filled.
    """
    A = lowtri(A)  # Get the lower triangular part of A
    # Create a mask for the upper triangular part
    upper_tri_mask = tf.linalg.band_part(tf.ones_like(A), 0, -1)  # mask for upper triangle
    # A = A + tf.experimental.numpy.triu(tf.ones_like(A)) * value
    A += upper_tri_mask * value  # Fill upper triangular part with the specified value
    return A



def gaussian_loglik(z, mean, log_std):
    # Ensure mean and log_std are tensors
    mean = tf.convert_to_tensor(mean)
    log_std = tf.convert_to_tensor(log_std)
    
    # Calculate the constant term
    c = tf.math.log(2 * math.pi)
    
    # Calculate the inverse of the standard deviation
    inv_sigma = tf.exp(-log_std)
    
    # Compute the log likelihood
    tmp = (z - mean) * inv_sigma
    log_likelihood = -0.5 * (tmp ** 2 + 2 * log_std + c)
    
    return log_likelihood

def condgmm_example_usage():
    model = ConditionalGMM(dim=2, hidden_dims=[64, 64], aux_dim=4, n_mixtures=5)

    # Example input data
    N, T, D = 10, 5, 2  # Number of samples and time steps
    event_times = tf.random.normal((N, T, 1))  # (N, T, 1)
    spatial_locations = tf.random.normal((N, T, D))  # (N, T, D)
    aux_state = tf.random.normal((N, T, 4))  # (N, T, D_a)

    # Call the model to compute log likelihoods
    log_likelihoods = model.call(event_times, spatial_locations, aux_state)
    print(f"Log Likelihoods: {log_likelihoods} and shape is {log_likelihoods.shape}")

    # Sample from the model
    nsamples = 100
    samples = model.sample_spatial(nsamples, event_times, spatial_locations, aux_state)
    print("Samples Shape:", samples.shape)


# Main function for usage example
if __name__ == "__main__":
    
    gmm_example_usage()
    condgmm_example_usage()

    
