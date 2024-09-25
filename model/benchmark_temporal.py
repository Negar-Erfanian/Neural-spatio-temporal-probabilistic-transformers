import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress INFO logs

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
        """
        Computes the log-likelihood of the observed events given the input times.

        Args:
            input_time: Tensor of shape (N, T, 1) representing event times.

        Returns:
            loglik: Tensor of shape (N,) representing the log-likelihood.
            lamb: Tensor representing the intensity rate.
        """
        lamb = tf.nn.softplus(self.lamb)
        N, T, _ = input_time.shape
        input_time = tf.reshape(input_time, (N, T))
        
        # Compute compensator and log-likelihood
        compensator = (input_time[:, -1] - input_time[:, 0]) * lamb
        loglik = tf.math.log(lamb + 1e-20) - compensator
        
        return loglik, lamb  # log-likelihood and intensity

    def predict(self, input_time, output_time):
        """
        Predicts the expected intensity for given output times based on the input event times.

        Args:
            input_time: Tensor of shape (N, T, 1) representing input event times.
            output_time: Tensor of shape (N, M, 1) representing output times.

        Returns:
            expected: Tensor containing the expected intensity values for the output times.
        """
        _, lamb = self.call(input_time)
        num_output = output_time.shape[1]
        output_time = tf.squeeze(output_time, -1)
        
        # Initialize expected intensity list and initial value
        expected = []
        initial = tf.broadcast_to([0.], [output_time.shape[0]])[..., tf.newaxis]
        
        # Predict expected values over output time
        for i in range(num_output):
            t_range = tf.linspace(initial[:, -1], output_time[:, -1], 1000, axis=-1)
            predicted = tf.reduce_sum(t_range * lamb, axis=-1)

            expected.append(predicted)
            initial = tf.concat([initial, predicted[..., tf.newaxis]], axis=-1)
        
        return tf.convert_to_tensor(expected)  # Convert list to tensor


def poisson_process_example_usage():
    # Initialize the Homogeneous Poisson Point Process model
    model = HomogeneousPoissonPointProcess()

    # Example input data
    N, T = 10, 5  # Number of samples and time steps
    # Generate random event times (shape: (N, T, 1))
    event_times = tf.random.normal((N, T, 1)) 

    # Compute the log-likelihood of the observed events
    log_likelihoods, lamb = model.call(event_times)
    print(f"Log Likelihoods: {log_likelihoods.numpy()} and shape is {log_likelihoods.shape}")
    print(f"Intensity Rate (lamb): {lamb.numpy()}")

    # Generate some output times for predictions
    num_output = 3  # Number of output time points to predict
    output_times = tf.random.uniform((N, num_output, 1), minval=0, maxval=10)  # Random output times

    # Predict expected intensities at the output times
    expected_intensities = model.predict(event_times, output_times)
    print(f"Expected Intensities: {expected_intensities.numpy()} and shape is {expected_intensities.shape}")





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
        input_time = tf.reshape(input_time, (N, T))

        # Compute the intensity
        lamb = mu + alpha * tf.reduce_sum(
            tf.exp(-beta * (input_time[:, :, tf.newaxis] - input_time[:, tf.newaxis, :])),
            axis=-1
        )  # (N, T)

        loglik = tf.reduce_sum(tf.math.log(lamb + 1e-8), axis=-1)  # (N,)
        
        # Compute compensator
        compensator = mu * (input_time[:, -1] - input_time[:, 0])  # Compensator for baseline
        compensator += alpha * tf.reduce_sum(
            tf.exp(-beta * (input_time[:, -1][..., tf.newaxis] - input_time)),
            axis=-1
        )  # Adjust for past events

        return (loglik - compensator), lamb  # (N,)

    def predict(self, input_time, output_time):
        num_output = output_time.shape[1]
        output_time = tf.squeeze(output_time, -1)
        expected = []
        initial = tf.broadcast_to([0.], [output_time.shape[0]])[..., tf.newaxis]

        for i in range(num_output):
            t_range = tf.linspace(initial[:, -1], output_time[:, -1], 1000, axis=-1)
            predicted = predict_hawkes(t_range, self.beta, self.alpha, self.mu)
            expected.append(predicted)
            initial = tf.concat([initial, predicted[..., tf.newaxis]], -1)

        expected = tf.convert_to_tensor(expected)
        return expected

    
def Hawkes_process_example_usage():
    # Initialize the Hawkes Point Process model
    model = HawkesPointProcess()

    # Example input data
    N, T = 10, 5  # Number of samples and time steps
    input_time = tf.random.normal((N, T, 1))  # Randomly generated event times (N, T, 1)
    output_time = tf.random.normal((N, 3, 1))  # Future time points for prediction (N, num_outputs, 1)

    # Call the model to compute log-likelihoods
    log_likelihoods, lambdas = model.call(input_time)
    print(f"Log Likelihoods: {log_likelihoods.numpy()} and shape is {log_likelihoods.shape}")
    print(f"Lambdas: {lambdas.numpy()} and shape is {lambdas.shape}")

    # Make predictions
    expected_times = model.predict(input_time, output_time)
    print(f"Expected Times: {expected_times.numpy()} and shape is {expected_times.shape}")


class SelfCorrectingPointProcess(tf.keras.Model):
    def __init__(self):
        super(SelfCorrectingPointProcess, self).__init__()
        self.mu = tf.Variable(tf.random.normal([1]) * 0.5)
        self.beta = tf.Variable(tf.random.normal([1]) * 0.5)

    def call(self, input_time: tf.Tensor) -> tuple:
        N, T, _ = input_time.shape
        input_time = tf.reshape(input_time, (N, T))
        
        mu = tf.nn.softplus(self.mu)
        beta = tf.nn.softplus(self.beta)

        betaN = beta * tf.linspace(input_time[:, 0], input_time[:, -1], T)  # (N, T)
        loglik = input_time / mu - betaN  # (N, T)
        lamb = tf.math.exp(loglik)
        loglik = tf.reduce_sum(loglik, axis=-1)  # (N,)

        # Initialize variables for compensator calculation
        t0_i = input_time[:, 0]  # (N,)
        N_i = tf.zeros(N, dtype=input_time.dtype)  # (N,)
        compensator = tf.zeros(N, dtype=input_time.dtype)  # (N,)

        # Compute the compensator
        for i in range(1, T):
            t1_i = input_time[:, i]  # (N,)
            compensator += tf.math.exp(-beta * N_i) / mu * (
                tf.math.exp(mu * t1_i) - tf.math.exp(mu * t0_i)
            )  # (N,)
            t0_i = t1_i
            N_i += tf.ones(N, dtype=input_time.dtype)  # Increment count

        return (loglik - compensator), lamb  # (N,)

    def predict(self, input_time: tf.Tensor, output_time: tf.Tensor) -> tf.Tensor:
        num_output = output_time.shape[1]
        output_time = tf.squeeze(output_time, axis=-1)
        expected = []
        initial = tf.zeros((output_time.shape[0], 1), dtype=output_time.dtype)

        for _ in range(num_output):
            t_range = tf.linspace(initial[:, -1], output_time[:, -1], 1000)  # Shape (N, 1000)
            N, T = t_range.shape
            input_time_reshaped = tf.reshape(t_range, (N, T))

            mu = tf.nn.softplus(self.mu)
            beta = tf.nn.softplus(self.beta)
            betaN = beta * tf.broadcast_to(tf.range(T, dtype=beta.dtype), (N, T))  # (N, T)

            loglik = mu * input_time_reshaped - betaN  # (N, T)
            lamb = tf.math.exp(loglik)

            # Normalize lamb
            lamb = (lamb - tf.reduce_min(lamb, axis=-1, keepdims=True)) / (
                tf.reduce_max(lamb, axis=-1, keepdims=True) - tf.reduce_min(lamb, axis=-1, keepdims=True)
            )
            predicted = tf.reduce_sum(t_range * lamb, axis=-1)  # Shape (N,)
            expected.append(predicted)
            initial = tf.concat([initial, predicted[..., tf.newaxis]], axis=-1)

        return tf.convert_to_tensor(expected)  # Shape (num_output, N)

    
def SelfCorrecting_process_example_usage():
    # Initialize the Self-Correcting Point Process model
    model = SelfCorrectingPointProcess()

    # Example input data
    N, T = 10, 5  # Number of samples and time steps
    input_time = tf.random.normal((N, T, 1))  # (N, T, 1)

    # Call the model to compute log likelihoods
    log_likelihoods, lambdas = model.call(input_time)
    print(f"Log Likelihoods: {log_likelihoods.numpy()} and shape is {log_likelihoods.shape}")
    print(f"Lambdas: {lambdas.numpy()} and shape is {lambdas.shape}")

    # Prepare output time for predictions
    output_time = tf.random.normal((N, 3, 1))  # (N, num_outputs, 1)

    # Use the model to predict expected values
    expected_values = model.predict(input_time, output_time)
    print(f"Expected Values: {expected_values.numpy()} and shape is {expected_values.shape}")

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

def predict_hawkes(t_range, beta, alpha, mu):
    N = t_range.shape[0]  # Number of samples
    T = t_range.shape[1]  # Number of time points

    # Initialize the expected intensity array
    expected_intensity = tf.zeros((N,))

    for t in range(T):
        # Compute the contribution from each past event
        contribution = alpha * tf.exp(-beta * (t_range[:, t][:, tf.newaxis] - t_range))
        expected_intensity += tf.reduce_sum(contribution, axis=-1)  # Sum contributions

    # Add the baseline intensity
    expected_intensity += tf.nn.softplus(mu)

    return expected_intensity





# Main function for usage example
if __name__ == "__main__":
    #poisson_process_example_usage()
    Hawkes_process_example_usage()
    #SelfCorrecting_process_example_usage()