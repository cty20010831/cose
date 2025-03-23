"""
Script for calculating flexibility-related measures using the pretrained CoSE model.
"""
import tensorflow as tf
import numpy as np
from scipy.linalg import det, inv
from scipy.special import logsumexp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def entropy_gmm(predict_result):
    """Calculate entropy for Gaussian Mixture Model prediction."""
    pi = predict_result['pi'].numpy().flatten()
    mus = predict_result['mu'].numpy().squeeze()
    sigmas = [np.diag(sigma) for sigma in predict_result['sigma'].numpy().squeeze()]

    N = len(pi)
    kl_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            diff_mu = mus[j] - mus[i]
            inv_sigma_j = inv(sigmas[j])
            kl_matrix[i, j] = 0.5 * (
                np.trace(inv_sigma_j @ sigmas[i]) + 
                diff_mu.T @ inv_sigma_j @ diff_mu - 
                mus.shape[1] + 
                np.log(det(sigmas[j]) / det(sigmas[i]))
            )

    log_terms = -0.5 * kl_matrix
    log_sum_exp = logsumexp(log_terms, b=pi, axis=1)
    entropy = -np.sum(pi * log_sum_exp)
    return entropy

def bhattacharyya_distance(predict_result):
    """Calculate Bhattacharyya distance between components."""
    pi = predict_result['pi'].numpy().flatten()
    mus = predict_result['mu'].numpy().squeeze()
    sigmas = [np.diag(sigma) for sigma in predict_result['sigma'].numpy().squeeze()]

    N = len(pi)
    bc_values = []

    for i in range(N):
        for j in range(i+1, N):
            sigma_i = sigmas[i]
            sigma_j = sigmas[j]
            sigma_avg = 0.5 * (sigma_i + sigma_j)
            delta_mu = mus[j] - mus[i]
            term = 0.25 * delta_mu.T @ inv(sigma_avg) @ delta_mu
            bc = np.sqrt(det(sigma_i)**0.25 * det(sigma_j)**0.25 / det(sigma_avg)**0.5) * np.exp(-term)
            bc_values.append(-np.log(bc))

    return np.mean(bc_values)

def calculate_flexibility_measures(model, reshaped_ink, stroke_length):
    """Calculate flexibility measures for the given drawing."""
    # Load model signatures
    encode_stroke = model.signatures["encode_stroke"]
    predict_position = model.signatures["predict_position"]
    predict_embedding = model.signatures["predict_embedding"]

    # Get encoded vectors for each stroke
    encoded_result = encode_stroke(
        input_seq_len=tf.cast(stroke_length, tf.int32),
        input_stroke=tf.gather(reshaped_ink, indices=[0, 1, 3], axis=2)
    )
    encoded_embedding_sample = encoded_result['embedding_sample']

    # Initialize arrays for measures
    n_stroke = reshaped_ink.shape[0]
    entropy_array = np.zeros(n_stroke, dtype=float)
    bhattacharyya_distance_array = np.zeros(n_stroke, dtype=float)

    # Calculate measures for each stroke
    for i in range(n_stroke):
        # Predict next starting position
        input_position = tf.expand_dims(reshaped_ink[:i+1, 0, :2], axis=0)
        input_embedding = tf.expand_dims(encoded_embedding_sample[:i+1, :], axis=0)
        
        predict_position_result = predict_position(
            inp_pos=input_position,
            inp_embeddings=input_embedding
        )
        
        # Predict embedding of next stroke
        predict_embedding_result = predict_embedding(
            inp_pos=input_position,
            target_pos=tf.expand_dims(predict_position_result['position_sample'], axis=0),
            inp_embeddings=input_embedding
        )

        # Calculate measures (consider both position and embedding)
        entropy_pos = entropy_gmm(predict_position_result)
        entropy_emb = entropy_gmm(predict_embedding_result)
        bhatta_pos = bhattacharyya_distance(predict_position_result)
        bhatta_emb = bhattacharyya_distance(predict_embedding_result)

        # Sum both sources of uncertainty
        entropy_array[i] = entropy_pos + entropy_emb
        bhattacharyya_distance_array[i] = bhatta_pos + bhatta_emb

    return entropy_array, bhattacharyya_distance_array

def detect_inflection_points(x, y):
    """
    Detects inflection points in a 1D time series by fitting a cubic spline and
    analyzing sign changes in its second derivative.

    Parameters
    ----------
    x : array-like
        1D array of the independent variable (e.g., stroke indices).
    y : array-like
        1D array of the dependent variable (e.g., flexibility metrics).
        Must be the same length as x.

    Returns
    -------
    inflection_points : list of float
        A list of x-values at which the second derivative of the spline
        changes sign, indicating potential inflection points.

    Notes
    -----
    This function:
      1. Creates a cubic spline interpolator from the input data (x, y).
      2. Evaluates the spline and its second derivative on a dense grid of 200 points.
      3. Detects where the second derivative changes sign, indicating an inflection point.
    """
    # 1. Create a Cubic Spline interpolator
    cs = CubicSpline(x, y)

    # 2. Evaluate the spline and its second derivative on a dense grid
    x_dense = np.linspace(x.min(), x.max(), 200)
    y_dense = cs(x_dense)
    y2 = cs(x_dense, 2)  # second derivative

    # 3. Detect where the second derivative changes sign
    inflection_points = []
    for i in range(1, len(y2)):
        # Sign change check: if y2[i-1] and y2[i] have different signs
        if y2[i-1] * y2[i] < 0:
            # (Optional) Linear interpolation to refine the x-coordinate
            # For simplicity, we just record the midpoint of i-1 and i
            x_inflect = (x_dense[i-1] + x_dense[i]) / 2
            inflection_points.append(x_inflect)

    return inflection_points

def calculate_inflection_proportion(time_series):
    """
    Calculate the proportion of inflection points in a time series.
    
    Parameters:
        time_series (np.array): 1D array of the metric (e.g., entropy values over strokes).
        
    Returns:
        proportion (float): Proportion of inflection points in the time series.
    """
    # Define x and y for `detect_inflection_points` fucntion
    x = np.arange(1, len(time_series) + 1)
    y = time_series

    inflection_points = detect_inflection_points(x, y)

    return len(inflection_points) / len(time_series)

def visualize_flexibility_measures(entropy_array, bhattacharyya_distance_array):
    """Visualize flexibility measures using line plots."""
    number_of_strokes = np.arange(1, len(entropy_array) + 1)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
    fig.suptitle('Flexibility Measures per Number of Input Strokes', fontsize=16)

    # Entropy plot
    axes[0].plot(number_of_strokes, entropy_array, 
                 label='Entropy', marker='o', linestyle='-', color='blue')
    axes[0].set_title('Entropy Over Number of Input Strokes')
    axes[0].set_xlabel('Number of Input Strokes')
    axes[0].set_ylabel('Entropy')
    axes[0].grid(True)
    axes[0].legend()

    # Bhattacharyya Distance plot
    axes[1].plot(number_of_strokes, bhattacharyya_distance_array, 
                 label='Bhattacharyya Distance', marker='s', linestyle='--', color='red')
    axes[1].set_title('Bhattacharyya Distance Over Number of Input Strokes')
    axes[1].set_xlabel('Number of Input Strokes')
    axes[1].set_ylabel('Bhattacharyya Distance')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()