import numpy as np
from scipy.linalg import det, inv
from scipy.special import logsumexp
import tensorflow as tf

def calculate(model, reshaped_ink, stroke_length):
    '''
    Inputs: 
        1) model: pre-trained CoSE model
        2) reshaped_ink: preprocessed (normalized) ink
        3) stroke_length: length for each stroke
    '''
    
    # Load signatures of pre-trained models
    encode_stroke = model.signatures["encode_stroke"]
    predict_position = model.signatures["predict_position"]
    predict_embedding = model.signatures["predict_embedding"]

    # Get encoded vectors for each stroke
    encoded_result = encode_stroke(
        input_seq_len=tf.cast(stroke_length, tf.int32),
        input_stroke=tf.gather(reshaped_ink, indices=[0, 1, 3], axis=2) # x coordinate, y coordinate, and pen state
    )

    encoded_embedding_sample = encoded_result['embedding_sample']

    # Iterate through each stroke in the given drawing (ink) to predict the next stroke
    n_stroke, _, _ = reshaped_ink.shape

    # Initialize an numpy array to store entropy and bhattacharyya distance
    entropy_array = np.zeros(n_stroke, dtype=float)
    bhattacharyya_distance_array = np.zeros(n_stroke, dtype=float)

    for i in range(n_stroke):
        # Predict next starting position
        input_position = tf.expand_dims(reshaped_ink[:i+1, 0, :2], axis=0)
        input_embedding = tf.expand_dims(encoded_embedding_sample[:i+1, :], axis=0)
        predict_position_result = predict_position(
            inp_pos=input_position,
            inp_embeddings=input_embedding
        )
        
        # Predict the embedding of next stroke
        predict_embedding_result = predict_embedding(
            inp_pos=input_position, 
            target_pos=tf.expand_dims(predict_position_result['position_sample'], axis=0),
            inp_embeddings=input_embedding
        )
        
        # Calculate entropy and bhattacharyya distance
        entropy_array[i] = entropy_gmm(predict_embedding_result)
        bhattacharyya_distance_array[i] = bhattacharyya_distance(predict_embedding_result)

    return entropy_array, bhattacharyya_distance_array


def entropy_gmm(predict_result):
    pi = predict_result['pi'].numpy().flatten()
    mus = predict_result['mu'].numpy().squeeze()
    sigmas = [np.diag(sigma) for sigma in predict_result['sigma'].numpy().squeeze()]

    N = len(pi)
    kl_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            diff_mu = mus[j] - mus[i]
            inv_sigma_j = inv(sigmas[j])
            kl_matrix[i, j] = 0.5 * (np.trace(inv_sigma_j @ sigmas[i]) + diff_mu.T @ inv_sigma_j @ diff_mu - mus.shape[1] + np.log(det(sigmas[j]) / det(sigmas[i])))

    log_terms = -0.5 * kl_matrix  # -1/2 factor from the exponent in the entropy formula
    log_sum_exp = logsumexp(log_terms, b=pi, axis=1)
    entropy = -np.sum(pi * log_sum_exp)
    return entropy

def bhattacharyya_distance(predict_result):
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
            bc_values.append(-np.log(bc))  # Convert BC to a distance

    return np.mean(bc_values)