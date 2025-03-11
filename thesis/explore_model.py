"""
Script for exploring and testing different components of the pretrained CoSE model.
"""
import tensorflow as tf
import numpy as np
from undo_data_preprocessing import undo_preprocessing
from stroke_visualization import visualize_stroke

def try_encode_stroke(model, reshaped_ink, stroke_length):
    """Test the encode_stroke signature."""
    encode_stroke = model.signatures["encode_stroke"]
    selected_strokes = tf.gather(reshaped_ink, indices=[0, 1, 3], axis=2)
    encoded_result = encode_stroke(
        input_seq_len=tf.cast(stroke_length, tf.int32),
        input_stroke=selected_strokes
    )
    return encoded_result

def try_decode_stroke(model, embedding_sample, target_seq_len):
    """Test the decode_stroke signature."""
    decode_stroke = model.signatures["decode_stroke"]
    decode_result = decode_stroke(
        embedding_sample=embedding_sample,
        target_seq_len=tf.cast(target_seq_len, tf.int32)
    )
    return decode_result

def try_predict_position(model, input_positions, input_embeddings):
    """Test the predict_position signature."""
    predict_position = model.signatures["predict_position"]
    predict_position_result = predict_position(
        inp_pos=tf.expand_dims(input_positions, axis=0),
        inp_embeddings=tf.expand_dims(input_embeddings, axis=0)
    )
    return predict_position_result

def try_predict_embedding(model, input_position, target_position, input_embedding):
    """Test the predict_embedding signature."""
    predict_embedding = model.signatures["predict_embedding"]
    predict_embedding_result = predict_embedding(
        inp_pos=tf.expand_dims(tf.expand_dims(input_position, axis=0), axis=0),
        target_pos=tf.expand_dims(target_position, axis=0),
        inp_embeddings=tf.expand_dims(tf.expand_dims(input_embedding, axis=0), axis=0)
    )
    return predict_embedding_result

def try_forward_pass(model, input_stroke, input_seq_len, target_seq_len):
    """Test the forward_pass signature."""
    forward_pass = model.signatures["forward_pass"]
    forward_pass_result = forward_pass(
        input_seq_len=tf.cast(input_seq_len, tf.int32),
        input_stroke=input_stroke,
        target_seq_len=tf.cast(target_seq_len, tf.int32)
    )
    return forward_pass_result

def visualize_decoded_stroke(decode_result, stroke_length, original_drawing, statistics_json_path):
    """Helper function to visualize decoded strokes."""
    decoded_stroke = undo_preprocessing(
        decode_result['stroke'], 
        stroke_length, 
        original_drawing,
        statistics_json_path
    )
    
    # Filter out invalid coordinates
    filtered_arrays = []
    for arr in decoded_stroke:
        for sub_arr in arr:
            if np.all(sub_arr >= 0):
                filtered_arrays.append(sub_arr)
    
    filtered_arrays = np.array(filtered_arrays)
    visualize_stroke([filtered_arrays])

if __name__ == "__main__":
    # Load model
    model = tf.saved_model.load("../pretrained_model/saved_model_with_signatures")
    
    # Example usage
    # 1. Try encode_stroke
    encoded_result = try_encode_stroke(model, reshaped_ink, stroke_length)
    print("Encoded result:", encoded_result)
    
    # 2. Try decode_stroke with the encoded result
    decode_result = try_decode_stroke(
        model, 
        tf.expand_dims(encoded_result["embedding_sample"][0, :], axis=0),
        stroke_length[0]
    )
    print("Decode result:", decode_result)
    