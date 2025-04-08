'''
This python script is the main script for calculating the flexbility metrics 
utilizing the CoSE model for the incomplete shape drawings.
'''
import tensorflow as tf
import argparse
import numpy as np
import csv
import re

from calculate_flexibility import (
    calculate_flexibility_measures,
    calculate_inflection_proportion
)

# Helper to find the first stroke where any t > 0 (participant stroke)
def first_participant_index(ink_array):
    for i, stroke in enumerate(ink_array):
        if np.any(stroke[:, 2] > 0):
            return i
    return len(ink_array)

# Define the feature structure of TF_example
def _parse_function(example_proto, mode):
    if mode == "test":
        feature_description = {
            'key': tf.io.FixedLenFeature([], tf.string),
            'label_id': tf.io.FixedLenFeature([], tf.string),
            'ink': tf.io.VarLenFeature(tf.float32),
            'stroke_length': tf.io.VarLenFeature(tf.int64),
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'num_strokes': tf.io.FixedLenFeature([], tf.int64)
        }
    elif mode == "real":
        feature_description = {
            'key': tf.io.FixedLenFeature([], tf.string),
            'label_id': tf.io.FixedLenFeature([], tf.string),
            'batch_name': tf.io.FixedLenFeature([], tf.string),
            'worker_id': tf.io.FixedLenFeature([], tf.string),
            'ink': tf.io.VarLenFeature(tf.float32),
            'stroke_length': tf.io.VarLenFeature(tf.int64),
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'num_strokes': tf.io.FixedLenFeature([], tf.int64)
        }

    # Parse the input tf.train.Example proto based on feature_description
    return tf.io.parse_single_example(example_proto, feature_description)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert .ndjson data into tfrecords for CoSE model.')
    parser.add_argument('--data_dir', type=str, default="data", 
                        help='Directory containing the input data files')
    parser.add_argument('--mode', type=str, choices=['test', 'real'], default='real',
                        help='Use test data or real data')

    # Get arguments
    args = parser.parse_args()

    # Define constants based on arguments
    DATA_DIR = args.data_dir

    # Set JSON_FILES based on mode
    if args.mode == 'test':
        JSON_FILES = ["raw_cat.ndjson"]
    else:
        JSON_FILES = [
          "raw_Group_1_drawings.ndjson", 
          "raw_Group_2_drawings.ndjson", 
          "raw_Group_3_drawings.ndjson"
        ]
    
    # Load pre-trained CoSE model
    model = tf.saved_model.load("../pretrained_model/saved_model_with_signatures")
    print("Pre-tained model loaded successfully.")

    # Prepare an output CSV
    output_csv = "CoSE_output.csv"
    if args.mode == 'test':
        fieldnames = [
            "drawing_group", 
            "avg_entropy", 
            "avg_bhatt_dist", 
            "inflection_prop_entropy", 
            "inflection_prop_bhatt"
        ]
    else:
        fieldnames = [
            "batch_name",
            "worker_id", 
            "drawing_group", 
            "avg_entropy", 
            "avg_bhatt_dist", 
            "inflection_prop_entropy", 
            "inflection_prop_bhatt"
        ]

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
        # Iterate over the drawing files
        for f in JSON_FILES:
            drawing_name = f.split(".")[0]
            group_label = re.search(r"Group_(\d+)", drawing_name).group(0)   

            tfrecord_path = f'data/{drawing_name}-00000-of-00001.tfrecord'
            raw_drawing_dataset = tf.data.TFRecordDataset(tfrecord_path)
            parsed_drawing_dataset = raw_drawing_dataset.map(lambda x: _parse_function(x, mode=args.mode))

            # Just test the first drawing
            # for parsed_features in parsed_drawing_dataset.take(1):
            # Iterate over the dataset
            for parsed_features in parsed_drawing_dataset:
                if args.mode == 'real':
                    batch_name = parsed_features['batch_name'].numpy().decode('utf-8')
                    worker_id = parsed_features['worker_id'].numpy().decode('utf-8')
                ink = tf.sparse.to_dense(parsed_features['ink'])
                # print("Drawing: ", ink)
                shape = parsed_features['shape']
                # print("Shape of Drawing: ", shape)
                stroke_length = tf.sparse.to_dense(parsed_features['stroke_length'])
                # print("Stroke length: ", stroke_length)
                
                # Reshape ink data to original dimensions (num_strokes, max_len, 4)
                reshaped_ink = tf.reshape(ink, shape)

                # Check if the reshaped_ink has zero strokes or zero time steps
                if reshaped_ink.shape[0] == 0 or reshaped_ink.shape[1] == 0:
                    print("Encountered empty drawing")
                    avg_entropy, avg_bhatt_dist, entropy_proportion_inflection, bhatt_proportion_inflection = 0, 0, 0, 0
                else:
                    # Otherwise, calculate flexibility metrics
                    entropy_array, bhattacharyya_distance_array = calculate_flexibility_measures(
                        model, reshaped_ink, stroke_length
                    )

                    # Only include values derived from the participant strokes (not the background)
                    participant_start_idx = first_participant_index(reshaped_ink.numpy())

                    # Only use participant-derived flexibility scores
                    participant_entropy = entropy_array[participant_start_idx:]
                    participant_bhatt = bhattacharyya_distance_array[participant_start_idx:]

                    # Calculate final flexibility metrics (participant only)
                    avg_entropy = np.mean(participant_entropy)
                    avg_bhatt_dist = np.mean(participant_bhatt)
                    entropy_proportion_inflection = calculate_inflection_proportion(participant_entropy)
                    bhatt_proportion_inflection = calculate_inflection_proportion(participant_bhatt)

                # Write results to CSV
                if args.mode == 'test':
                    row_data = {
                        "drawing_group": group_label,
                        "avg_entropy": avg_entropy,
                        "avg_bhatt_dist": avg_bhatt_dist,
                        "inflection_prop_entropy": entropy_proportion_inflection,
                        "inflection_prop_bhatt": bhatt_proportion_inflection
                    }
                else:
                    row_data = {
                        "batch_name": batch_name,
                        "worker_id": worker_id,
                        "drawing_group": group_label,
                        "avg_entropy": avg_entropy,
                        "avg_bhatt_dist": avg_bhatt_dist,
                        "inflection_prop_entropy": entropy_proportion_inflection,
                        "inflection_prop_bhatt": bhatt_proportion_inflection
                    }
                writer.writerow(row_data)

    print(f"Flexibility metrics saved to {output_csv}")

if __name__ == '__main__':
    main()