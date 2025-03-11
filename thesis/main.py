'''
This python script is the main script for calculating the flexbility metrics 
utilizing the CoSE model for the incomplete shape drawings.
'''
import tensorflow as tf
import argparse
import numpy as np
import csv

from calculate_flexibility import (
    calculate_flexibility_measures,
    calculate_inflection_proportion
)

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
          "raw_group_A_drawings.ndjson", 
          "raw_group_B_drawings.ndjson", 
          "raw_group_C_drawings.ndjson"
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
            parts = drawing_name.split("_")
            group_label = "Incomplete_Group_" + parts[2]         

            tfrecord_path = f'data/{drawing_name}-00000-of-00001.tfrecord'
            raw_drawing_dataset = tf.data.TFRecordDataset(tfrecord_path)
            parsed_drawing_dataset = raw_drawing_dataset.map(_parse_function)

            # Just test the first drawing
            # for parsed_features in parsed_drawing_dataset.take(1):
            # Iterate over the dataset
            for parsed_features in parsed_drawing_dataset:
                if args.mode == 'real':
                    worker_id = parsed_features['worker_id'].numpy().decode('utf-8')
                    # print("Worker ID: ", worker_id)
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
                    # Otherwise, calculate
                    entropy_array, bhattacharyya_distance_array = calculate_flexibility_measures(
                        model, reshaped_ink, stroke_length
                    )

                    # Calculate the average entropy and Bhattacharyya distance
                    avg_entropy = np.mean(entropy_array)
                    # print("Average Entropy:", avg_entropy)  
                    avg_bhatt_dist = np.mean(bhattacharyya_distance_array)
                    # print("Average Bhattacharyya Distance:", avg_bhatt_dist)

                    # Calcualte the proportion of inflection points
                    entropy_proportion_inflection = calculate_inflection_proportion(entropy_array)
                    # print("Proportion of inflection points for entropy array:", entropy_proportion_inflection)
                    bhatt_proportion_inflection = calculate_inflection_proportion(bhattacharyya_distance_array)
                    # print("Proportion of inflection points for Bhattacharyya distance array:", bhatt_proportion_inflection)

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