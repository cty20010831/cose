"""
This python script is used to perform data preprocessing for CoSE model 
(adopted from the original 'data_scripts/didi_json_to_tfrecords.py').

Converts .ndjson data into a format CoSE expects and stores in tfrecords.
"""
import contextlib
import json
import os
import random
import tensorflow as tf
import numpy as np
import re
import argparse

# def split_and_pad_strokes(stroke_list):
#   '''
#   The final stroke list has a shape of (num_strokes, max_len, 4):
#     1) First dimension represents the total number of strokes
#     2) Second dimension is the maximum length of any stroke in the provided list of strokes
#     3) Third dimension consists of four channels that represent different components of each point in a stroke:
#       Channel 0: x-coordinate of the point.
#       Channel 1: y-coordinate of the point.
#       Channel 2: Timestamp or a sequence index of the point within the stroke.
#       Channel 3: Pen-lift indicator, which is a binary flag indicating whether the current point is the end of a stroke. A value of 1 indicates the end of a stroke; otherwise, it's 0. 
#   '''
#   max_len = np.array([len(stroke[0]) for stroke in stroke_list]).max()
  
#   strokes = []
#   stroke_lengths = []
#   for stroke in stroke_list:
#     stroke_len = len(stroke[0])
#     padded_stroke_with_pen = np.zeros([1, max_len, 4], dtype=np.float32)
#     padded_stroke_with_pen[0, 0:stroke_len, 0] = stroke[0]
#     padded_stroke_with_pen[0, 0:stroke_len, 1] = stroke[1]
#     padded_stroke_with_pen[0, 0:stroke_len, 2] = stroke[2]
#     padded_stroke_with_pen[0, stroke_len - 1, 3] = 1
#     strokes.append(padded_stroke_with_pen)
#     stroke_lengths.append(stroke_len)
  
#   all_strokes = np.concatenate(strokes, axis=0).astype(float)  # (num_strokes, max_len, 4)
#   all_stroke_lengths = np.array(stroke_lengths).astype(int)
#   return all_strokes, all_stroke_lengths

# Debuggin version of the "split_and_pad_strokes" function. 
def split_and_pad_strokes(stroke_list):
    '''
    The final stroke list has a shape of (num_strokes, max_len, 4):
      1) First dimension represents the total number of strokes
      2) Second dimension is the maximum length of any stroke in the provided list of strokes
      3) Third dimension consists of four channels that represent different components of each point in a stroke:
        Channel 0: x-coordinate of the point.
        Channel 1: y-coordinate of the point.
        Channel 2: Timestamp or a sequence index of the point within the stroke.
        Channel 3: Pen-lift indicator, which is a binary flag indicating whether the current point is the end of a stroke. A value of 1 indicates the end of a stroke; otherwise, it's 0. 
    '''
    # print("DEBUG: Received stroke_list with {} strokes".format(len(stroke_list)))
    
    # Check if stroke_list is empty
    if not stroke_list:
        print("DEBUG: Empty stroke_list encountered in split_and_pad_strokes!")
        return np.empty((0, 0, 4), dtype=np.float32), np.array([], dtype=int)
    
    # Debug: Print the length of each stroke
    stroke_lengths = []
    for idx, stroke in enumerate(stroke_list):
        if not stroke or not stroke[0]:
            print("DEBUG: Stroke index {} is empty or missing x-coordinates: {}".format(idx, stroke))
            stroke_lengths.append(0)
        else:
            stroke_length = len(stroke[0])
            stroke_lengths.append(stroke_length)
    
    # If all strokes are empty, return empty arrays
    if all(l == 0 for l in stroke_lengths):
        print("DEBUG: All strokes are empty!")
        return np.empty((0, 0, 4), dtype=np.float32), np.array([], dtype=int)
    
    max_len = np.array([l for l in stroke_lengths if l > 0]).max()
    # print("DEBUG: max_len calculated as ", max_len)
    
    strokes = []
    final_stroke_lengths = []
    for idx, stroke in enumerate(stroke_list):
        stroke_len = len(stroke[0])
        if stroke_len == 0:
            print("DEBUG: Skipping stroke at index {} because it is empty.".format(idx))
            continue  # Skip empty strokes
        padded_stroke_with_pen = np.zeros([1, max_len, 4], dtype=np.float32)
        padded_stroke_with_pen[0, 0:stroke_len, 0] = stroke[0]
        padded_stroke_with_pen[0, 0:stroke_len, 1] = stroke[1]
        padded_stroke_with_pen[0, 0:stroke_len, 2] = stroke[2]
        padded_stroke_with_pen[0, stroke_len - 1, 3] = 1
        strokes.append(padded_stroke_with_pen)
        final_stroke_lengths.append(stroke_len)
    
    if not strokes:
        print("DEBUG: After processing, no valid strokes remain.")
        return np.empty((0, 0, 4), dtype=np.float32), np.array([], dtype=int)
    
    all_strokes = np.concatenate(strokes, axis=0).astype(float)  # (num_strokes, max_len, 4)
    all_stroke_lengths = np.array(final_stroke_lengths).astype(int)
    # print("DEBUG: Final all_strokes shape: ", all_strokes.shape)
    return all_strokes, all_stroke_lengths

def ink_to_tfexample(ink, mode):
  """Takes a LabeledInk and outputs a TF.Example with stroke information.

  Args:
    ink: A JSON array containing the drawing information.
    mode: A string indicating the mode of the data (test or real). If real data, 
      there would be additional field of worker_id

  Returns:
    a Tensorflow Example proto with the drawing data.
  """
  features = {}
  features["key"] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[ink["key"].encode("utf-8")]))
  features["label_id"] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[ink["label_id"].encode("utf-8")]))
  if mode == 'real': 
    features['batch_name'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[ink['batch_name'].encode("utf-8")]))
    features["worker_id"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[ink["worker_id"].encode("utf-8")]))
  
  all_strokes, all_stroke_lengths = split_and_pad_strokes(ink["drawing"])
  features["ink"] = tf.train.Feature(
      float_list=tf.train.FloatList(value=all_strokes.flatten()))
  features["stroke_length"] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=all_stroke_lengths))
  features["shape"] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=all_strokes.shape))
  features["num_strokes"] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=[len(ink["drawing"])]))
  
  example = tf.train.Example(features=tf.train.Features(feature=features))
  return example

def pick_output_shard(num_shards):
  return random.randint(0, num_shards - 1)

# def translate_to_origin(drawing):
#     translated_drawing = []
#     for stroke in drawing:
#         start_x = stroke[0][0]
#         start_y = stroke[1][0]
#         translated_stroke = [
#             [x - start_x for x in stroke[0]],
#             [y - start_y for y in stroke[1]],
#             stroke[2]  # time remains the same
#         ]
#         translated_drawing.append(translated_stroke)
#     return translated_drawing

def translate_t_to_zero(drawing):
    translated_drawing = []
    for stroke in drawing:
        start_t = stroke[2][0]
        translated_stroke = [
            stroke[0],  # x remains the same
            stroke[1],  # y remains the same
            [t - start_t for t in stroke[2]] 
        ]
        translated_drawing.append(translated_stroke)
    return translated_drawing

def size_normalization(drawing):
  def get_bounding_box(drawing):
    minx = 99999
    miny = 99999
    maxx = 0
    maxy = 0

    for s in drawing:
      minx = min(minx, min(s[0]))
      maxx = max(maxx, max(s[0]))
      miny = min(miny, min(s[1]))
      maxy = max(maxy, max(s[1]))
    return (minx, miny, maxx, maxy)

  bb = get_bounding_box(drawing)
  width, height = bb[2] - bb[0], bb[3] - bb[1]
  offset_x, offset_y = bb[0], bb[1]
  if height < 1e-6:
    height = 1

  size_normalized_drawing = [[[(x - offset_x) / height for x in stroke[0]],
                              [(y - offset_y) / height for y in stroke[1]],
                              [t for t in stroke[2]]]
                             for stroke in drawing]

  return size_normalized_drawing

def resample_ink(drawing, timestep):
  def resample_stroke(stroke, timestep):
    def interpolate(t, t_prev, t_next, v0, v1):
      d0 = abs(t-t_prev)
      d1 = abs(t-t_next)
      dist_sum = d0 + d1
      d0 /= dist_sum
      d1 /= dist_sum
      return d1 * v0 + d0 * v1

    x,y,t = stroke
    if len(t) < 3:
      return stroke
    r_x, r_y, r_t = [x[0]], [y[0]], [t[0]]
    final_time = t[-1]
    stroke_time = final_time - t[0]
    necessary_steps = int(stroke_time / timestep)

    i = 1
    current_time = t[i]
    while current_time < final_time:
      current_time += timestep
      while i < len(t) - 1 and current_time > t[i]:
        i += 1
      r_x.append(interpolate(current_time, t[i-1], t[i], x[i-1], x[i]))
      r_y.append(interpolate(current_time, t[i-1], t[i], y[i-1], y[i]))
      r_t.append(interpolate(current_time, t[i-1], t[i], t[i-1], t[i]))
    return [r_x, r_y, r_t]

  resampled = [resample_stroke(s, timestep) for s in drawing]
  return resampled

def preprocess_and_calculate_stats(json_file, timestep):
    all_drawings = []
    with open(json_file, 'r') as file:
        for line in file:
            ink = json.loads(line)
            # processed_drawing = translate_to_origin(ink['drawing'])
            processed_drawing = translate_t_to_zero(ink['drawing'])
            processed_drawing = size_normalization(ink['drawing'])
            processed_drawing = resample_ink(processed_drawing, timestep)
            all_drawings.append(processed_drawing)

    # Calculate stats
    all_x = np.concatenate([np.array(stroke[0]) for drawing in all_drawings for stroke in drawing])
    all_y = np.concatenate([np.array(stroke[1]) for drawing in all_drawings for stroke in drawing])

    mean_x, stddev_x = np.mean(all_x), np.std(all_x)
    mean_y, stddev_y = np.mean(all_y), np.std(all_y)

    return mean_x, stddev_x, mean_y, stddev_y, all_drawings

# Zero-mean unit-variance normalization
def normalize(drawing, mean_x, stddev_x, mean_y, stddev_y):
    for stroke in drawing:
        stroke[0] = [(x - mean_x) / stddev_x for x in stroke[0]]
        stroke[1] = [(y - mean_y) / stddev_y for y in stroke[1]]
    return drawing

@contextlib.contextmanager
def create_tfrecord_writers(output_dir, output_file, num_output_shards):
    # Create a single directory for all data if it does not exist.
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Create writers for the specified number of shards
    writers = [tf.io.TFRecordWriter(f"{output_dir}/{output_file}-{i:05d}-of-{num_output_shards:05d}.tfrecord")
               for i in range(num_output_shards)]
    try:
        yield writers
    finally:
        for writer in writers:
            writer.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
      description='Convert .ndjson data into tfrecords for CoSE model.',
      epilog="""
        Example usage:
        # Process real data
        python3 data_preprocessing.py --mode real

        # Process test data
        python3 data_preprocessing.py --mode test

        # See example usage of this function
        python3 scripts/save_image.py --help
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, default="data", 
                        help='Directory containing the input data files')
    parser.add_argument('--mode', type=str, choices=['test', 'real'], default='real',
                        help='Use test data or real data')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Number of tfrecord shards to create')

    # Get arguments
    args = parser.parse_args()

    # Define constants based on arguments
    DATA_DIR = args.data_dir
    NUM_TFRECORD_SHARDS = args.num_shards

    # Set JSON_FILES and timestep for resampling based on mode
    # Note: timestep shouldn't be too small, as transfomer model can't handle too many tokens
    if args.mode == 'test':
        JSON_FILES = ["raw_cat.ndjson"]
        timestep = 20
    else:
        JSON_FILES = [
          "raw_Group_1_drawings.ndjson", 
          "raw_Group_2_drawings.ndjson", 
          "raw_Group_3_drawings.ndjson"
        ]
        timestep = 0.1 

    for json_file in JSON_FILES:
        i = 0

        # Calculate statistics
        mean_x, stddev_x, mean_y, stddev_y, all_drawings = preprocess_and_calculate_stats(os.path.join(DATA_DIR, json_file), timestep=timestep)
        
        # Save mean and sd of coordinates (across all drawings)
        stats = {
            'mean_x': mean_x,
            'stddev_x': stddev_x,
            'mean_y': mean_y,
            'stddev_y': stddev_y
        }

        text = re.search(r'raw_(.*?)\.ndjson', json_file).group(1)

        with open(os.path.join(DATA_DIR, f"{text}_statistics.json"), 'w') as stats_file:
            json.dump(stats, stats_file)

        with create_tfrecord_writers(DATA_DIR, json_file.split(".")[0], NUM_TFRECORD_SHARDS) as writers:
          with open(os.path.join(DATA_DIR, json_file)) as f:
            print("Processing: %s" % json_file)
            for v, line in enumerate(f, 1):
                # print("Processing line {}".format(v))
                ink = json.loads(line)
                
                if "key" not in ink:
                    ink["key"] = str(hash(str(ink["drawing"])))
                    ink["label_id"] = ink["key"]
                
                # Normalize ink
                ink["drawing"] = normalize(all_drawings[i], mean_x, stddev_x, mean_y, stddev_y)
                
                # Convert to TFRecord
                example = ink_to_tfexample(ink, args.mode)
                
                # Write to a randomly picked shard
                shard_index = pick_output_shard(NUM_TFRECORD_SHARDS)
                writers[shard_index].write(example.SerializeToString())

                i += 1
                if i % 100 == 0:
                    print("# samples ", i)

            print("Finished writing: %s" % json_file)

if __name__ == "__main__":
    main()