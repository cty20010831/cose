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

# Define constants
DATA_DIR = "thesis/data"
JSON_FILES = ["raw_The_Eiffel_Tower.ndjson"]
NUM_TFRECORD_SHARDS = 1

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
  max_len = np.array([len(stroke[0]) for stroke in stroke_list]).max()
  
  strokes = []
  stroke_lengths = []
  for stroke in stroke_list:
    stroke_len = len(stroke[0])
    padded_stroke_with_pen = np.zeros([1, max_len, 4], dtype=np.float32)
    padded_stroke_with_pen[0, 0:stroke_len, 0] = stroke[0]
    padded_stroke_with_pen[0, 0:stroke_len, 1] = stroke[1]
    padded_stroke_with_pen[0, 0:stroke_len, 2] = stroke[2]
    padded_stroke_with_pen[0, stroke_len - 1, 3] = 1
    strokes.append(padded_stroke_with_pen)
    stroke_lengths.append(stroke_len)
  
  all_strokes = np.concatenate(strokes, axis=0).astype(float)  # (num_strokes, max_len, 4)
  all_stroke_lengths = np.array(stroke_lengths).astype(int)
  return all_strokes, all_stroke_lengths


def ink_to_tfexample(ink):
  """Takes a LabeledInk and outputs a TF.Example with stroke information.

  Args:
    ink: A JSON array containing the drawing information.
    dot: (Optional) textual content of the GrahViz dotfile that was used to
      generate the prompt image.

  Returns:
    a Tensorflow Example proto with the drawing data.
  """
  features = {}
  features["key"] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[ink["key"].encode("utf-8")]))
  features["label_id"] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[ink["label_id"].encode("utf-8")]))

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

def didi_preprocess(raw_ink, timestep=20):
  raw_ink = size_normalization(raw_ink)
  raw_ink = resample_ink(raw_ink, timestep)
  return raw_ink

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

for json_file in JSON_FILES:
    i = 0
    # Create writers for all data to be written to a single type of TFRecord files
    with create_tfrecord_writers(DATA_DIR, json_file.split(".")[0], NUM_TFRECORD_SHARDS) as writers:
        with open(os.path.join(DATA_DIR, json_file)) as f:
            for line in f:
                ink = json.loads(line)
                
                if "key" not in ink:
                    ink["key"] = str(hash(str(ink["drawing"])))
                    ink["label_id"] = ink["key"]
                
                # Size normalization and resample ink
                ink["drawing"] = didi_preprocess(ink["drawing"], timestep=20)
                
                # Convert to TFRecord
                example = ink_to_tfexample(ink)
                
                # Write to a randomly picked shard
                shard_index = pick_output_shard(NUM_TFRECORD_SHARDS)
                writers[shard_index].write(example.SerializeToString())  
                
                i += 1
                if i % 100 == 0:
                    print("# samples ", i)

        print("Finished writing: %s" % json_file)
