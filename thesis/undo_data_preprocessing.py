"""
Undo data preprocessing via "thesis/data_preprocessing.py" by transforming the 
padded, resampled, normalized coordinates back to padded, unnormalized coordinates.

Although it is challenging to undo resampling, it does 

Converts .ndjson data into a format CoSE expects and stores in tfrecords.
"""
import numpy as np

def undo_preprocessing(reshaped_ink, stroke_length, original_drawing):
    '''
    Undo data preprocessing to transform back to padded, unnormalized coordinates.
    Inputs:
        1) reshaped_ink: transformed coordinates of drawings
        2) stroke_length: length of the stroke
        3) original_drawing: drawing in the original ndjson file
    '''
    # Remove padding
    original_strokes = remove_padding(reshaped_ink, stroke_length)

    # Undo normalization
    minx, miny, maxx, maxy = get_bounding_box(original_drawing)
    reversed_strokes = reverse_normalization(original_strokes, minx, miny, maxx, maxy)

    return reversed_strokes


def remove_padding(strokes, stroke_lengths):
    original_strokes = []
    for stroke, length in zip(strokes, stroke_lengths):
        # Extract actual points from each stroke based on its recorded length
        original_strokes.append(stroke[:length])
    return original_strokes

def reverse_normalization(strokes, minx, miny, maxx, maxy):
    # Calculate width and height from the max and min values
    width = maxx - minx
    height = maxy - miny

    reversed_strokes = []
    for stroke in strokes:
        # Adjust coordinates by scaling based on height and shifting by minx and miny
        x = (stroke[:, 0] * height) + minx
        y = (stroke[:, 1] * height) + miny
        reversed_strokes.append(np.stack([x, y], axis=1))
    return reversed_strokes

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