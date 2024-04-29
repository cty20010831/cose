# """
# Undo data preprocessing via "thesis/data_preprocessing.py" by transforming the 
# padded, resampled, normalized coordinates back to padded, unnormalized coordinates.

# Although it is challenging to undo resampling, it does 

# Converts .ndjson data into a format CoSE expects and stores in tfrecords.
# """
# import numpy as np
# import json

# def undo_preprocessing(reshaped_ink, stroke_length, raw_drawing, statistics_json_path):
#     '''
#     Undo data preprocessing to transform back to padded, unnormalized coordinates.
#     Inputs:
#         1) reshaped_ink: transformed coordinates of drawings
#         2) stroke_length: length of the stroke
#         3) raw_drawing: line in the ndjson file associated with the 'drawing' key
#         4) statistics_json_path: path to the json file storing mean and sd of coordinates across all drawings
#     '''
#     # Convert tensor to numpy array (and only select x,y coordinates)
#     reshaped_ink = reshaped_ink[:, :, :2].numpy()

#     # Remove padding
#     original_strokes = remove_padding(reshaped_ink, stroke_length)

#     # Undo normalization
#     with open(statistics_json_path, "r") as json_file:
#         stats = json.load(json_file)
#         mean_x = stats['mean_x']
#         stddev_x = stats['stddev_x']
#         mean_y = stats['mean_y']
#         stddev_y = stats['stddev_y']

#     strokes = undo_normalization(original_strokes, mean_x, stddev_x, mean_y, stddev_y)
#     # Rescale the coordinates (based on the canvas)
#     minx, miny, maxx, maxy = get_bounding_box(raw_drawing)
#     strokes = undo_size_normalization(original_strokes, minx, miny, maxx, maxy)
    
#     # Undo translation to origin
#     # strokes = undo_translation(strokes, raw_drawing)
#     return strokes

# def remove_padding(strokes, stroke_lengths):
#     original_strokes = []
#     for stroke, length in zip(strokes, stroke_lengths):
#         # Extract actual points from each stroke based on its recorded length
#         original_strokes.append(stroke[:length])
#     return original_strokes

# def undo_normalization(strokes, mean_x, stddev_x, mean_y, stddev_y):
#     '''
#     Undo zero-mean unit-variance normalization.
#     '''
#     for stroke in strokes:
#         stroke[0] = [x * stddev_x + mean_x for x in stroke[0]]
#         stroke[1] = [y * stddev_y + mean_y for y in stroke[1]]
#     return np.array(strokes)

# def undo_size_normalization(strokes, minx, miny, maxx, maxy):
#     '''
#     Undo coordinate sacling.
#     '''
#     # Calculate width and height from the max and min values
#     # width = maxx - minx
#     height = maxy - miny

#     reversed_strokes = []
#     for stroke in strokes:
#         # Adjust coordinates by scaling based on height and shifting by minx and miny
#         x = (stroke[:, 0] * height) + minx
#         y = (stroke[:, 1] * height) + miny
#         reversed_strokes.append(np.stack([x, y], axis=1))
#     return np.array(reversed_strokes)

# # def undo_translation(strokes, raw_drawing):
# #     '''
# #     Undo translation to origin.
# #     '''
# #     start_positions = [(stroke[0][0], stroke[1][0]) for stroke in raw_drawing]

# #     untransl_drawing = []
# #     # Verify matching length of strokes and starting positions
# #     assert len(strokes) == len(start_positions), "Mismatch in number of strokes and starting positions."

# #     for stroke, (start_x, start_y) in zip(strokes, start_positions):
# #         untransl_stroke = [
# #             [x + start_x for x in stroke[:, 0]],  # Add back the original x-start to all x-coordinates
# #             [y + start_y for y in stroke[:, 1]]  # Add back the original y-start to all y-coordinates
# #         ]
# #         untransl_drawing.append(np.array(untransl_stroke).T)  # Transpose to match original shape [points, 2]
# #     return np.array(untransl_drawing)

# def get_bounding_box(drawing):
#     '''
#     Get bounding box of the drawing canvas (coordinate system).
#     '''
#     minx = 99999
#     miny = 99999
#     maxx = 0
#     maxy = 0

#     for s in drawing:
#       minx = min(minx, min(s[0]))
#       maxx = max(maxx, max(s[0]))
#       miny = min(miny, min(s[1]))
#       maxy = max(maxy, max(s[1]))
#     return (minx, miny, maxx, maxy)


