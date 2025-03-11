"""
Undo data preprocessing by transforming the padded, resampled, normalized 
coordinates back to their original scale and position.

Note: While we can precisely undo normalization and padding, the resampling step 
cannot be perfectly reversed as it involves interpolation of points.
"""
import numpy as np
import json

def undo_preprocessing(reshaped_ink, stroke_length, raw_drawing, statistics_json_path):
    """
    Undo data preprocessing to transform back to original coordinates.
    
    Args:
        reshaped_ink: Transformed coordinates of drawings (num_strokes, max_len, channels)
        stroke_length: Length of each stroke
        raw_drawing: Original drawing data from ndjson file
        statistics_json_path: Path to json containing normalization statistics
    
    Returns:
        List of strokes in original coordinate space
    """
    # Convert tensor to numpy array (select only x,y coordinates)
    reshaped_ink = reshaped_ink[:, :, :2].numpy()

    # 1. Remove padding
    unpadded_strokes = remove_padding(reshaped_ink, stroke_length)

    # 2. Undo normalization (zero-mean, unit-variance)
    with open(statistics_json_path, "r") as json_file:
        stats = json.load(json_file)
        mean_x, stddev_x = stats['mean_x'], stats['stddev_x']
        mean_y, stddev_y = stats['mean_y'], stats['stddev_y']
    
    denormalized_strokes = undo_normalization(unpadded_strokes, 
                                            mean_x, stddev_x, 
                                            mean_y, stddev_y)

    # 3. Undo size normalization
    minx, miny, maxx, maxy = get_bounding_box(raw_drawing)
    final_strokes = undo_size_normalization(denormalized_strokes, 
                                          minx, miny, maxx, maxy)
    
    return final_strokes

def remove_padding(strokes, stroke_lengths):
    """Remove padding from each stroke based on its original length."""
    return [stroke[:length] for stroke, length in zip(strokes, stroke_lengths)]

def undo_normalization(strokes, mean_x, stddev_x, mean_y, stddev_y):
    """
    Undo zero-mean unit-variance normalization.
    
    Args:
        strokes: List of strokes with normalized coordinates
        mean_x, stddev_x: Statistics for x coordinates
        mean_y, stddev_y: Statistics for y coordinates
    """
    denormalized_strokes = []
    for stroke in strokes:
        # Denormalize x and y coordinates
        x = stroke[:, 0] * stddev_x + mean_x
        y = stroke[:, 1] * stddev_y + mean_y
        denormalized_strokes.append(np.stack([x, y], axis=1))
    return denormalized_strokes

def undo_size_normalization(strokes, minx, miny, maxx, maxy):
    """
    Undo size normalization by rescaling to original coordinate space.
    
    Args:
        strokes: List of strokes with normalized size
        minx, miny, maxx, maxy: Original bounding box coordinates
    """
    height = maxy - miny
    if height < 1e-6:
        height = 1.0

    restored_strokes = []
    for stroke in strokes:
        # Scale back to original size and shift to original position
        x = (stroke[:, 0] * height) + minx
        y = (stroke[:, 1] * height) + miny
        restored_strokes.append(np.stack([x, y], axis=1))
    return restored_strokes

def get_bounding_box(drawing):
    """Get the bounding box of the original drawing."""
    minx = min(min(stroke[0]) for stroke in drawing)
    miny = min(min(stroke[1]) for stroke in drawing)
    maxx = max(max(stroke[0]) for stroke in drawing)
    maxy = max(max(stroke[1]) for stroke in drawing)
    return minx, miny, maxx, maxy

def verify_reconstruction(original_drawing, reconstructed_strokes, tolerance=1e-3):
    """
    Verify the reconstruction quality by comparing with original drawing.
    
    Args:
        original_drawing: Original drawing from ndjson
        reconstructed_strokes: Reconstructed strokes after undoing preprocessing
        tolerance: Maximum allowed difference between coordinates
    
    Returns:
        bool: True if reconstruction is within tolerance
    """
    for orig_stroke, recon_stroke in zip(original_drawing, reconstructed_strokes):
        orig_points = np.array(list(zip(orig_stroke[0], orig_stroke[1])))
        diff = np.abs(orig_points - recon_stroke).max()
        if diff > tolerance:
            return False
    return True


