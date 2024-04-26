from PIL import Image, ImageDraw
from IPython.display import display

def visualize_stroke(unnormalized_stroke):
    '''
    Visualize strokes on a canvas.

    Input: 
        1) unnormalized_stroke: a list of numpy arrays (of coordinates)
    '''
    
    # Create a white canvas
    canvas_width = 240
    canvas_height = 270
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)

    # Draw each stroke on the canvas
    stroke_width = 2
    stroke_color = 'black'
    for stroke in unnormalized_stroke:
        if len(stroke) > 1:
            polyline = tuple(map(tuple, stroke))
            draw.line(polyline, fill=stroke_color, width=stroke_width)

    # Display the resulting image (succeed to recover the original drawing)
    display(image)