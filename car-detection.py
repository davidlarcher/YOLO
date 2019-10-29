# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import iou, read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes, yolo_filter_boxes, yolo_eval
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

### session creation to start the graph
sess = K.get_session()

### get the names of the classes : person, bicycle, car, motorbike ...
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")

# for the scaling of the bounding boxes on the image
image_shape = (1024., 1224.)

### Load the pre-trained Model
yolo_model = load_model("model_data/yolo.h5")

### SHow the model structure
yolo_model.summary()

### calc yolo outputs
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))


### filter
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape, score_threshold=.5)

### predict


directory = os.fsencode('images')

for file in os.listdir(directory):
     image_file = os.fsdecode(file)
     if image_file.endswith(".jpeg"):
         # Preprocess your image
        image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

        ###Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
        out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

        # Print predictions info
        print('Found {} boxes for {}'.format(len(out_boxes), image_file))
        # Generate colors for drawing bounding boxes.
        colors = generate_colors(class_names)
        # Draw bounding boxes on the image file
        draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
        # Save the predicted bounding box on the image
        image.save(os.path.join("out", image_file), quality=90)
        # Display the results in the notebook
        #output_image = scipy.misc.imread(os.path.join("out", image_file))
        #imshow(output_image)

     else:
         continue


