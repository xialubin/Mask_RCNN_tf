import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
import coco_train
from coco_data import CocoDataset
import tensorflow as tf

tf.keras.backend.set_learning_phase(True)

# Directory to save logs and trained model
MODEL_DIR = "./logs"

# Local path to trained weights file
COCO_MODEL_PATH = "./checkpoint/model-479"
# Download COCO trained weights from Releases if needed

# Directory of images to run detection on
IMAGE_DIR = "./images"

# dataset_train = CocoDataset()
# dataset_train.load_coco('./coco', "train", year="2014", auto_download=False)
# dataset_train.load_coco('./coco', "valminusminival", year="2014", auto_download=False)
# dataset_train.prepare()

dataset_val = CocoDataset()
dataset_val.load_coco('./coco', "minival", year="2014", auto_download=False)
dataset_val.prepare()


class InferenceConfig(coco_train.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.detect_load_weight(COCO_MODEL_PATH)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image])
# Visualize results
r = results[0]
# visualize.display_images([r['images']])
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])

# # detect1
# results = model.detect1(dataset_val)
# #
# # Visualize results
# r = results[0]
# visualize.display_images([r['images']])
# visualize.display_instances(r['images'], r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
