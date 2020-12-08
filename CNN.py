#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
from preprocess import get_data
from AoA import Transformer_AoA
from mrcnn import utils
from mrcnn.config import Config
from mrcnn.model import MaskRCNN


class InferenceConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    NAME = "coco"

    # Batch size
    IMAGES_PER_GPU = 10

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


class AoAEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(AoAEncoder, self).__init__()
        # Directory to save logs and trained model
        self.ROOT_DIR = os.path.abspath("../")
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")
        # Local path to trained weights file
        self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "mask_rcnn_coco.h5")
        self.config = InferenceConfig()
        # Create model object in inference mode.
        self.model = MaskRCNN(mode='inference', model_dir=self.MODEL_DIR, config=self.config)
        # Load weights trained on MS-COCO
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)
        self.embedding_sz = 6
        self.AoA = Transformer_AoA(self.embedding_sz)

    def call(self, inputs):
        """
        This function will use the pre-trained R-CNN to obtain feature vectors of size
        [batch_size x output channels x dimensions].
        It will feed these feature vectors into the AoA module to refine representation.

        A' = LayerNorm(A + AoA^E(fmh−att, WQeA, WKeA, WVeA))

        :param inputs: batch images of shape [batch_size x input_height x input_width]
        :return: a refined version of the feature vectors A' of shape [batch_size x output_channels x dimension]
        """
        detections, mrcnn = self.model.detect(inputs, verbose=0)
        print(tf.shape(detections))
        refined = self.AoA(detections)
        return refined

# inputs = get_data()
# images = inputs[0]
# encoder = AoAEncoder()
# encoder.call(images)