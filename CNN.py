import numpy as np
import tensorflow as tf
from preprocess import get_data

from mrcnn.config import Config
from mrcnn.model import MaskRCNN

class InferenceConfig(Config):
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 50

class AoAEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(AoAEncoder, self).__init__()
        # Root directory of the project
        self.ROOT_DIR = os.path.abspath("../")
        # Directory to save logs and trained model
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")
        # Local path to trained weights file
        self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "mask_rcnn_coco.h5")
        self.config = InferenceConfig()
        # Create model object in inference mode.
        self.MASK_RCNN = MaskRCNN(mode='inference', model_dir=self.MODEL_DIR, config=self.config)
        # Load weights trained on MS-COCO
        self.MASK_RCNN.load_weights(self.COCO_MODEL_PATH, by_name=True)
        self.embedding_sz = 75
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
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
        features = self.MASK_RCNN.conv_block(inputs, 3, [[1, 2, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1]], 1, 'a')
        print(tf.shape(features))
        refined = self.AoA(features)
        return refined