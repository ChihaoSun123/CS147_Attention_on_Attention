import numpy as np
import tensorflow as tf
from preprocess import get_data

class AoAEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(AoAEncoder, self).__init__()
        #
        self.RNN_layer = tf.keras.layers.LSTM()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        """
        This function will use the pre-trained R-CNN to obtain feature vectors of size
        [batch_size x output channels x dimensions].
        It will feed these feature vectors into the AoA module to refine representation.

        A' = LayerNorm(A + AoA^E(fmh−att, WQeA, WKeA, WVeA))

        :param inputs: batch images of shape [batch_size x input_height x input_width]
        :return: a refined version of the feature vectors A' of shape [batch_size x output_channels x dimension]
        """