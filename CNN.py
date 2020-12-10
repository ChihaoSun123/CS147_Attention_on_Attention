#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
from preprocess import get_data
from AoA import Transformer_AoA


class AoAEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(AoAEncoder, self).__init__()
        self.hidden_layer_size = 1024
        self.embedding_sz = 81
        self.AoA = Transformer_AoA(self.embedding_sz)
        self.expander = tf.keras.layers.Dense(self.hidden_layer_size)

    def call(self, inputs):
        """
        This function will use the pre-trained R-CNN to obtain feature vectors of size
        [batch_size x output channels x dimensions].
        It will feed these feature vectors into the AoA module to refine representation.

        A' = LayerNorm(A + AoA^E(fmh−att, WQeA, WKeA, WVeA))

        :param inputs: batch images of shape [batch_size x input_height x input_width]
        :return: a refined version of the feature vectors A' of shape [batch_size x output_channels x dimension]
        """
        refined = self.AoA(inputs)
        refined_expanded = self.expander(refined)
        return refined_expanded

# inputs = get_data(num_images=10)
# images = inputs
# encoder = AoAEncoder()
# res = encoder.call(images)
# print("res: ")
# print(res.shape)