import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from AoA import transformer_AoA



class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the caption of images given their encodings.

        :param vocab_size: The number of unique words in the caption dictionary.
        """

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = 64
        self.batch_size = 64
        self.RNN_size = 128
        self.learning_rate = 0.01

        self.embedding_matrix = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=0.1))
        self.lstm = tf.keras.layers.LSTM(self.RNN_size, return_sequences = True, return_state = False)
        self.AoA = transformer_AoA(self.batch_size)
        self.dense_layer_1 = tf.keras.layers.Dense(self.hidden_layer_size, activation='relu')
        self.dense_layer_2 = tf.keras.layers.Dense(self.vocab_size, activation="softmax")

    def call(self, caption_inputs, encoder_output, initial_state = None):
        #TODO: contract dim of encoder_output by averaging along the columns

        batchSize = len(caption_inputs)
        vectorized_inputs = tf.nn.embedding_lookup(self.embedding_matrix, caption_inputs)

        #TODO: concatenate processed encoder_output with vectorized inputs

        lstm_output = self.lstm(vectorized_inputs, initial_state=initial_state)
        attended_output = self.AoA(lstm_output)
        dense_layer_1_output = self.dense_layer_1(attended_output)
        dense_layer_2_output = self.dense_layer_2(dense_layer_1_output)

        return dense_layer_2_output
    
    def loss(self, probs, labels):
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        loss = tf.reduce_mean(losses)
        return loss