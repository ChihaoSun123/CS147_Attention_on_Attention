import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from AoA import Transformer_AoA



class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the caption of images given their encodings.

        :param vocab_size: The number of unique words in the caption dictionary.
        """

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = 1024
        self.batch_size = 64
        self.RNN_size = 1024
        self.learning_rate = 0.01
        self.hidden_layer_size = 1024

        self.embedding_matrix = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=0.1))
        self.lstm = tf.keras.layers.LSTM(self.RNN_size, return_sequences = True, return_state = True)
        self.AoA = Transformer_AoA(self.embedding_size, True)
        self.dense_layer_1 = tf.keras.layers.Dense(self.hidden_layer_size, activation='relu')
        self.dense_layer_2 = tf.keras.layers.Dense(self.vocab_size, activation="softmax")

    def call(self, caption_inputs, encoder_output, initial_state = None):
        """

        :param caption_inputs: [batch size x window size of captions (15)]
        :param encoder_output: [batch size x feature vectors x dimension of feature vectors] The encoder output
        is the layer normalized output of the AoA from the encoder
        :param initial_state: The initial state for the LSTM. Should be None at the start of training
        :return:
        """

        batchSize = len(caption_inputs)
        vectorized_inputs = tf.nn.embedding_lookup(self.embedding_matrix, caption_inputs)
        #vectorize_inputs should currently be shape [batch size x window size x embedding size]

        #TODO:contract dimensions of encoder_output by averaging along the columns
        # AoA_mean_pool = tf.nn.avg_pool(encoder_output, 2, 2, padding="VALID", data_format="NCW")
        AoA_reduced = tf.reduce_mean(encoder_output, axis=[1])

        #TODO: concatenate processed encoder_output with vectorized inputs
        vectorized_inputs = tf.math.add(vectorized_inputs, AoA_reduced)

        lstm_output, a, b = self.lstm(vectorized_inputs, initial_state=initial_state)
        attended_output = self.AoA.call(lstm_output, AoA_hat)
        dense_layer_1_output = self.dense_layer_1(attended_output)
        dense_layer_2_output = self.dense_layer_2(dense_layer_1_output)
        probs = tf.nn.softmax(dense_layer_2_output)

        return probs
    
    def loss(self, probs, labels):
        """
        The loss will be calculated through force teaching. We compare the softmax probabilities
        with the labels to obtain the loss.
        :param probs: Output from the decoder
        :param labels:
        :return:
        """
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        loss = tf.reduce_mean(losses)
        return loss

# dummy test of vocab size 50
model = Model(50)
AoA_hat = tf.zeros([1, 30, 1024])
caption_list = np.array([[1,2,3,4,5], [1, 3, 5, 7, 9]])

probs = model.call(caption_list, AoA_hat)
print(probs.shape)