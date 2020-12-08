import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from CNN import AoAEncoder
from Decoder import Decoder_Model
from preprocess import get_data

class Main_Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Main_Model, self).__init__()
        self.encoder = AoAEncoder()
        self.decoder = Decoder_Model(vocab_size)

    def call(self, images, captions):
        encoder_output = self.encoder.call(images.numpy())
        probs = self.decoder(captions, encoder_output)
        return probs

inputs = get_data(num_images = 10)
print("vocab_size:")
print(len(inputs[2]))
model = Main_Model(len(inputs[2]))
#print(type(inputs[0]))
res = model(inputs[0], inputs[1])
print("res: ")
print(res.shape)