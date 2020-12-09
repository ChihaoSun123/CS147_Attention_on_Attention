import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from CNN import AoAEncoder
from Decoder import Decoder_Model
from preprocess import get_data

class Main_Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Main_Model, self).__init__()
        self.batch_size = 10
        self.learning_rate = 0.01

        self.encoder = AoAEncoder()
        self.decoder = Decoder_Model(vocab_size)

    def call(self, images, captions):
        encoder_output = self.encoder.call(images.numpy())
        probs = self.decoder(captions, encoder_output)
        return probs

    def loss(self, probs, labels, mask):
        """
        The loss will be calculated through force teaching. We compare the softmax probabilities
        with the labels to obtain the loss.
        :param probs: Output from the decoder
        :param labels:
        :return:
        """
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        losses = tf.boolean_mask(losses, mask)
        loss = tf.reduce_sum(losses)
        return loss

    def accuracy_function(self, prbs, labels, mask):
        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
        return accuracy

def train(model, train_image, train_caption, padding_index):
    #print(type(train_caption))
    decoder_input = train_caption[0:,0:14]
    labels = train_caption[0:,1:15]
    mask = tf.convert_to_tensor(np.where(np.asarray(labels)==padding_index, 0, 1))
    optimizer = tf.keras.optimizers.Adam(learning_rate = model.learning_rate)
    start_index = 0
    end_index = start_index + model.batch_size
    while end_index <= train_image.shape[0]:
        with tf.GradientTape() as tape:
            probs = model(train_image[start_index:end_index], decoder_input[start_index:end_index])
            print("shape of probs: ")
            print(probs.shape)   
            loss = model.loss(probs, labels[start_index:end_index], mask[start_index:end_index])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        start_index += model.batch_size
        end_index += model.batch_size

def test(model, test_image, test_caption, padding_index):
    decoder_input = test_caption[0:,0:14]
    labels = test_caption[0:,1:15]
    mask = tf.convert_to_tensor(np.where(np.asarray(labels)==padding_index, 0, 1))
    start_index = 0
    end_index = start_index + model.batch_size
    total_loss = 0
    accurate_count = 0

    while end_index <= test_image.shape[0]:
        probs = model(test_image[start_index:end_index], decoder_input[start_index:end_index])
        total_loss += model.loss(probs, labels[start_index:end_index], mask[start_index:end_index])
        temp1 = model.accuracy_function(probs, labels[start_index:end_index], mask[start_index:end_index])
        temp2 = tf.reduce_sum(mask[start_index:end_index])
        accurate_count += temp1 * tf.cast(temp2, dtype=tf.float32)
        start_index += model.batch_size
        end_index += model.batch_size
    
    my_perplexity = np.exp(tf.cast(total_loss, dtype=tf.float32)/tf.cast(tf.reduce_sum(mask[0:mask.shape[0]-mask.shape[0] % model.batch_size]), dtype=tf.float32))
    my_accuracy = accurate_count/tf.cast(tf.reduce_sum(mask[0:mask.shape[0]-mask.shape[0] % model.batch_size]), dtype=tf.float32)
    return my_perplexity, my_accuracy

def main():
    images, captions, dictionary = get_data(num_images = 10)
    train_image = images[0:10]
    test_image = images[0:10]
    train_caption = captions[0:10]
    test_caption = captions[0:10]

    print("dictionary size: ")
    print(len(dictionary))
    model = Main_Model(len(dictionary))

    # test whether model produces result of expected dimensions
    #decoder_input = train_caption[0:,0:14]
    #labels = train_caption[0:,1:15]
    #probs = model(images, train_caption[0:10])
    #print("probs shape: ")
    #print(probs.shape)

    train(model, train_image, train_caption, dictionary['<PAD>'])
    perplexity, accuracy = test(model, test_image, test_caption, dictionary['<PAD>'])
    print('perplexity:', perplexity)
    print('accuracy:', accuracy)


if __name__ == '__main__':
    main()

#inputs = get_data(num_images = 10)
#print("vocab_size:")
#print(len(inputs[2]))
#model = Main_Model(len(inputs[2]))
#res = model(inputs[0], inputs[1])
#print("res: ")
#print(res.shape)