import numpy as np
import tensorflow as tf
import AoA_internals as transformer

class Transformer_AoA(tf.keras.layers.Layer):
	def __init__(self, batch_sz, k, D):

		super(Transformer_AoA, self).__init__()

		# Define batch size and optimizer/learning rate
		self.batch_size = batch_sz
		self.embedding_size = 150

		self.stddev = 0.01

		# Create positional encoder layers
		self.positional_encoder1 = transformer.Position_Encoding_Layer(window_sz=self.french_window_size, emb_sz=self.embedding_size)
		self.positional_encoder2 = transformer.Position_Encoding_Layer(window_sz=self.english_window_size, emb_sz=self.embedding_size)

		# Define encoder and decoder layers
		self.encoder = transformer.Transformer_Block(emb_sz=self.embedding_size, is_decoder=False, multi_headed=True)
		self.encoder_2 = transformer.Transformer_Block(emb_sz=self.embedding_size, is_decoder=False, multi_headed=True)
		self.decoder = transformer.Transformer_Block(emb_sz=self.embedding_size, is_decoder=True, multi_headed=True)
	
		# Define dense layer
		self.dense = tf.keras.layers.Dense(units=self.english_vocab_size, activation="softmax", use_bias=True, name="dense")

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: 
		:param decoder_input: 
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x vocab_size]
		"""
	
		#1) Add the positional embeddings to french sentence embeddings
		embedded_encoder_input = tf.nn.embedding_lookup(params=self.french_embedding_matrix, ids=encoder_input)
		embedded_encoder_input += self.positional_encoder1(embedded_encoder_input)

		#2) Pass the french sentence embeddings to the encoder
		encoder_output = self.encoder(embedded_encoder_input)
		#2) Pass the french sentence embeddings to the encoder
		encoder_output = self.encoder_2(encoder_output)

		#3) Add positional embeddings to the english sentence embeddings
		embedded_decoder_input = tf.nn.embedding_lookup(params=self.english_embedding_matrix, ids=decoder_input)
		embedded_decoder_input += self.positional_encoder2(embedded_decoder_input)

		#4) Pass the english embeddings and output of your encoder, to the decoder
		decoder_output = self.decoder(inputs=embedded_decoder_input, context=encoder_output)

		#5) Apply dense layer(s) to the decoder out to generate probabilities
		probs = self.dense(decoder_output)
		return probs

