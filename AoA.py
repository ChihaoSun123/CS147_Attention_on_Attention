import numpy as np
import tensorflow as tf
import AoA_internals as transformer

"""
Model uses two Transformer_AoA layers: one for the encoder and one for the decoder

The Transformer_AoA class and its helper layers are based on (extensively modified from!)
my implementation of the Transformer model for the Machine Translation assignment, and therefore
in turn is based on the stencil provided to us by Professor Ritchie and the CS 1470 staff.

(Specifically, the AoA development went like:
-->	modify Machine Translation transformer to use multiheaded attention (without breaking it)
--> modify that to use Attention on Attention design (without breaking it)
--> transplant code over to our project
--> further modify code to fit our tensor shape/dimensions,
	and have the correct number of transformer Blocks and recurrent connections / layer normalization
	depending on whether it's the encoder or decoder Transformer_AoA

(just stating here so there's full written credit! this was also discussed with our TA during development))
"""

class Transformer_AoA(tf.keras.layers.Layer):
	def __init__(self, emb_sz, is_decoder=False):

		super(Transformer_AoA, self).__init__()

		# Define hyperparameters
		self.embedding_size = emb_sz
		self.is_decoder = is_decoder

		# Define encoder and decoder layers
		if self.is_decoder:
			self.decoder = transformer.Transformer_Block(emb_sz=self.embedding_size, is_decoder=True, multi_headed=True)
		else:
			self.encoder_0 = transformer.Transformer_Block(emb_sz=self.embedding_size, is_decoder=False, multi_headed=True)
			self.encoder_1 = transformer.Transformer_Block(emb_sz=self.embedding_size, is_decoder=False, multi_headed=True)
			self.encoder_2 = transformer.Transformer_Block(emb_sz=self.embedding_size, is_decoder=False, multi_headed=True)
			self.encoder_3 = transformer.Transformer_Block(emb_sz=self.embedding_size, is_decoder=False, multi_headed=True)
			self.encoder_4 = transformer.Transformer_Block(emb_sz=self.embedding_size, is_decoder=False, multi_headed=True)
			self.encoder_5 = transformer.Transformer_Block(emb_sz=self.embedding_size, is_decoder=False, multi_headed=True)


	@tf.function
	def call(self, input, context=None):
		# Forward pass

		if self.is_decoder:
			output = self.decoder(input, context)
		else:	
			output = self.encoder_0(input)
			output = self.encoder_1(output)
			output = self.encoder_2(output)
			output = self.encoder_3(output)
			output = self.encoder_4(output)
			output = self.encoder_5(output)
		
		return output

def main():	
	# Shape tests
	inputs = tf.random.uniform(shape=(10,100,6))
	print("Input shape:")
	print(inputs.shape)
	test_refinement = Transformer_AoA(6)
	outputs = test_refinement(inputs)
	print("Output shape:")
	print(outputs.shape)

	inputs = tf.random.uniform(shape=(10,100,13))
	print("Input shape:")
	print(inputs.shape)
	test_refinement = Transformer_AoA(13)
	outputs = test_refinement(inputs)
	print("Output shape:")
	print(outputs.shape)

	inputs = tf.random.uniform(shape=(10,100,81))
	print("Input shape:")
	print(inputs.shape)
	test_refinement = Transformer_AoA(81)
	outputs = test_refinement(inputs)
	print("Output shape:")
	print(outputs.shape)


if __name__ == '__main__':
	main()