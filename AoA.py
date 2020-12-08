import numpy as np
import tensorflow as tf
import AoA_internals as transformer

class Transformer_AoA(tf.keras.layers.Layer):
	def __init__(self, emb_sz, is_decoder=False):

		super(Transformer_AoA, self).__init__()

		# Define hyperparameters
		self.embedding_size = emb_sz
		self.is_decoder = is_decoder

		# Define encoder and decoder layers
		if self.is_decoder:
			# self.positional = transformer.Position_Encoding_Layer(window_sz=, emb_sz=)
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
		"""
		:param input:
		:return output [batch_size x window_size x vocab_size]
		"""

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
	# Shape test
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


if __name__ == '__main__':
	main()