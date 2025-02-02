import numpy as np
import tensorflow as tf
import numpy as np

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

def Attention_Matrix(K, Q, use_mask=False):
	"""

	Compute attention matrix for a single attention head.

	:param K: is [batch_size x size_keys x embedding_size]
	:param Q: is [batch_size x size_queries x embedding_size]
	:return: attention matrix
	"""
	
	window_size_queries = Q.get_shape()[1] # window size of queries
	window_size_keys = K.get_shape()[1] # window size of keys
	mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)
	atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1])

	# Compute attention weights using queries and key matrices
	presoftmax_score = tf.matmul(Q, tf.transpose(K, perm=[0,2,1]))/tf.sqrt(tf.cast(window_size_keys,dtype=tf.float32))

	# (If use_mask==True, add the attention mask before softmax)
	if use_mask:
		presoftmax_score = presoftmax_score + atten_mask

	attention_matrix = tf.nn.softmax(presoftmax_score)
	return attention_matrix


class Atten_Head(tf.keras.layers.Layer):
	def __init__(self, input_size, output_size, use_mask):		
		super(Atten_Head, self).__init__()

		self.use_mask = use_mask

		# Initialize the weight matrices for K, V, and Q
		# (i.e. W^K, W^V, W^Q)
		self.WK = self.add_weight(name="weightsK",shape=[input_size,output_size])
		self.WV = self.add_weight(name="weightsV",shape=[input_size,output_size])
		self.WQ = self.add_weight(name="weightsQ",shape=[input_size,output_size])

		
	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

		"""
		Call for a single attention head
		"""

		# Apply 3 matrices to turn inputs into keys, values, and queries
		K = tf.matmul(inputs_for_keys, self.WK)
		V = tf.matmul(inputs_for_values, self.WV)
		Q = tf.matmul(inputs_for_queries, self.WQ)

		# Call Attention_Matrix with the keys and queries, and with self.use_mask.
		attention_matrix = Attention_Matrix(K,Q, self.use_mask)

		# Apply the attention matrix to the values
		attended = tf.matmul(attention_matrix,V)

		return (Q,attended)



class Multi_Headed(tf.keras.layers.Layer):
	def __init__(self, emb_sz, use_mask):
		super(Multi_Headed, self).__init__()
		
		# Initialize heads, each with output attentions to be concatenated
		self.split_emb_sz = int(emb_sz/8)
		# (one head will have larger output size than others if embedding size
		# is not divisible by 8)
		self.remainder_sz = emb_sz - (7*self.split_emb_sz)

		self.head_0 = Atten_Head(emb_sz, self.split_emb_sz, use_mask=use_mask)
		self.head_1 = Atten_Head(emb_sz, self.split_emb_sz, use_mask=use_mask)
		self.head_2 = Atten_Head(emb_sz, self.split_emb_sz, use_mask=use_mask)
		self.head_3 = Atten_Head(emb_sz, self.split_emb_sz, use_mask=use_mask)
		self.head_4 = Atten_Head(emb_sz, self.split_emb_sz, use_mask=use_mask)
		self.head_5 = Atten_Head(emb_sz, self.split_emb_sz, use_mask=use_mask)
		self.head_6 = Atten_Head(emb_sz, self.split_emb_sz, use_mask=use_mask)
		self.head_7 = Atten_Head(emb_sz, self.remainder_sz, use_mask=use_mask)

		# Matrix for concatenated vectors, W^O
		self.WO = self.add_weight(name="weightsO",shape=[emb_sz,emb_sz])

		# AoA Information:
		self.information_v = tf.keras.layers.Dense(units=emb_sz, activation=None, use_bias=True)
		self.information_q = tf.keras.layers.Dense(units=emb_sz, activation=None, use_bias=False)

		# AoA Gate:
		self.gate_v = tf.keras.layers.Dense(units=emb_sz, activation=None, use_bias=True)
		self.gate_q = tf.keras.layers.Dense(units=emb_sz, activation=None, use_bias=False) 
		self.gate_sigmoid = tf.keras.layers.Activation('sigmoid')


	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
		"""
		Runs a multiheaded attention layer, with H=8 heads

		:param inputs_for_keys: tensor of [batch_size x num_vectors x input_size ]
		:param inputs_for_values: tensor of [batch_size x num_vectors x input_size ]
		:param inputs_for_queries: tensor of [batch_size x num_vectors x input_size ]
		:return: tensor of [batch_size x num_vectors x output_size ]
		"""

		Qs_and_attentions = [self.head_0(inputs_for_keys, inputs_for_values, inputs_for_queries),
							self.head_1(inputs_for_keys, inputs_for_values, inputs_for_queries),
							self.head_2(inputs_for_keys, inputs_for_values, inputs_for_queries),
							self.head_3(inputs_for_keys, inputs_for_values, inputs_for_queries),
							self.head_4(inputs_for_keys, inputs_for_values, inputs_for_queries),
							self.head_5(inputs_for_keys, inputs_for_values, inputs_for_queries),
							self.head_6(inputs_for_keys, inputs_for_values, inputs_for_queries),
							self.head_7(inputs_for_keys, inputs_for_values, inputs_for_queries)]

		# Unzip as lists of
		# each head's query matrices: [Q_0, Q_1, ..., Q_7]
		# each head's attention results: [Vhat_0, Vhat_1, ..., Vhat_7]
		Qs, attentions = [list(unzipped) for unzipped in zip(*Qs_and_attentions)]

		# Combine the attentions from each head
		concat_heads = tf.concat(values=attentions, axis=2)
		attended = tf.matmul(concat_heads, self.WO)

		# ATTENTION ON ATTENTION:

		# Concatenate together the Q's for each head
		# (each head has its own Q = (query input) times WQ which need to be combined
		# for calculating the information and gate vectors)
		Q = tf.concat(values=Qs, axis=2)

		# I  =  W^q_i Q  +  W^v_i Vhat  +  bias^i
		# (bias is included in the information_v keras layer)
		information = self.information_q(Q) + self.information_v(attended)

		# G  =  sigma( W^q_g Q  +  W^v_g Vhat  +  bias^g )
		# (bias is included in the gate_v keras layer)
		gate = self.gate_q(Q) + self.gate_v(attended)
		gate = self.gate_sigmoid(gate)

		# Ihat = I * G
		gated_information = information*gate
		return gated_information


class Transformer_Block(tf.keras.layers.Layer):
	def __init__(self, emb_sz, is_decoder, multi_headed=True):
		super(Transformer_Block, self).__init__()

		self.is_decoder = is_decoder
		if self.is_decoder:
			self.self_context_atten = Atten_Head(emb_sz,emb_sz,use_mask=False) if not multi_headed else Multi_Headed(emb_sz,use_mask=False)
		else:
			self.self_atten = Atten_Head(emb_sz,emb_sz,use_mask=is_decoder) if not multi_headed else Multi_Headed(emb_sz,use_mask=is_decoder)
			# Decoder AoA block does not use layer norm or recurrent connection!
			self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

	@tf.function
	def call(self, inputs, context=None):

		if self.is_decoder:
			assert context is not None,"Decoder blocks require context"
			# No layer norm or recurrent connection for Decoder AoA Module
			atten_out = self.self_context_atten(context,context,inputs)
		else:
			# Attention with a recurrent connection and layer normalization
			atten_out = self.self_atten(inputs,inputs,inputs)
			atten_out += inputs
			atten_out = self.layer_norm(atten_out)

		return atten_out
