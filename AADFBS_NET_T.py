# NET_T - Long-term Strategic Transformer
# Artificial Belief-Integrated Decision Engine: Enhanced Through Abstract Latent Long-term Reasoning (ABIDE-ENTHRALLED) 
# Version: 0.1.0
# Tensorflow version: 2.15.0
# Activation functions used:

import tensorflow as tf
from tensorflow.keras.model import Model
from tensorflow.keras.layers import Dense, Input, LayerNormalization, MultiHeadAttention, Dropout, Add

# Parameters
SEQ_LEN = 20 # Max legnth plays considered
EMBED_DIM = 16 # Embedding per play
NUM_HEADS = 2 # Multiple attention
FF_DIM = 32 # Feed-forward dimension inside of transformer
DROPOUT = 0.1

# Input: Embebed history plays
inputs = input(shape=(SEQ_LEN, EMBED_DIM))

# Transformer architecture block
attn_output = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM)(inputs, inputs)
attn_output = Dropout(DROPOUT)(attn_output)
out1 = Add()([inputs, attn_output])
out1 = LayerNormalization()(out1)

# Feed-forward
ff_output = Dense(FF_DIM, activation="relu")(out1)
ff_output = Dense(EMBED_DIM)(ff_output)
ff_output = Dropout(DROPOUT)(ff_output)
out2 = Add()([out1, ff_output])
out2 = LayerNormalization()(out2)

# Pooling for vector resum
historical_embedding = tf.reduce_mean(out2, axis=1)

# Final model
transfromer_model = Model(inpus=inputs, outputs=historical_embedding)
transfromer_model.summary()