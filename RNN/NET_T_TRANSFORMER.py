# NET_T - Long-term Strategic Transformer (Temporal Transformer Block)
# Artificial Belief-Integrated Decision Engine: Enhanced Through Abstract Latent Long-term Reasoning (ABIDE-ENTHRALLED) 
# Version: 2.0.0
# Tensorflow version: 2.15.0
# Activation functions used: relu, linear / sigmoid

# ----------------------------------------------------------
# INPUT SEQUENCE DEFINITION
# ----------------------------------------------------------
# Per timestep embedding:
# Index | Source | Meaning
# ----------------------------------------------------------
# 0-5   | NET_A  | Ball semantic embedding
# 6-11  | NET_B  | Self state semantic embedding
# 12-17 | NET_C  | Enemy state semantic embedding
# ----------------------------------------------------------
# EMBED_DIM = 18
# ----------------------------------------------------------

# ----------------------------------------------------------
# OUTPUT (HISTORICAL CONTEXT EMBEDDING / INERTIAL PULSE h_tau)
# Index | Meaning | Type
# ----------------------------------------------------------
# 0  | Strategic pressure trend      | Continuous
# 1  | Offensive momentum            | Continuous
# 2  | Defensive overload            | Continuous
# 3  | Counterattack readiness       | Continuous
# 4  | Risk accumulation             | Continuous
# 5  | Tempo acceleration            | Continuous
# 6  | Positional stability          | Continuous
# 7  | Tactical chaos level          | Continuous
# 8  | Aggression window             | Binary-like
# 9  | Regroup recommended           | Binary-like
# 10 | Long-play opportunity         | Binary-like
# 11 | Emergency defense             | Binary-like
# ----------------------------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, LayerNormalization,
    MultiHeadAttention, Dropout, Add
)
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Parameters
np.random.seed(1448)
tf.random.set_seed(145)

SEQ_LEN = 20      # History buffer length M (timesteps at slow loop frequency f_low)
EMBED_DIM = 18    # Combined semantic embedding dimension per timestep (Net A + Net B + Net C)
HIST_EMBED = 12   # Compact historical context vector h_tau dimension (d_v)

NUM_HEADS = 12    # Multi-head attention heads
FF_DIM = 48       # Feed-forward dimension inside transformer block
DROPOUT = 0.1

# ================================================
# ARCHITECTURE DEFINITION (TEMPORAL TRANSFORMER)
# ================================================

inputs = Input(shape=(SEQ_LEN, EMBED_DIM), name="Semantic_Sequence_Input")

# Multi-Head Self-Attention Block (Eq. from Section 2.3)
attn_output = MultiHeadAttention(
    num_heads=NUM_HEADS,
    key_dim=EMBED_DIM,
    name="temporal_mha"
)(inputs, inputs)

attn_output = Dropout(DROPOUT)(attn_output)
x = Add(name="residual_add_1")([inputs, attn_output])
x = LayerNormalization(name="layer_norm_1")(x)

# Feed-Forward Sublayer
ff = Dense(FF_DIM, activation="relu", name="ff_dense_1")(x)
ff = Dense(EMBED_DIM, name="ff_dense_2")(ff)
ff = Dropout(DROPOUT)(ff)

x = Add(name="residual_add_2")([x, ff])
x = LayerNormalization(name="layer_norm_2")(x)

# Temporal Aggregation via Average Pooling over sequence dimension M (h_tau vector)
# h_tau = (1 / M) * sum(A_tau^(j))
x = tf.reduce_mean(x, axis=1, name="temporal_average_pooling")

# Projections to the historical context embedding h_tau
outputs = Dense(HIST_EMBED, activation="sigmoid", name="historical_context_output")(x)

# Model instantiation
model = Model(inputs=inputs, outputs=outputs, name="NET_T_TEMPORAL_TRANSFORMER")
model.summary()

# ================================================
# SEQUENCE & TARGET GENERATOR
# ================================================

def generate_sequence():
    base = np.random.rand(EMBED_DIM)
    seq = []

    for t in range(SEQ_LEN):
        noise = np.random.normal(0, 0.05, EMBED_DIM)
        seq.append(base + noise + 0.01 * t)

    seq = np.array(seq, dtype=np.float32)

    # Future context target extraction based on semantic rules
    target = np.array([
        np.mean(seq[:, 0:6]),                 # Strategic pressure trend
        np.mean(seq[:, 6]),                   # Offensive momentum proxy
        np.mean(seq[:, 12]),                  # Defensive overload
        np.mean(seq[:, 7]),                   # Counterattack readiness
        np.std(seq[:, 12:18]),                # Risk accumulation
        np.mean(np.diff(seq[:, 0])),          # Tempo acceleration
        float(np.clip(1.0 - np.std(seq[:, 6:12]), 0.0, 1.0)), # Positional stability
        float(np.clip(np.std(seq), 0.0, 1.0)),                  # Tactical chaos level
        1.0 if np.mean(seq[:, 0]) > 0.7 else 0.0,  # Aggression window
        1.0 if np.mean(seq[:, 12]) > 0.7 else 0.0, # Regroup recommended
        1.0 if np.mean(seq[:, 6]) > 0.6 else 0.0,  # Long-play opportunity
        1.0 if np.mean(seq[:, 12]) > 0.8 else 0.0  # Emergency defense
    ], dtype=np.float32)

    return seq, target

def build_dataset(samples=5000):
    X, Y = [], []

    for _ in range(samples):
        seq, target = generate_sequence()
        X.append(seq)
        Y.append(target)

    X_arr = np.array(X, dtype=np.float32)
    Y_arr = np.array(Y, dtype=np.float32)
    
    print(f"Dataset Shape X: {X_arr.shape}")
    print(f"Dataset Shape Y: {Y_arr.shape}")
    
    return X_arr, Y_arr

print("Building transformer dataset...")
X, Y = build_dataset()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

print("Training NET_T Transformer model...")
history = model.fit(
    X, Y,
    epochs=75,
    batch_size=256,
    validation_split=0.2,
    shuffle=True
)

# ================================================
# MANUAL VERIFICATION
# ================================================

test_seq, expected = generate_sequence()
pred = model.predict(test_seq[np.newaxis])

print("\n===== NET_T MANUAL TEST RESULTS =====")
for i, v in enumerate(pred[0]):
    print(f"Context feature {i}: {v:.2f}")

model.save("NET_T_CONTEXT_TRANSFORMER.h5")
print("\nModel saved successfully as NET_T_CONTEXT_TRANSFORMER.h5")

# Training visual representation
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("NET_T - Historical Context Transformer Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()