# NET_A - BALL STATE NETWORK (GRU/RNN Update)
# Artificial Belief-Integrated Decision Engine
# (ABIDE-ENTHRALLED)
# Version: 2.0.1
# Tensorflow: 2.15.0
# Activation functions used: relu, relu, sigmoid

# ===============================================
# BELIEF VECTOR DEFINITION (N_BELIEFS = 10)
# Index | Meaning | Range
# ---------------------------------------
# 0 | P_ball_possession_ego      | [0,1]
# 1 | P_ball_possession_ally     | [0,1]
# 2 | P_enemy1_ball_threat       | [0,1]
# 3 | P_enemy2_ball_threat       | [0,1]
# 4 | ball_distance_norm         | [0,1]
# 5 | ball_speed_norm            | [0,1]
# 6 | P_shot_opportunity_ego     | [0,1]
# 7 | P_pass_opportunity         | [0,1]
# 8 | ball_direction_alignment   | [0,1]
# 9 | P_ball_free                | [0,1]
# ===============================================

# ===============================================
# BALL SEMANTIC EMBEDDING (BALL_EMBED = 6)
# Index | Meaning | Type
# ------------------------------------------------------
# 0 | Offensive opportunity     | Continuous
# 1 | Enemy threat level        | Continuous
# 2 | Ball free likelihood      | Continuous
# 3 | Shoot window              | Binary-like
# 4 | Defensive urgency         | Binary-like
# 5 | Chase ball condition      | Binary-like
# ===============================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GRU, Activation, Input, Dropout
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# General values
np.random.seed(1585)
tf.random.set_seed(167)

N_BELIEFS = 10
BALL_EMBED = 6 
TIME_STEPS = 5  # Number of time steps for the RNN input

#================================
# Architecture Definition
#================================

model = Sequential(name="NET_A_RNN_GRU")

# Temporal input layer (timesteps, features)
model.add(Input(shape=(TIME_STEPS, N_BELIEFS), name="belief_sequence_input"))

# Layer 1: Temporal GRU Filter (d_h = 64 units)
model.add(GRU(units=64, return_sequences=False, name="gru_state_filter"))

# Layer 2: Dense Compression (16 units)
model.add(Dense(units=16, name="dense_compression"))
model.add(Activation("relu"))

# Layer 3: Ball Semantic Embedding (6 units)
model.add(Dense(units=BALL_EMBED, name="semantic_output"))
# FIX: Sigmoid activation guarantees outputs stay within [0, 1] bounds
model.add(Activation("sigmoid")) 

#================================
# LOSS FUNCTIONS
#================================

EMBED_LOSS_WEIGHTS = tf.constant([
    1.0, # Offensive opportunity
    1.0, # Enemy threat
    1.0, # Ball free likelihood
    2.0, # Shoot window
    2.0, # Defensive urgency
    2.0  # Chase condition
], dtype=tf.float32)

def weighted_mse(y_true, y_pred):
    error = tf.square(y_true - y_pred)
    weighted_error = error * EMBED_LOSS_WEIGHTS
    return tf.reduce_mean(weighted_error)

def semantic_penalty(y_pred):
    shoot = y_pred[:, 3]
    defend = y_pred[:, 4]
    chase = y_pred[:, 5]
    free = y_pred[:, 2]

    # Invalid semantic combinations
    p1 = tf.maximum(0.0, shoot + defend - 1.0)  # Shoot and defend cannot be high simultaneously
    
    # FIX: Chase condition must strictly align with ball free likelihood.
    p2_a = tf.maximum(0.0, chase - free) # Penalize chasing if it is NOT free
    p2_b = tf.maximum(0.0, free - chase) # Penalize ignoring the ball if it IS free
    p2 = p2_a + p2_b 
    
    return tf.reduce_mean(p1 + p2)

def total_loss(y_true, y_pred):
    return weighted_mse(y_true, y_pred) + 0.3 * semantic_penalty(y_pred)

model.compile(optimizer="adam", loss=total_loss)

#================================
# DATASET GENERATION (SEQUENTIAL)
#================================

def ball_semantic_embedding(b):
    shot_ego = b[6]
    enemy_threat = max(b[2], b[3])
    loose_ball = b[9]
    
    return np.array([
        shot_ego,
        enemy_threat,
        loose_ball,
        1.0 if shot_ego > 0.8 else 0.0, 
        1.0 if enemy_threat > 0.7 else 0.0,
        1.0 if loose_ball > 0.6 else 0.0
    ])
    
def generate_random_belief():
    b = np.random.rand(N_BELIEFS)
    if b[0] > 0.7: b[9] *= 0.2  # FIX: Corrected typo 'b[0] + 0.7' to 'b[0] > 0.7'
    if b[9] > 0.7:
        b[0] *= 0.2
        b[1] *= 0.2
    return b.astype(np.float32)

def build_sequential_dataset(samples=200000, time_steps=TIME_STEPS):
    X = np.zeros((samples, time_steps, N_BELIEFS), dtype=np.float32)
    Y = np.zeros((samples, BALL_EMBED), dtype=np.float32)
    
    for i in range(samples):
        # Generate a sequence of beliefs
        sequence = [generate_random_belief() for _ in range(time_steps)]
        X[i] = sequence
        
        # Assume the last belief in the sequence determines the target embedding
        Y[i] = ball_semantic_embedding(sequence[-1])
        
    print(f"Dataset Shape X: {X.shape}")
    print(f"Dataset Shape Y: {Y.shape}")
    
    return X, Y

# Training adjustment
# Reduce samples if the RAM is limited, or adjust batch size accordingly.
print("Building dataset...")
X, Y = build_sequential_dataset(100000)

print("Training the model...")
history = model.fit(
    X, Y,
    epochs=100,
    batch_size=128,
    validation_split=0.3,
    shuffle=True
)

#================================
# EVALUATION AND PLOTTING
#================================

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE + Penalty)')
plt.legend()
plt.grid(True)
plt.show()

#================================
# SEMANTIC VALIDATION
#================================

static_test_cases = np.array([
    [0.9, 0.1, 0.2, 0.1, 0.2, 0.3, 0.9, 0.4, 0.8, 0.1], # Case 1
    [0.1, 0.1, 0.8, 0.7, 0.4, 0.6, 0.1, 0.2, 0.3, 0.9], # Case 2
    [0.05, 0.05, 0.9, 0.85, 0.6, 0.4, 0.0, 0.1, 0.2, 0.05], # Case 3
    [0.6, 0.7, 0.2, 0.2, 0.3, 0.4, 0.3, 0.8, 0.7, 0.1], # Case 4
    [0.4, 0.4, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.5, 0.3]  # Case 5
], dtype=np.float32)

# Expansion into a 3D tensor simulating flat memory for RNN input
test_sequences = np.repeat(static_test_cases[:, np.newaxis, :], TIME_STEPS, axis=1)

semantic_outputs = model.predict(test_sequences)

print("\n===== SEMANTIC TEST RESULTS =====")
for i, out in enumerate(semantic_outputs):
    print(f"Test Case {i+1}")
    print(f"Offensive Opportunity: {out[0]:.2f}")
    print(f"Enemy threat level:    {out[1]:.2f}")
    print(f"Ball free likelihood:  {out[2]:.2f}")
    print(f"Shoot window:          {out[3]:.2f}")
    print(f"Defensive urgency:     {out[4]:.2f}")
    print(f"Chase condition:       {out[5]:.2f}")
    
model.save("NET_A_BALL_ENCODER_GRU.h5")
print("\nModel saved as NET_A_BALL_ENCODER_GRU.h5")