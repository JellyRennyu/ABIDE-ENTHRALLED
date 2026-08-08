# NET_B - SELF STATE NETWORK (GRU/RNN Update)
# Artificial Belief-Integrated Decision Engine: Enhanced Through Abstract Latent Long-term Reasoning (ABIDE-ENTHRALLED)
# Version: 2.0.0
# Tensorflow: 2.15.0
# Activation functions used: relu, relu, sigmoid

# ==================================================
# SELF STATE VECTOR DEFINITION (N_SELF_STATES = 14)
# Index | Meaning | Range
# --------------------------------------------------
# 0  | ego_speed_norm            | [0,1]
# 1  | ego_accel_norm            | [0,1]
# 2  | ego_velocity_stability    | [0,1]
# 3  | ego_pose_confidence       | [0,1]
# 4  | yaw_rate_norm             | [0,1]
# 5  | angular_stability         | [0,1]
# 6  | slip_indicator            | [0,1]
# 7  | field_zone_confidence     | [0,1]
# 8  | near_boundary_risk        | [0,1]
# 9  | ally_distance_norm        | [0,1]
# 10 | ally_bearing_alignment    | [0,1]
# 11 | ally_pose_confidence      | [0,1]
# 12 | free_space_ahead          | [0,1]
# 13 | visual_occlusion_level    | [0,1]
# ==================================================

# ==================================================
# SELF SEMANTIC EMBEDDING (SELF_EMBED = 8)
# Index | Meaning | Type
# --------------------------------------------------
# 0 | Mobility readiness        | Continuous
# 1 | Localization confidence   | Continuous
# 2 | Dynamic stability         | Continuous
# 3 | Field safety              | Continuous
# 4 | Ally coordination         | Continuous
# 5 | Exploration capability    | Continuous
# 6 | Emergency state           | Binary-like
# 7 | Control reliability       | Continuous
# ==================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Activation, Input
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# General values
np.random.seed(429)
tf.random.set_seed(429)

N_SELF_STATES = 14
SELF_EMBED = 8
TIME_STEPS = 5  # Number of time steps for the RNN input

#================================
# Architecture Definition
#================================

model = Sequential(name="NET_B_RNN_GRU")

# Temporal input layer (timesteps, features)
model.add(Input(shape=(TIME_STEPS, N_SELF_STATES), name="self_sequence_input"))

# Layer 1: Temporal GRU Filter (d_h = 64 units)
model.add(GRU(units=64, return_sequences=False, name="gru_state_filter"))

# Layer 2: Dense Compression (16 units)
model.add(Dense(units=16, name="dense_compression"))
model.add(Activation("relu"))

# Layer 3: Self Semantic Embedding (8 units)
model.add(Dense(units=SELF_EMBED, name="semantic_output"))
# Sigmoid activation guarantees outputs stay within [0, 1] bounds
model.add(Activation("sigmoid")) 

#================================
# LOSS FUNCTIONS
#================================

EMBED_LOSS_WEIGHTS = tf.constant([
    1.0, # mobility readiness ("future develop")
    1.0, # Localization confidence
    1.0, # Dynamic stability
    1.2, # Field safety
    1.0, # Ally coordination
    1.0, # Capability for exploration
    2.0, # Emergency state
    1.5  # Control reliability
], dtype=tf.float32)

def weighted_mse(y_true, y_pred):
    error = tf.square(y_true - y_pred)
    weighted_error = error * EMBED_LOSS_WEIGHTS
    return tf.reduce_mean(weighted_error)

def semantic_penalty(y_pred):
    emergency = y_pred[:, 6]
    control = y_pred[:, 7]
    stability = y_pred[:, 2]

    # Invalid semantic combinations
    p1 = tf.maximum(0.0, emergency + stability - 1.0)
    p2 = tf.maximum(0.0, emergency + control - 1.0)

    return tf.reduce_mean(p1 + p2)

def total_loss(y_true, y_pred):
    return weighted_mse(y_true, y_pred) + 0.3 * semantic_penalty(y_pred)

model.compile(optimizer="adam", loss=total_loss)

#================================
# DATASET GENERATION (SEQUENTIAL)
#================================

def self_semantic_target(s):
    mobility = 0.5 * (s[0] + s[1])
    localization_quality = s[3]
    stability = np.clip(1.0 - max(s[4], s[6]), 0.0, 1.0)
    field_safety = 1.0 - s[8]
    coordination = 0.5 * (1.0 - s[9] + s[10])
    exploration = s[12] * (1.0 - s[13])
    emergency = float((s[6] > 0.7) or (s[8] > 0.7))
    control = 0.5 * (stability + localization_quality)

    return np.array([
        mobility,
        localization_quality,
        stability,
        field_safety,
        coordination,
        exploration,
        emergency,
        control
    ], dtype=np.float32)
    
def generate_random_self_state():
    s = np.random.rand(N_SELF_STATES)

    # Coherence Adjustments
    if s[6] > 0.7:  # slip
        s[0] *= 0.3
        s[1] *= 0.3
    if s[8] > 0.7:  # boundary
        s[12] *= 0.2
        
    return s.astype(np.float32)

def build_sequential_dataset(samples=50000, time_steps=TIME_STEPS):
    X = np.zeros((samples, time_steps, N_SELF_STATES), dtype=np.float32)
    Y = np.zeros((samples, SELF_EMBED), dtype=np.float32)
    
    for i in range(samples):
        # Generate a sequence of self-states
        sequence = [generate_random_self_state() for _ in range(time_steps)]
        X[i] = sequence
        
        # Assume the last self-state in the sequence determines the target embedding
        Y[i] = self_semantic_target(sequence[-1])
        
    print(f"Dataset Shape X: {X.shape}")
    print(f"Dataset Shape Y: {Y.shape}")
    
    return X, Y

# Training adjustment
print("Building dataset...")
X, Y = build_sequential_dataset(30000)

print("Training the model...")
history = model.fit(
    X, Y,
    epochs=50,
    batch_size=64,
    validation_split=0.3,
    shuffle=True
)

#================================
# EVALUATION AND PLOTTING
#================================

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("NET_B - SELF STATE EMBEDDING LOSS (GRU)")
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE + Penalty)')
plt.legend()
plt.grid(True)
plt.show()

#================================
# SEMANTIC VALIDATION
#================================

static_test_cases = np.array([
    [0.8, 0.7, 0.9, 0.95, 0.1, 0.9, 0.05, 0.9, 0.1, 0.4, 0.8, 0.9, 0.8, 0.1], # Case 1: Normal healthy state
    [0.2, 0.1, 0.4, 0.5, 0.8, 0.3, 0.85, 0.6, 0.2, 0.6, 0.5, 0.4, 0.3, 0.4], # Case 2: High slip
    [0.5, 0.5, 0.8, 0.8, 0.2, 0.8, 0.1, 0.4, 0.85, 0.5, 0.5, 0.5, 0.1, 0.2], # Case 3: Boundary risk
    [0.9, 0.8, 0.9, 0.9, 0.1, 0.9, 0.0, 0.9, 0.05, 0.3, 0.9, 0.9, 0.9, 0.0], # Case 4: Optimal high mobility
    [0.1, 0.1, 0.2, 0.3, 0.9, 0.1, 0.9, 0.2, 0.9, 0.8, 0.2, 0.3, 0.05, 0.8]  # Case 5: Emergency/Critical
], dtype=np.float32)

# Expansion into a 3D tensor simulating flat memory for RNN input
test_sequences = np.repeat(static_test_cases[:, np.newaxis, :], TIME_STEPS, axis=1)

semantic_outputs = model.predict(test_sequences)

print("\n===== NET_B SEMANTIC TEST RESULTS =====")
for i, out in enumerate(semantic_outputs):
    print(f"\nTest case {i+1}")
    print(f"Mobility readiness:      {out[0]:.2f}")
    print(f"Localization confidence: {out[1]:.2f}")
    print(f"Dynamic stability:       {out[2]:.2f}")
    print(f"Field safety:            {out[3]:.2f}")
    print(f"Ally coordination:       {out[4]:.2f}")
    print(f"Exploration capability:  {out[5]:.2f}")
    print(f"Emergency state:         {out[6]:.2f}")
    print(f"Control reliability:     {out[7]:.2f}")
    
model.save("NET_B_SELF_ENCODER_GRU.h5")
print("\nModel saved as NET_B_SELF_ENCODER_GRU.h5")