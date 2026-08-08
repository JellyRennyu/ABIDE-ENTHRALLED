# NET_C - ENEMY STATE NETWORK (GRU/RNN Update)
# Artificial Belief-Integrated Decision Engine: Enhanced Through Abstract Latent Long-term Reasoning (ABIDE-ENTHRALLED) - enemy state network
# Version: 2.0.0
# Tensorflow version: 2.15.0
# Activation functions used: relu, relu, sigmoid

# ---------------------------------------
# Index | Meaning | Range
# ---------------------------------------
# 0 | enemy1_distance_norm        | [0,1]
# 1 | enemy2_distance_norm        | [0,1]
# 2 | enemy1_velocity_norm        | [0,1]
# 3 | enemy2_velocity_norm        | [0,1]
# 4 | enemy1_ball_alignment       | [0,1]
# 5 | enemy2_ball_alignment       | [0,1]
# 6 | enemy1_blocking_lane        | [0,1]
# 7 | enemy2_blocking_lane        | [0,1]
# 8 | enemy1_goal_alignment       | [0,1]
# 9 | enemy2_goal_alignment       | [0,1]
# 10 | enemy_pressure_level       | [0,1]
# 11 | enemy_observation_conf     | [0,1]
# ---------------------------------------

# ---------------------------------------
# Index | Meaning | Type
# ---------------------------------------
# 0 | Overall enemy threat        | Continuous
# 1 | Immediate pressure          | Continuous
# 2 | Defensive blocking          | Continuous
# 3 | Interception risk           | Binary-like
# 4 | Evasion recommended         | Binary-like
# 5 | Aggressive play viable      | Binary-like
# ---------------------------------------

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
np.random.seed(445)
tf.random.set_seed(447)

N_BELIEFS = 12
ENEMY_EMBED = 6
TIME_STEPS = 5  # Number of time steps for the RNN input

#================================
# Architecture Definition
#================================

model = Sequential(name="NET_C_RNN_GRU")

# Temporal input layer (timesteps, features)
model.add(Input(shape=(TIME_STEPS, N_BELIEFS), name="belief_sequence_input"))

# Layer 1: Temporal GRU Filter (d_h = 64 units)
model.add(GRU(units=64, return_sequences=False, name="gru_state_filter"))

# Layer 2: Dense Compression (16 units)
model.add(Dense(units=16, name="dense_compression"))
model.add(Activation("relu"))

# Layer 3 (Enemy semantic embedding)
model.add(Dense(units=ENEMY_EMBED, name="semantic_output"))
# Sigmoid activation guarantees outputs stay within [0, 1] bounds
model.add(Activation("sigmoid"))

# Loss weighting (Importance of each semantic output)
EMBED_LOSS_WEIGHTS = tf.constant([
    1.5,  # overall threat
    1.5,  # pressure
    1.0,  # blocking
    2.0,  # interception
    2.0,  # evasion
    1.5   # aggressive viable
], dtype=tf.float32)

def weighted_mse(y_true, y_pred):
    err = tf.square(y_true - y_pred)
    weighted_error = err * EMBED_LOSS_WEIGHTS
    return tf.reduce_mean(weighted_error)

# Semantic penalty
def semantic_penalty(y_pred):
    intercept = y_pred[:, 3]
    evade = y_pred[:, 4]
    agressive = y_pred[:, 5]
    threat = y_pred[:, 0]

    p1 = tf.maximum(0.0, agressive + intercept - 1.0)
    p2 = tf.maximum(0.0, agressive - (1.0 - threat))

    return tf.reduce_mean(p1 + p2)

# Total loss function
def total_loss(y_true, y_pred):
    return weighted_mse(y_true, y_pred) + 0.3 * semantic_penalty(y_pred)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=total_loss
)

# =========================
# INTELIGENT TARGET GENERATOR
# =========================
def enemy_semantic_target(b):

    d1, d2 = b[0], b[1]

    v1, v2 = b[2], b[3]

    align1, align2 = b[4], b[5]

    block1, block2 = b[6], b[7]

    g1, g2 = b[8], b[9]

    pressure_raw = b[10]

    # =========================
    # THREAT ASSESSMENT
    # =========================
    t1 = (1 - d1) * align1
    t2 = (1 - d2) * align2

    threat = max(t1, t2)

    # =========================
    # PRESSURE
    # =========================
    pressure = 0.5 * pressure_raw + 0.5 * max(v1, v2)

    # =========================
    # BLOCKING
    # =========================
    blocking = max(block1, block2) * threat

    # =========================
    # INTERCEPTION
    # =========================
    interception = 1.0 if (blocking > 0.6 and threat > 0.5) else 0.0

    # =========================
    # EVASION
    # =========================
    evasion = 1.0 if (threat > 0.6 and pressure > 0.5) else 0.0

    # =========================
    # AGGRESSIVE IS VIABLE
    # =========================
    aggressive = 1.0 if (threat < 0.4 and pressure < 0.5) else 0.0

    return np.array([
        threat,
        pressure,
        blocking,
        interception,
        evasion,
        aggressive
    ], dtype=np.float32)

# =========================
# DATASET GENERATION (SEQUENTIAL)
# =========================
def generate_realistic_belief():

    b = np.zeros(N_BELIEFS)

    b[0] = np.random.beta(2, 5)
    b[1] = np.random.beta(2, 5)

    b[2] = np.random.rand()
    b[3] = np.random.rand()

    b[4] = np.random.rand()
    b[5] = np.random.rand()

    b[6] = b[4] * (1 - b[0])
    b[7] = b[5] * (1 - b[1])

    b[8] = np.random.rand()
    b[9] = np.random.rand()

    b[10] = np.clip((1 - b[0]) + (1 - b[1]), 0, 1)

    b[11] = 1.0

    return b.astype(np.float32)

def build_sequential_dataset(samples=50000, time_steps=TIME_STEPS):
    X = np.zeros((samples, time_steps, N_BELIEFS), dtype=np.float32)
    Y = np.zeros((samples, ENEMY_EMBED), dtype=np.float32)

    for i in range(samples):
        sequence = [generate_realistic_belief() for _ in range(time_steps)]
        X[i] = sequence
        Y[i] = enemy_semantic_target(sequence[-1])

    print("Dataset Shape X:", X.shape)
    print("Dataset Shape Y:", Y.shape)

    return X, Y

print("Building sequential dataset...")
X, Y = build_sequential_dataset(30000)

print("Training the model...")
history = model.fit(
    X,
    Y,
    epochs=45,
    batch_size=128,
    validation_split=0.1,
    shuffle=True
)

model.save("NET_C_ENEMY_ENCODER_GRU.h5")
print("\nModel saved as NET_C_ENEMY_ENCODER_GRU.h5")

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE + Penalty)")
plt.title("NET_C - ENEMY EMBEDDING LOSS (GRU)")
plt.legend()
plt.grid(True)
plt.show()

# Semantic test beliefs (FOR ONLY MANUAL VALIDATION)
static_test_cases = np.array([
    [0.2, 0.3, 0.8, 0.7, 0.9, 0.8, 0.9, 0.8, 0.7, 0.6, 0.9, 0.9],
    [0.7, 0.6, 0.2, 0.2, 0.3, 0.4, 0.2, 0.2, 0.3, 0.4, 0.2, 0.8],
    [0.4, 0.4, 0.6, 0.7, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.7]
], dtype=np.float32)

# Expansion into a 3D tensor simulating flat memory for RNN input
test_sequences = np.repeat(static_test_cases[:, np.newaxis, :], TIME_STEPS, axis=1)

semantic_outputs = model.predict(test_sequences)

print("\n===== NET_C SEMANTIC TEST RESULTS =====")
for i, o in enumerate(semantic_outputs):
    print(f"\nCase {i+1}")
    print(f"Threat:              {o[0]:.2f}")
    print(f"Pressure:            {o[1]:.2f}")
    print(f"Blocking:            {o[2]:.2f}")
    print(f"Interception risk:   {o[3]:.2f}")
    print(f"Evasion recommended: {o[4]:.2f}")
    print(f"Aggressive viable:   {o[5]:.2f}")