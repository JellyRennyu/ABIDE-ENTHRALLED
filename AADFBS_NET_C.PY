# NET_C -  ENEMY STATE NETWORK
# Artificial Belief-Integrated Decision Engine: Enhanced Through Abstract Latent Long-term Reasoning (ABIDE-ENTHRALLED)   - enemy state network
# Version: 1.1.2
# Tensorflow version: 2.15.0
# Activation functions used: relu, relu, linear

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

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import matplotlib.pyplot as plt

# General values
np.random.seed(445)
tf.random.set_seed(447)

N_BELIEFS = 12
ENEMY_EMBED = 6

# Type of model
model=Sequential()

# First layer
model.add(Dense(units=32, input_dim=12))
model.add(Activation("relu"))

# Layers
# Layer 2
model.add(Dense(units=16))
model.add(Activation("relu"))

# Layer 3 (Enemy semantic embedding)
model.add(Dense(units=6))
model.add(Activation("linear"))

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
    return tf.reduce_mean(err * EMBED_LOSS_WEIGHTS)

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
    optimizer="adam",
    loss=total_loss
)

# Target generation
def enemy_semantic_target(b):
    threat = max(1.0 - b[0], 1.0 - b[1])
    pressure = max(b[2], b[3])
    blocking = max(b[6], b[7])

    return np.array([
        threat,
        pressure,
        blocking,
        1.0 if blocking > 0.7 else 0.0, # Interception risk
        1.0 if pressure > 0.7 else 0.0, # Evasion needed
        1.0 if threat < 0.4 else 0.0 # Agressive viable
    ], dtype=np.float32)

# Dataset generator - This is a random generator for different states
def generate_random_belief():
    b = np.random.rand(N_BELIEFS)

    # Semantic coherence constraints
    if b[10]  > 0.7: # high pressure
        b[0] *= 0.5
    
    return b.astype(np.float32)

def build_dataset(samples=20000):
    X = []
    Y = []
    for _ in range(samples):
        belief = generate_random_belief()
        target = enemy_semantic_target(belief)

        X.append(belief)
        Y.append(target)
    
    X = np.array(X)
    Y = np.array(Y)

    print("Dataset Shape X:", X.shape)
    print("Dataset Shape Y:", Y.shape)

    return X, Y

# Training adjustment
X, Y = build_dataset(15000)

history = model.fit(
    X,
    Y,
    epochs=100,
    batch_size=128,
    validation_split=0.1,
    shuffle=True
)

# Plot code block
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("NET_C - ENEMY EMBEDDING LOSS")
plt.legend()
plt.grid(True)
plt.show()

# Semantic test beliefs (FOR ONLY MANUAL VALIDATION)
test_beliefs = np.array([
    [0.2,0.3,0.8,0.7,0.9,0.8,0.9,0.8,0.7,0.6,0.9,0.9],
    [0.7,0.6,0.2,0.2,0.3,0.4,0.2,0.2,0.3,0.4,0.2,0.8],
    [0.4,0.4,0.6,0.7,0.6,0.7,0.8,0.9,0.5,0.6,0.7,0.7]
], dtype=np.float32)

semantic_outputs = model.predict(test_beliefs)

print("\n===== ENEMY SEMANTIC TEST =====")
for i, o in enumerate(semantic_outputs):
    print(f"\nCase {i+1}")
    print(f"Threat:              {o[0]:.2f}")
    print(f"Pressure:            {o[1]:.2f}")
    print(f"Blocking:            {o[2]:.2f}")
    print(f"Interception risk:   {o[3]:.2f}")
    print(f"Evasion recommended: {o[4]:.2f}")
    print(f"Aggressive viable:   {o[5]:.2f}")

# Saving model
model.save("NET_C_ENEMY_ENCODER.h5")