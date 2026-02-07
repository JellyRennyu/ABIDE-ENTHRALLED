# NET_ENTHRALLED -  DETERMINISTIC DECISION NETWORK
# Artificial Belief-Integrated Decision Engine: Enhanced Through Abstract Latent Long-term Reasoning (ABIDE-ENTHRALLED)   - decision network
# Version: 1.0.1
# Tensorflow version: 2.15.0
# Activation functions used: relu, relu, tanh

# INPUTS FROM ALL SUB-NETWORKS
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# Index | Meaning | Type
# ----------------------------------------------------------
# 0 | Strategic pressure trend      | Continuous
# 1 | Offensive momentum            | Continuous
# 2 | Defensive overload             | Continuous
# 3 | Counterattack readiness       | Continuous
# 4 | Risk accumulation             | Continuous
# 5 | Tempo acceleration            | Continuous
# 6 | Positional stability           | Continuous
# 7 | Tactical chaos level           | Continuous
# 8 | Aggression window              | Binary-like
# 9 | Regroup recommended            | Binary-like
# 10| Long-play opportunity          | Binary-like
# 11| Emergency defense              | Binary-like
# ----------------------------------------------------------

# OUTPUT ACTION VECTOR
# ----------------------------------------------------------
# ACTION_EMBED (8)
# Index | Meaning | Type
# ----------------------------------------------------------
# 0 | Linear velocity command        | Continuous [-1,1]
# 1 | Angular velocity command       | Continuous [-1,1]
# 2 | Kick / Actuation intensity     | Continuous [0,1]
# 3 | Action urgency                 | Continuous [0,1]
# 4 | Aggressiveness level           | Continuous [0,1]
# 5 | Defensive bias                 | Continuous [0,1]
# 6 | Pass preference                | Continuous [0,1]
# 7 | Emergency override             | Binary-like
# ----------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Config values
tf.random.set_seed(546)
np.random.seed(123)

INPUT_DIM = 32 # From concatenated embeddings A(6) + B(8) + C(6) + temporal history consideration (12)
OUTPUT_DIM = 8 # Action embedding dimension
SAMPLES = 12000
EPOCHS = 60
BATCH_SIZE = 256
LR = 1e-3

# Dataset generation (random for testing purposes)
def generate_sample():
    # Random input embedding all normalized
    net_a = np.random.rand(6)
    net_b = np.random.rand(8)
    net_c = np.random.rand(6)
    net_t = np.random.rand(12)
    
    x = np.concatenate([net_a, net_b, net_c, net_t])

    # Logic of action generation (deterministic)
    offensive_momentum = net_a[1]
    emergency = net_t[11]
    enemy_threat = net_c[0]
    mobility = net_b[0]
    shooot_window = net_a[3]

    # For linear velocity
    v = np.clip(
        0.8 * mobility - 0.6 * enemy_threat,
        -1.0, 1.0
    )

    # For angular velocity
    w = np.clip(
        net_t[5] - net_c[1],
        -1.0, 1.0
    )

    # for kick intensity
    kick = shooot_window * offensive_momentum

    # For action urgency
    urgency = np.clip(net_t[4] + emergency, 0., 1)

    # For aggressiveness level
    aggr = np.clip(offensive_momentum - enemy_threat, 0, 1)

    # For defensive bias
    defense = np.clip(enemy_threat + net_t[2], 0, 1)

    # For pass preference
    pass_pref = np.clip(net_a[1] * net_b[4], 0, 1)

    # For emergency override (binary-like)
    emergency_flag = 1.0 if emergency > 0.6 else 0.0

    y = np.array([v,
                  w,
                  kick,
                  urgency,
                  aggr,
                  defense,
                  pass_pref,
                  emergency_flag], dtype=np.float32)

    return x, y

# Build dataset
def build_dataset(n_samples):
    X = []
    Y = []
    for _ in range(n_samples):
        x, y = generate_sample()
        X.append(x)
        Y.append(y)
    
    return np.array(X), np.array(Y)

# Build the model
def build_model():
    model = models.Sequential([
        layers.Input(shape=(INPUT_DIM,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        
        # True action embedding output
        layers.Dense(OUTPUT_DIM, activation="tanh") 
    ])

# Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="mse"
    )

    return model

# Train the model

print("[INFO] Generating dataset...")
X, Y = build_dataset(SAMPLES)

split = int(0.85 * SAMPLES)
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

print("[INFO] Building model...")
model = build_model()
model.summary()

print("[INFO] Training model...")
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Plot training graph

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("NET_ENTHRALLED Embedding Training")
plt.legend()
plt.grid()
plt.show()

# Verify with a sample input

print("\n[INFO] Testing model with a sample input...")
test_x, test_y = generate_sample()
pred = model.predict(test_x[np.newaxis])[0]

labels = [
    "linear velocity", "angular velocity", "kick", "Urgency", "Aggresiveness",
    "Defensive bias", "Pass pref", "Emergency"
]

print("\n[INFO] Testing model with a sample input...")
test_x, test_y = generate_sample()
pred = model.predict(test_x[np.newaxis])[0]

# save the model
model.save("AADFBS_ENTHRALLED.h5")