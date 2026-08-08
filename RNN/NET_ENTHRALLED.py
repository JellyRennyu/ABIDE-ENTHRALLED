# NET_ENTHRALLED - CENTRAL CONTROL & KINEMATIC MAPPING NETWORK
# Artificial Belief-Integrated Decision Engine: Enhanced Through Abstract Latent Long-term Reasoning (ABIDE-ENTHRALLED)
# Version: 2.0.0
# Tensorflow version: 2.15.0

# ----------------------------------------------------------
# INPUT ARCHITECTURE CONCATENATION (Section 2.4 & 2.5)
# ----------------------------------------------------------
# Consolidates belief states from GRU modules (Net A, Net B, Net C)
# concatenated with the inertial historical context pulse (Net T / h_tau).
# z_t = Concat(Flatten(B_t), h_tau)
# ----------------------------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# =========================
# CONFIG & SEEDING
# =========================
tf.random.set_seed(54)
np.random.seed(13)

INPUT_DIM = 32  # Net A (6) + Net B (8) + Net C (6) + Net T / h_tau (12)
OUTPUT_DIM = 8  # Action command vector u_t

SAMPLES = 80000
EPOCHS = 80
BATCH_SIZE = 256
LR = 1e-4

# =========================
# INTELLIGENT SAMPLE GENERATION
# =========================
fn_generate_sample = lambda: _generate_sample_internal()

def _generate_sample_internal():
    # Input embeddings representing belief spaces and temporal pulse
    net_a = np.clip(np.random.normal(0.5, 0.25, 6), 0, 1)
    net_b = np.clip(np.random.normal(0.5, 0.25, 8), 0, 1)
    net_c = np.clip(np.random.normal(0.5, 0.3, 6), 0, 1)
    net_t = np.clip(np.random.normal(0.5, 0.25, 12), 0, 1)

    # Context parameters
    ball_distance = np.random.beta(2, 5)
    alignment = np.random.uniform(0, 1)
    enemy_distance = np.random.beta(3, 2)
    blocking = np.random.uniform(0, 1)

    mobility = net_b[0]
    enemy_threat = net_c[0] * (1 - enemy_distance)
    interception = net_c[3]
    shoot_window = net_a[3] * alignment * (1 - blocking)

    pressure = net_t[0]
    defensive_overload = net_t[2]
    emergency = net_t[11]

    good_shot = (
        ball_distance < 0.2 and
        alignment > 0.8 and
        blocking < 0.3 and
        enemy_threat < 0.4
    )

    bad_shot = (
        ball_distance > 0.3 or
        alignment < 0.5 or
        blocking > 0.5
    )

    danger = (
        enemy_threat > 0.6 or
        interception > 0.5 or
        emergency > 0.7
    )

    # Kinematic mappings and action limits U_max (Section 2.4)
    if danger:
        v = -0.6
    else:
        v = 0.8 * (1 - ball_distance) * mobility

    w = np.clip((alignment - 0.5) * 2, -1, 1)

    if good_shot:
        kick = 1.0
    elif bad_shot:
        kick = 0.0
    else:
        kick = 0.2 * shoot_window

    urgency = np.clip(pressure + (1 - ball_distance), 0, 1)
    aggr = np.clip((1 - enemy_threat) * alignment, 0, 1)
    defense = np.clip(enemy_threat + defensive_overload, 0, 1)
    pass_pref = np.clip(blocking * (1 - alignment), 0, 1)
    emergency_flag = 1.0 if danger else 0.0

    x = np.concatenate([net_a, net_b, net_c, net_t])
    y = np.array([
        v, w, kick, urgency, aggr, defense, pass_pref, emergency_flag
    ], dtype=np.float32)

    return x, y


# =========================
# DATASET CONSTRUCTION
# =========================
def build_dataset(n):
    X, Y = [], []
    for _ in range(n):
        x, y = fn_generate_sample()
        X.append(x)
        Y.append(y)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# =========================
# CUSTOM LOSS FUNCTION (Concept-to-Action Optimization)
# =========================
def custom_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    wrong_kick = tf.maximum(0.0, y_pred[:, 2] - y_true[:, 2])
    miss_kick = tf.maximum(0.0, y_true[:, 2] - y_pred[:, 2])
    conflict = tf.maximum(0.0, y_pred[:, 4] - tf.abs(y_pred[:, 0]))

    return (
        mse
        + 0.8 * tf.reduce_mean(wrong_kick)
        + 0.6 * tf.reduce_mean(miss_kick)
        + 0.3 * tf.reduce_mean(conflict)
    )


# =========================
# ENTHRALLED CENTRAL CONTROL MODEL
# =========================
def build_model():
    model = models.Sequential([
        layers.Input(shape=(INPUT_DIM,), name="Consolidated_Belief_Inertial_Input"),

        layers.Dense(128, activation="relu", name="central_dense_1"),
        layers.BatchNormalization(name="bn_1"),

        layers.Dense(64, activation="relu", name="central_dense_2"),
        layers.BatchNormalization(name="bn_2"),

        layers.Dense(32, activation="relu", name="central_dense_3"),

        # Bounded activation mapping corresponding to U_max * tanh(W_c z_t + beta_c) (Section 2.4)
        layers.Dense(OUTPUT_DIM, activation="tanh", name="action_output_tanh")
    ], name="NET_ENTHRALLED_CENTRAL_CONTROLLER")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=custom_loss
    )

    return model


# =========================
# EXECUTION & TRAINING
# =========================
if __name__ == "__main__":
    print("[INFO] Generando dataset unificado para NET_ENTHRALLED...")
    X, Y = build_dataset(SAMPLES)

    split = int(0.85 * SAMPLES)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    print("[INFO] Construyendo arquitectura de control central...")
    model = build_model()
    model.summary()

    print("[INFO] Iniciando entrenamiento de política central...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # =========================
    # VISUALIZATION
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.title("NET_ENTHRALLED - Central Control Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Custom Loss")
    plt.tight_layout()
    plt.show()

    # =========================
    # MODEL PERSISTENCE
    # =========================
    model.save("NET_ENTHRALLED_CENTRAL_CONTROLLER.h5")
    print("\n[INFO] Modelo guardado exitosamente como NET_ENTHRALLED_CENTRAL_CONTROLLER.h5")