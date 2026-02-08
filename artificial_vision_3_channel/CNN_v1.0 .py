# CNN - Vision to detect orange golf ball
# Artificial Belief-Integrated Decision Engine: Enhanced Through Abstract Latent Long-term Reasoning (ABIDE-ENTHRALLED)  - ball visual pose
# Version: 0.3.2
# Tensorflow version: 2.15.0
# Activation functions used: 
# First idea description: relu, relu, relu, relu, relu
# Use HSV to filter the orange color of the ball before any CNN, this allows to reduce the parameters and optimizes the program

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import os

def preprocess(frame, size=64):
    # Frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Values for the orange tone
    lower_orange = np.array([5, 120, 120])
    upper_orange = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.resize(mask, (size, size))
    mask = mask.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=-1)

    return mask

def build_model(input_shape=(64, 64, 1)):
    inputs = layers.Input(shape=input_shape)

    #Layer block 1
    x = layers.Conv2D(8, (3,3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Layer Block 2
    x = layers.DepthwiseConv2D((3,3), padding='same')(x)
    x = layers.Conv2D(16, (1,1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Layer Block 3
    x = layers.DepthwiseConv2D((3,3), padding='same')(x)
    x = layers.Conv2D(32, (1,1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(16, activation='relu')(x)

    # Output: x, y, confidence
    outputs = layers.Dense(3, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model

# This is a custom loss function for the compile
def custom_loss(y_true, y_pred):
    # y_true, y_pred: [batch, 3]
    pos_loss = tf.reduce_mean(tf.square(y_true[:, :2] - y_pred[:, :2]))
    conf_loss = tf.keras.losses.binary_crossentropy(
        y_true[:, 2], y_pred[:, 2]
    )

    return pos_loss + 0.5 * conf_loss


model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss= custom_loss
)

model.summary()

# Loading dataset function
def load_dataset(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    X, Y = [], []

    for _, row in df.iterrows():
     img_path = os.path.join(img_dir, row["filename"])
     frame = cv2.imread(img_path)
     if frame is None:
        continue

    x = preprocess(frame)
    y = np.array([row['x'], row['y'], row['conf']], dtype=np.float32)

    X.append(x)
    Y.append(y)



    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Loading dataset direction
X, Y = load_dataset(
    "dataset/labels.csv",
    "dataset/images"
)

model.fit(
    X, Y,
    batch_size = 32,
    epochs = 30,
    validation_split = 0.2,
    shuffle = True
)