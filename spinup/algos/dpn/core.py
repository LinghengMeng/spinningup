import tensorflow as tf
import numpy as np

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

# Atari environment outputs 210x160x3 RGBs
class DQN(tf.keras.Model):
    def __init__(self, act_dim):
        super(DQN, self).__init__()
        self.act_dim = act_dim
        self.cov1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4),
                                           padding="valid", activation="relu", data_format="channels_first")
        self.cov2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2),
                                           padding="valid", activation="relu", data_format="channels_first")
        self.cov3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1),
                                           padding="valid", activation="relu", data_format="channels_first")
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.out = tf.keras.layers.Dense(act_dim)
        self.hidden_layers = [self.cov1, self.cov2, self.cov3, self.flatten, self.fc1]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out(x)
