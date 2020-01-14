import numpy as np
import tensorflow as tf


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

"""
Actor-Critics
"""

class MLP(tf.keras.Model):
    def __init__(self, layer_sizes=[32],
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(MLP, self).__init__()
        self.layers_list = []
        for h_size in layer_sizes[:-1]:
            self.layers_list.append(tf.keras.layers.Dense(h_size, activation=hidden_activation))

        self.layers_list.append(tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x