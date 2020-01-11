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
def mlp_actor_critic(x, a, hidden_sizes=(400,300), activation=tf.nn.relu, 
                     output_activation=tf.tanh, action_space=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    with tf.variable_scope('q1'):
        q1 = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q2'):
        q2 = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q1', reuse=True):
        q1_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q2', reuse=True):
        q2_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, q1, q2, q1_pi, q2_pi

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


class SkipConnectionMLP(tf.keras.Model):
    def __init__(self, layer_sizes=[32],
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(SkipConnectionMLP, self).__init__()
        self.layers_list = []

        for h_size in layer_sizes[:-1]:
            self.layers_list.append(tf.keras.layers.Dense(h_size, activation=hidden_activation))

        self.layers_list.append(tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for l_i, layer in enumerate(self.layers_list):
            if l_i == 1:
                x = tf.concat([x, inputs], axis=1)
            x = layer(x)
        return x