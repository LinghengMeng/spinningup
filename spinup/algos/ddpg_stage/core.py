import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli

class SimpleDense:
    """Dense Layer Class"""
    def __init__(self, n_in, n_out, activation=None, name="hidden"):
        if activation is None:
            self.activation = tf.identity
        else:
            self.activation = activation

        kernel_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=0.01)
        self.model_W = tf.get_variable("{}_W".format(name), initializer=kernel_initializer([n_in, n_out])) # variational parameters
        self.model_b = tf.get_variable("{}_b".format(name), initializer=tf.zeros([n_out]))

    def __call__(self, X):
        output = self.activation(tf.matmul(X, self.model_W) + self.model_b)
        return output

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp_simple(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    # Hidden layers
    for l, h in enumerate(hidden_sizes[:-1]):
        hidden_layer = SimpleDense(n_in=x.shape.as_list()[1],
                                        n_out=h,
                                        activation=activation,
                                        name="h{}".format(l+1))
        x = hidden_layer(x)
    # Output layer
    out_layer = SimpleDense(n_in=x.shape.as_list()[1],
                                 n_out=hidden_sizes[-1],
                                 activation=output_activation,
                                 name="Out")
    x = out_layer(x)
    return x

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation, kernel_initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.01))
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
                     output_activation=tf.tanh, action_space=None, dropout_rate=0, nn_type='simple_dense'):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    if nn_type == 'tf_dense':
        with tf.variable_scope('pi'):
            pi = act_limit * mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
        with tf.variable_scope('q'):
            q = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
        with tf.variable_scope('q', reuse=True):
            q_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)

    elif nn_type == 'simple_dense':
        with tf.variable_scope('pi'):
            pi = act_limit * mlp_simple(x, list(hidden_sizes) + [act_dim], activation, output_activation)
        with tf.variable_scope('q'):
            q = tf.squeeze(mlp_simple(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
        with tf.variable_scope('q', reuse=True):
            q_pi = tf.squeeze(mlp_simple(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)

    else:
        raise ValueError('Please choose a right nn_type!')

    return pi, q, q_pi
