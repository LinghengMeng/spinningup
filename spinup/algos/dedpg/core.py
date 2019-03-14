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
    with tf.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, q, q_pi

"""
Actor-Critics Ensemble
"""
def mlp_actor_critic_ensemble(x, a, hidden_sizes=(400,300), activation=tf.nn.relu,
                               output_activation=tf.tanh, action_space=None,
                               AC_names = ['ac_1', 'ac_2', 'ac_3', 'ac_4', 'ac_5']):
    # TODO: acotor-critic in an emsemble could have different structure.
    ac_ensemble = {}
    print('Creating actor-critic ensemble ...')
    for ac_i, ac_name in enumerate(AC_names):
        ac_ensemble[ac_name] = {}
        with tf.variable_scope(ac_name):
            pi, q, q_pi = mlp_actor_critic(x, a, hidden_sizes,
                                           activation, output_activation, action_space)
            ac_ensemble[ac_name]['pi'] = pi
            ac_ensemble[ac_name]['q'] = q
            ac_ensemble[ac_name]['q_pi'] = q_pi
    return ac_ensemble
