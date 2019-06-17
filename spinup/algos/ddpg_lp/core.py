import numpy as np
import tensorflow as tf
from collections import deque
from spinup.utils.logx import EpochLogger, Logger
import time

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

class MLP(tf.keras.Model):
    """
    Multi-Layer Perceptron Network:
        Model class used to create mlp.
    """
    def __init__(self, layer_sizes=[32],
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(MLP, self).__init__()
        self.hidden_layers = []
        for h in layer_sizes[:-1]:
            self.hidden_layers.append(tf.keras.layers.Dense(h, activation=hidden_activation,
                                                            kernel_initializer=kernel_initializer,
                                                            bias_initializer=bias_initializer))
        self.out_layer = tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation,
                                               kernel_initializer=kernel_initializer,
                                               bias_initializer=bias_initializer)

    def call(self, inputs):
        x = inputs
        for h_layer in self.hidden_layers:
            x = h_layer(x)
        return self.out_layer(x)

# """
# Actor-Critics
# """
# def mlp_actor_critic(x, a, hidden_sizes=(400,300), activation=tf.nn.relu,
#                      output_activation=tf.tanh, action_space=None):
#     act_dim = a.shape.as_list()[-1]
#     act_limit = action_space.high[0]
#     with tf.variable_scope('pi'):
#         pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
#     with tf.variable_scope('q'):
#         q = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
#     with tf.variable_scope('q', reuse=True):
#         q_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
#     return pi, q, q_pi
class LearningProgress(object):
    """
    Learning Progress encapsulates multiple versions of learned policy or value function in a time sequence.
    These different versions are used to make predictions and calculate learning progress.
    """
    def __init__(self, memory_length, input_dim=1, output_dim=1,
                 hidden_sizes=[32],
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear,
                 logger_kwargs=None, loger_file_name='learning_progress_log.txt'):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.memory_length = memory_length
        self.memory_track_models = deque(maxlen=self.memory_length)
        self.memory_track_outputs = deque(maxlen=self.memory_length)
        # Define model holders
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.input_dim))
        for m_i in range(self.memory_length):
            self.memory_track_models.append(MLP(hidden_sizes + [output_dim],
                                                hidden_activation=hidden_activation,
                                                output_activation=output_activation))
            self.memory_track_outputs.append(self.memory_track_models[m_i](self.input_ph))
        # Define logger
        self.lp_logger = Logger(output_fname=loger_file_name, **logger_kwargs)

    def compute_outputs(self, input, sess):
        outputs = np.zeros((self.memory_length, self.output_dim))
        for o_i in range(self.memory_length):
            outputs[o_i, :] = sess.run(self.memory_track_outputs[o_i], feed_dict={self.input_ph: input.reshape(1,-1)})
        return outputs

    def compute_learning_progress(self, input, sess, t=0, start_time=0):
        outputs = self.compute_outputs(input, sess)
        # First half part of memory track window
        first_half_outputs = outputs[0:int(np.floor(self.memory_length/2)),:]
        # Second half part of memory track window
        second_half_outputs = outputs[int(np.floor(self.memory_length/2)):,:]
        # L2 Norm i.e. Euclidean Distance
        lp_norm = np.linalg.norm(np.mean(second_half_outputs, axis=0) - np.mean(first_half_outputs, axis=0), ord=2)

        # Measure sum of variance
        var_outputs = np.sum(np.var(outputs, axis=0))
        var_first_half_outputs = np.sum(np.var(first_half_outputs, axis=0))
        var_second_half_outputs = np.sum(np.var(second_half_outputs, axis=0))
        self.lp_logger.log_tabular('VarAll', var_outputs)
        self.lp_logger.log_tabular('VarFirstHalf', var_first_half_outputs)
        self.lp_logger.log_tabular('VarSecondHalf', var_second_half_outputs)
        self.lp_logger.log_tabular('VarChange', var_second_half_outputs-var_first_half_outputs)
        # Log
        self.lp_logger.log_tabular('Step', t)
        self.lp_logger.log_tabular('L2LP', lp_norm)
        self.lp_logger.log_tabular('Time', time.time() - start_time)
        self.lp_logger.dump_tabular(print_data=False)
        return lp_norm, var_outputs, var_first_half_outputs, var_second_half_outputs

    def update_latest_memory(self, weights):
        """Update oldest model to latest weights, then append the latest model and output_placeholder
        to the top of queue."""
        # Set oldest model to latest weights
        oldest_model = self.memory_track_models.popleft()
        oldest_model.set_weights(weights)
        self.memory_track_models.append(oldest_model)
        # Move the corresponding output_placeholder to the top of queue
        self.memory_track_outputs.append(self.memory_track_outputs.popleft())

