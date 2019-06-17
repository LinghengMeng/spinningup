import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

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

# def count_vars(scope):
#     v = get_vars(scope)
#     return sum([np.prod(var.shape.as_list()) for var in v])

def count_vars(variables):
    return sum([np.prod(var.shape.as_list()) for var in variables])

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


class MLPAleatoricUnc(tf.keras.Model):
    """
    Multi-Layer Perceptron Network with Aleatoric Uncertainty:
        Model class used to create mlp.
    """
    def __init__(self, layer_sizes=[32],
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(MLPAleatoricUnc, self).__init__()
        self.hidden_layers = []
        for h in layer_sizes[:-1]:
            self.hidden_layers.append(tf.keras.layers.Dense(h, activation=hidden_activation,
                                                            kernel_initializer=kernel_initializer,
                                                            bias_initializer=bias_initializer))
        self.out_layer = tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation,
                                               kernel_initializer=kernel_initializer,
                                               bias_initializer=bias_initializer)
        # Log(sigma^2)
        self.out_layer_aleatoric_unc = tf.keras.layers.Dense(layer_sizes[-1], activation=tf.keras.activations.tanh,
                                                             kernel_initializer=kernel_initializer,
                                                             bias_initializer=bias_initializer)

    def call(self, inputs):
        x = inputs
        for h_layer in self.hidden_layers:
            x = h_layer(x)
        return self.out_layer(x), self.out_layer_aleatoric_unc(x)


class BernoulliDropout(tf.keras.layers.Wrapper):
    """
    This wrapper is to implement regularization in BernoulliDropout given by Eq.(4) in https://arxiv.org/abs/1506.02142v6
    # Arguments
        layer: a layer instance
        weight_regularizer:
            a positive number satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$
             (inverse observation noise),
            and N the number of instances in the dataset
            Note that kernel_regularizer is not needed, since we will implement it in this wrapper.
        dropout_rate:
            a positive number in (1, 0]
    """
    def __init__(self, layer, weight_regularizer=1e-6, dropout_rate=5e-2, **kwargs):
        assert 'kernel_ragularizer' not in kwargs
        super(BernoulliDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_rate = dropout_rate
        self.retain_rate = 1 - self.dropout_rate
        self.bernoulli_dist = tfp.distributions.Bernoulli(probs=self.retain_rate)

    def build(self, input_shape=None):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        if hasattr(self.layer, 'built') and not self.layer.built:
            self.layer.build(input_shape)

        # initialise regularizer
        # weight = self.layer.weights
        kernel = self.layer.kernel
        bias = self.layer.bias

        kernel_regularizer = self.weight_regularizer * (1-self.dropout_rate) * tf.reduce_sum(tf.square(kernel))
        bias_regualarizer = self.weight_regularizer * tf.reduce_sum(tf.square(bias))
        regularizer = tf.reduce_sum(kernel_regularizer + bias_regualarizer)
        # Add the regularization loss to collection.
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    def conpute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def bernoulli_dropout(self, x):
        """
        Bernoulli dropout - used to mask input
        :param x:
        :return:
        """
        mask = self.bernoulli_dist.sample(sample_shape=tf.shape(x))
        return x*tf.dtypes.cast(mask, tf.float32)

    def call(self, inputs, training=None):
        if training:
            return self.layer.call(self.bernoulli_dropout(inputs))
        else:
            return self.layer.call(inputs)

class BeroulliDropoutMLP(tf.keras.Model):
    """
    Bernoulli Dropout Multi-Layer Perceptron Network:
        Model class used to create mlp with bernoulli dropout.
    """
    def __init__(self, layer_sizes=[32], weight_regularizer=1e-6, dropout_rate=0.05,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(BeroulliDropoutMLP, self).__init__()
        self.hidden_layers = []
        for h in layer_sizes[:-1]:
            self.hidden_layers.append(BernoulliDropout(tf.keras.layers.Dense(h,
                                                                             activation=hidden_activation,
                                                                             kernel_initializer=kernel_initializer,
                                                                             bias_initializer=bias_initializer),
                                                       weight_regularizer=weight_regularizer,
                                                       dropout_rate=dropout_rate))
        self.out_layer = BernoulliDropout(tf.keras.layers.Dense(layer_sizes[-1],
                                                                activation=output_activation,
                                                                kernel_initializer=kernel_initializer,
                                                                bias_initializer=bias_initializer),
                                          weight_regularizer=weight_regularizer,
                                          dropout_rate=dropout_rate)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for h_layer in self.hidden_layers:
            x = h_layer(x, training)
        return self.out_layer(x, training)

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
    return pi, q1, q2, q1_pi