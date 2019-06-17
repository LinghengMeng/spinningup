import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensorflow.python.layers.core import Dense
# from tensorflow.keras.layers import Dense
# import tensorflow.keras.layers.Dense as Dense

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

class ConcreteDropout(layers.Wrapper):
    # class ConcreteDropout(layers.Layer):
    """This wrapper allows to learn the dropout probability
        for any given input layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = Sequential()
        model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$
             (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and
             N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eucledian
            loss.

    # Warning
        You must import the actual layer class from tf layers,
         else this will not work.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = (np.log(init_min) - np.log(1. - init_min))
        self.init_max = (np.log(init_max) - np.log(1. - init_max))

    def build(self, input_shape=None):
        self.input_spec = layers.InputSpec(shape=input_shape)
        if hasattr(self.layer, 'built') and not self.layer.built:
            self.layer.build(input_shape)

        # initialise p
        self.p_logit = self.add_variable(name='p_logit',
                                         shape=(1,),
                                         initializer=tf.keras.initializers.RandomUniform((1,), self.init_min,
                                                                                         self.init_max),
                                         dtype=tf.float32,
                                         trainable=True)

        self.p = tf.nn.sigmoid(self.p_logit[0])
        tf.add_to_collection("LAYER_P", self.p)

        # initialise regulariser / prior KL term
        input_dim = int(np.prod(input_shape[-1]))

        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(
            weight)) / (1. - self.p)
        dropout_regularizer = self.p * tf.log(self.p)
        dropout_regularizer += (1. - self.p) * tf.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
        # Add the regularisation loss to collection.
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = 1e-7
        temp = 0.1

        unif_noise = tf.random_uniform(shape=tf.shape(x))
        drop_prob = (
                tf.log(self.p + eps)
                - tf.log(1. - self.p + eps)
                + tf.log(unif_noise + eps)
                - tf.log(1. - unif_noise + eps)
        )
        drop_prob = tf.nn.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if training:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            return inputs

# TODO: Only used in Critic
def mlp_concrete_dropout(x, hidden_sizes=(32,), weight_regularizer=1e-8, dropout_regularizer=1e-5, activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = ConcreteDropout(Dense(h, activation=tf.nn.relu), weight_regularizer=weight_regularizer,
                            dropout_regularizer=dropout_regularizer, trainable=True)(x, training=True)
    mean = ConcreteDropout(Dense(hidden_sizes[-1]), weight_regularizer=weight_regularizer,
                           dropout_regularizer=dropout_regularizer, trainable=True)(x, training=True)
    log_var = ConcreteDropout(Dense(hidden_sizes[-1]), weight_regularizer=weight_regularizer,
                              dropout_regularizer=dropout_regularizer, trainable=True)(x, training=True)
    return mean, log_var

class ConcreteDropoutMLP(tf.keras.Model):
    """Model class used to create mlp with concrete dropout."""
    def __init__(self, hidden_sizes=(32,), weight_regularizer=1e-8, dropout_regularizer=1e-5,
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(ConcreteDropoutMLP, self).__init__()
        self.hidden_layers = []
        for h in hidden_sizes[:-1]:
            self.hidden_layers.append(ConcreteDropout(tf.keras.layers.Dense(h, activation=hidden_activation),
                                                      weight_regularizer=weight_regularizer,
                                                      dropout_regularizer=dropout_regularizer, trainable=True))
        self.out_layer_mean = ConcreteDropout(Dense(hidden_sizes[-1], activation=output_activation),
                                              weight_regularizer=weight_regularizer,
                                              dropout_regularizer=dropout_regularizer, trainable=True)
        self.out_layer_logvar = ConcreteDropout(Dense(hidden_sizes[-1], activation=output_activation),
                                                weight_regularizer=weight_regularizer,
                                                dropout_regularizer=dropout_regularizer, trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for h_layer in self.hidden_layers:
            x = h_layer(x, training)
        mean = self.out_layer_mean(x, training)
        logvar = self.out_layer_logvar(x, training)
        return mean, logvar

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
                     output_activation=tf.tanh, action_space=None,
                     targetNN=False, target_noise=0.2, noise_clip=0.5):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)

    # If creating Target NNs, do Target policy smoothing, by adding clipped noise to target actions
    if targetNN:
        epsilon = tf.random_normal(tf.shape(pi), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a = pi + epsilon
        a = tf.clip_by_value(a, -act_limit, act_limit)

    with tf.variable_scope('q1'):
        # Share the same parameters by creating one critic model
        critic_1 = ConcreteDropoutMLP(list(hidden_sizes)+[1], weight_regularizer=1e-8, dropout_regularizer=1e-5)

        q1_mean, q1_logvar = critic_1(tf.concat([x,a], axis=-1), training=True)
        q1_mean, q1_logvar = tf.squeeze(q1_mean, axis=1), tf.squeeze(q1_logvar, axis=1)

        q1_pi_mean, q1_pi_logvar = critic_1(tf.concat([x,pi], axis=-1), training=True)
        q1_pi_mean, q1_pi_logvar = tf.squeeze(q1_pi_mean, axis=1), tf.squeeze(q1_pi_logvar, axis=1)

    with tf.variable_scope('q2'):
        critic_2 = ConcreteDropoutMLP(list(hidden_sizes) + [1], weight_regularizer=1e-8, dropout_regularizer=1e-5)
        q2_mean, q2_log_var = critic_2(tf.concat([x, a], axis=-1), training=True)
        q2_mean, q2_log_var = tf.squeeze(q2_mean, axis=1), tf.squeeze(q2_log_var, axis=1)

    return pi, q1_mean, q1_logvar, q2_mean, q2_log_var, q1_pi_mean, q1_pi_logvar