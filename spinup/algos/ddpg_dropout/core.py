import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

class BernoulliDropout(tf.keras.layers.Wrapper):
    """
    This wrapper is to implement regularization in BernoulliDropout given by Eq.(4) in https://arxiv.org/abs/1506.02142v6
    # Arguments
        layer: a layer instance
        dropout_rate:
            a positive number in (1, 0]
    """
    def __init__(self, layer, dropout_rate=5e-2, **kwargs):
        assert 'kernel_ragularizer' not in kwargs
        super(BernoulliDropout, self).__init__(layer, **kwargs)
        self.dropout_rate = dropout_rate
        self.retain_rate = 1 - self.dropout_rate
        self.bernoulli_dist = tfp.distributions.Bernoulli(probs=self.retain_rate)

    def build(self, input_shape=None):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        if hasattr(self.layer, 'built') and not self.layer.built:
            self.layer.build(input_shape)

    def conpute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def bernoulli_dropout(self, x):
        """
        Bernoulli dropout - used to mask input
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
    def __init__(self, layer_sizes=[32], dropout_rate=0.05,
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(BeroulliDropoutMLP, self).__init__()
        self.hidden_layers = []
        for h in layer_sizes[:-1]:
            self.hidden_layers.append(BernoulliDropout(tf.keras.layers.Dense(h, activation=hidden_activation),
                                                       dropout_rate=dropout_rate))
        self.out_layer = BernoulliDropout(tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation),
                                          dropout_rate=dropout_rate)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for h_layer in self.hidden_layers:
            x = h_layer(x, training)
        return self.out_layer(x, training)


class MLP(tf.keras.Model):
    def __init__(self, layer_sizes=[32], dropout_rate=0.05,
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(MLP, self).__init__()
        self.layers_list = []

        for h_size in layer_sizes[:-1]:
            self.layers_list.append(tf.keras.layers.Dropout(rate=dropout_rate))
            self.layers_list.append(tf.keras.layers.Dense(h_size, activation=hidden_activation))

        self.layers_list.append(tf.keras.layers.Dropout(rate=dropout_rate))
        self.layers_list.append(tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layers_list:
            if 'dropout' in layer.name:
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x