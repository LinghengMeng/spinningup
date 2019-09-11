import numpy as np
import tensorflow as tf


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


class MultiHeadMLP(tf.keras.Model):
    def __init__(self, shared_hidden_layer_sizes=[32], multi_head_layer_sizes=[[1]],
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(MultiHeadMLP, self).__init__()
        self.shared_hidden_layers_list = []
        self.separated_layers_list = []
        # shared hidden layers
        for shared_h_size in shared_hidden_layer_sizes:
            self.shared_hidden_layers_list.append(tf.keras.layers.Dense(shared_h_size, activation=hidden_activation))

        # separated head layers
        for head_index, head_layer_sizes in enumerate(multi_head_layer_sizes):
            self.separated_layers_list.append([])
            # hidden layer
            for h_size in head_layer_sizes[:-1]:
                self.separated_layers_list[head_index].append(tf.keras.layers.Dense(h_size,
                                                                                    activation=hidden_activation))
            # output layer
            self.separated_layers_list[head_index].append(tf.keras.layers.Dense(head_layer_sizes[-1],
                                                                                activation=output_activation))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        # propagate through shared hidden layers
        for h_layer in self.shared_hidden_layers_list:
            x = h_layer(x)

        # propagate through separated layers of each head
        multi_head_outputs = []
        for head_index in range(len(self.separated_layers_list)):
            head_x = x
            for separated_layer in self.separated_layers_list[head_index]:
                head_x = separated_layer(head_x)
            multi_head_outputs.append(head_x)

        return multi_head_outputs

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