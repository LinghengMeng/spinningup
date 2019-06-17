import numpy as np
import tensorflow as tf

from tensorflow.contrib.keras import layers
from tensorflow.python.layers.core import Dense

class VariationalDense:
    """Variational Dense Layer Class"""
    def __init__(self, n_in, n_out, dropout_mask_ph,
                 model_prob=0.9, model_lam=3e-4, activation=None, name="hidden"):
        self.model_prob = model_prob    # probability to keep units
        self.model_lam = model_lam      # l^2 / 2*tau: l=1e-2, tau=[0.1, 0.15, 0.2]

        self.dropout_mask_ph = dropout_mask_ph # placeholder: p_s * i_s
        self.p_s = tf.shape(self.dropout_mask_ph)[0] # post sample size
        self.DM = tf.zeros(shape=[self.p_s, n_in, n_in]) # Dropout masks: p_s * i_s * i_s
        self.DM = tf.linalg.set_diag(self.DM, self.dropout_mask_ph)

        kernel_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=0.01)
        self.model_W = tf.get_variable("{}_W".format(name), initializer=kernel_initializer([n_in, n_out])) # variational parameters
        self.model_b = tf.get_variable("{}_b".format(name), initializer=tf.zeros([n_out]))

        self.model_DMW = tf.einsum('pij,jk->pik', self.DM, self.model_W) # Masked weight: p_s * i_s * o_s
        self.model_tiled_b = tf.tile(tf.reshape(self.model_b, [1, n_out]), [self.p_s, 1])

        if activation is None:
            self.activation = tf.identity
        else:
            self.activation = activation

    def __call__(self, X):
        # X shape (p_s * b_s * i_s)
        net_input = tf.einsum('pbi,pio->pbo', X, self.model_DMW) + self.model_tiled_b
        output = self.activation(net_input) # output: p_s * b_s * o_s
        return output

    @property
    def regularization(self):
        return self.model_lam * (
            self.model_prob * tf.reduce_sum(tf.square(self.model_W)) +
            tf.reduce_sum(tf.square(self.model_b))
        )

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
            self.layer.call(inputs)

# TODO: Only used in Critic
def mlp_concrete_dropout(x, hidden_sizes=(32,), weight_regularizer=1e-8, dropout_regularizer=1e-5, activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = ConcreteDropout(Dense(h, activation=tf.nn.relu), weight_regularizer=weight_regularizer,
                            dropout_regularizer=dropout_regularizer, trainable=True)(x, training=True)
    mean = ConcreteDropout(Dense(hidden_sizes[-1]), weight_regularizer=weight_regularizer,
                           dropout_regularizer=dropout_regularizer, trainable=True)(x, training=True)
    log_var = ConcreteDropout(Dense(hidden_sizes[-1]), weight_regularizer=weight_regularizer,
                              dropout_regularizer=dropout_regularizer, trainable=True)(x, training=True)
    return tf.concat([mean, log_var], -1, name='output')

class DropoutMaskGenerator:
    """Class used to generate dropout mask."""
    def __init__(self,x_dim, hidden_sizes=(32,), model_prob=0.9):
        self.x_dim = x_dim
        self.hidden_sizes = hidden_sizes
        self.model_prob = model_prob

    def generate_dropout_mask_placeholders(self):
        dropout_mask_placeholders = []
        for l, size in enumerate((self.x_dim, *self.hidden_sizes)):
            dropout_mask_placeholders.append(tf.placeholder(dtype=tf.float32, shape=(None, size),
                                                            name='dropout_mask_{}'.format(l)))
        return dropout_mask_placeholders

    def generate_dropout_mask(self, post_size):
        # TODO: generate masks accroding to post_size
        new_dropout_masks = []
        for l, size in enumerate((self.x_dim, *self.hidden_sizes)):
            new_dropout_masks.append(np.random.binomial(1, self.model_prob, (post_size, size)))
        return new_dropout_masks

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

# TODO: add batch normalization
def mlp_variational(x, dropout_mask_phs, hidden_sizes=(32,),
                    activation=tf.tanh, output_activation=None, dropout_rate=0.1):

    # layer_sizes = (input_size, h1, h2, ..., output_size)
    layer_sizes = hidden_sizes.copy()
    layer_sizes.insert(0, x.shape.as_list()[1])

    # tile x from shape (b_s * i_s) to (p_s * b_s * i_s)
    post_size = tf.shape(dropout_mask_phs[0])[0]
    x = tf.tile(tf.reshape(x, [1, tf.shape(x)[0], tf.shape(x)[1]]), [post_size, 1, 1])
    # TODO: no dropout on input
    regularization = 0
    # Create hidden layers
    for layer_i in range(1,len(layer_sizes)-1):
        hidden_layer = VariationalDense(n_in=layer_sizes[layer_i-1],
                                        n_out=layer_sizes[layer_i],
                                        dropout_mask_ph=dropout_mask_phs[layer_i-1],
                                        model_prob=1.0 - dropout_rate,
                                        model_lam=3e-4,
                                        activation=activation,
                                        name="h{}".format(layer_i + 1))
        x = hidden_layer(x)
        regularization += hidden_layer.regularization

    # Output layer
    out_layer = VariationalDense(n_in=layer_sizes[-2],
                                 n_out=layer_sizes[-1],
                                 dropout_mask_ph=dropout_mask_phs[-1],
                                 model_prob=1.0-dropout_rate,
                                 model_lam=3e-4,
                                 activation=output_activation,
                                 name="Out")
    x = out_layer(x)
    regularization += out_layer.regularization
    return x, regularization

def mlp_dropout(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, dropout_rate=0):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
        x = tf.layers.dropout(x, rate=dropout_rate, training=True)
    x = tf.layers.dropout(x, rate=dropout_rate, training=True)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

# """
# Random Network Distillation
# """
# def random_net_distill(x_ph, a_ph,  hidden_sizes=(400,300), activation=tf.nn.relu,
#                        output_activation=tf.tanh, action_space=None):
#     act_dim = a_ph.shape.as_list()[-1]
#     act_limit = action_space.high[0]
#     with tf.variable_scope('rnd_targ_act'):
#         rnd_targ_act = act_limit * mlp(x_ph, list(hidden_sizes) + [act_dim], activation, output_activation)
#     with tf.variable_scope('rnd_pred_act'):
#         rnd_pred_act = act_limit * mlp(x_ph, list(hidden_sizes) + [act_dim], activation, output_activation)
#     with tf.variable_scope('rnd_targ_cri'):
#         rnd_targ_cri = tf.squeeze(mlp(tf.concat([x_ph, a_ph], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
#     with tf.variable_scope('rnd_pred_cri'):
#         rnd_pred_cri = tf.squeeze(mlp(tf.concat([x_ph, a_ph], axis=-1), list(hidden_sizes) + [1], activation, None),
#                                   axis=1)
#     return rnd_targ_act, rnd_pred_act, rnd_targ_cri, rnd_pred_cri

"""
Random Network Distillation
"""
def random_net_distill(x_ph, a_ph,  hidden_sizes=(400,300), activation=tf.nn.relu,
                       output_activation=tf.tanh, action_space=None, dropout_rate=0.1):
    act_dim = a_ph.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('rnd_targ_act'):
        rnd_targ_act = act_limit * mlp(x_ph, list(hidden_sizes) + [act_dim], activation, output_activation)
    with tf.variable_scope('rnd_pred_act'):
        # rnd_pred_act = act_limit * mlp(x_ph, list(hidden_sizes) + [act_dim], activation, output_activation)
        rnd_pred_act_in_dim = x_ph.shape.as_list()[1]
        rnd_pred_act_dropout_mask_generator = DropoutMaskGenerator(rnd_pred_act_in_dim,
                                                                   hidden_sizes, model_prob=1.0 - dropout_rate)
        rnd_pred_act_dropout_mask_phs = rnd_pred_act_dropout_mask_generator.generate_dropout_mask_placeholders()
        rnd_pred_act, rnd_pred_act_reg = mlp_variational(x_ph, rnd_pred_act_dropout_mask_phs,
                                                         list(hidden_sizes) + [act_dim], activation,
                                                         output_activation, dropout_rate)
        rnd_pred_act = act_limit * rnd_pred_act

    with tf.variable_scope('rnd_targ_cri'):
        rnd_targ_cri = tf.squeeze(mlp(tf.concat([x_ph, a_ph], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
    with tf.variable_scope('rnd_pred_cri'):
        rnd_pred_cri = tf.squeeze(mlp(tf.concat([x_ph, a_ph], axis=-1), list(hidden_sizes) + [1], activation, None),
                                  axis=1)
        rnd_pred_cri_in_ph = tf.concat([x_ph, a_ph], axis=-1)
        rnd_pred_cri_in_dim = rnd_pred_cri_in_ph.shape.as_list()[1]
        rnd_pred_cri_dropout_mask_generator = DropoutMaskGenerator(rnd_pred_cri_in_dim,
                                                                   hidden_sizes, model_prob=1.0 - dropout_rate)
        rnd_pred_cri_dropout_mask_phs = rnd_pred_cri_dropout_mask_generator.generate_dropout_mask_placeholders()

        rnd_pred_cri, rnd_pred_cri_reg = mlp_variational(rnd_pred_cri_in_ph, rnd_pred_cri_dropout_mask_phs,
                                                         list(hidden_sizes) + [1],
                                                         activation, None, dropout_rate)
        rnd_pred_cri = tf.squeeze(rnd_pred_cri, axis=2)
    return rnd_targ_act,\
           rnd_pred_act, rnd_pred_act_reg, rnd_pred_act_dropout_mask_generator, rnd_pred_act_dropout_mask_phs,\
           rnd_targ_cri,\
           rnd_pred_cri, rnd_pred_cri_reg, rnd_pred_cri_dropout_mask_generator, rnd_pred_cri_dropout_mask_phs

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(400,300), activation=tf.nn.relu,
                     output_activation=tf.tanh, action_space=None,
                     dropout_rate=0, nn_type='mlp_variational'):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    if nn_type == 'mlp':
        with tf.variable_scope('pi'):
            pi = act_limit * mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
        with tf.variable_scope('q1'):
            q1 = tf.squeeze(mlp(tf.concat([x, a], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
        with tf.variable_scope('q2'):
            q2 = tf.squeeze(mlp(tf.concat([x, a], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
        with tf.variable_scope('q1', reuse=True):
            q1_pi = tf.squeeze(mlp(tf.concat([x, pi], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
    elif nn_type == 'mlp_dropout':
        with tf.variable_scope('pi'):
            pi = act_limit * mlp_dropout(x, list(hidden_sizes)+[act_dim], activation, output_activation)
        with tf.variable_scope('q'):
            q = tf.squeeze(mlp_dropout(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None, dropout_rate), axis=1)
        with tf.variable_scope('q', reuse=True):
            q_pi = tf.squeeze(mlp_dropout(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None, dropout_rate), axis=1)
    elif nn_type == 'mlp_variational':
        with tf.variable_scope('pi'):
            pi_in_dim = x.shape.as_list()[1]
            pi_dropout_mask_generator = DropoutMaskGenerator(pi_in_dim, hidden_sizes, model_prob=1.0 - dropout_rate)
            pi_dropout_mask_phs = pi_dropout_mask_generator.generate_dropout_mask_placeholders()

            pi, pi_reg = mlp_variational(x, pi_dropout_mask_phs, list(hidden_sizes) + [act_dim], activation, output_activation, dropout_rate)
            pi = act_limit * pi
        with tf.variable_scope('q1'):
            q1_in_ph = tf.concat([x, a], axis=-1)
            q1_in_dim = q1_in_ph.shape.as_list()[1]
            q1_dropout_mask_generator = DropoutMaskGenerator(q1_in_dim, hidden_sizes, model_prob=1.0 - dropout_rate)
            q1_dropout_mask_phs = q1_dropout_mask_generator.generate_dropout_mask_placeholders()

            q1, q1_reg = mlp_variational(q1_in_ph, q1_dropout_mask_phs, list(hidden_sizes) + [1],
                                         activation, None, dropout_rate)
            q1 = tf.squeeze(q1, axis=2)
        with tf.variable_scope('q1', reuse=True):
            q1_pi, q1_pi_reg = mlp_variational(tf.concat([x, pi[0]], axis=-1), q1_dropout_mask_phs, list(hidden_sizes) + [1],
                                                activation, None, dropout_rate)
            q1_pi = tf.squeeze(q1_pi, axis=2)
        with tf.variable_scope('q2'):
            q2_in_ph = tf.concat([x, a], axis=-1)
            q2_in_dim = q2_in_ph.shape.as_list()[1]
            q2_dropout_mask_generator = DropoutMaskGenerator(q2_in_dim, hidden_sizes, model_prob=1.0 - dropout_rate)
            q2_dropout_mask_phs = q2_dropout_mask_generator.generate_dropout_mask_placeholders()

            q2, q2_reg = mlp_variational(q2_in_ph, q2_dropout_mask_phs, list(hidden_sizes) + [1],
                                         activation, None, dropout_rate)
            q2 = tf.squeeze(q2, axis=2)
    else:
        raise ValueError('Please choose a proper nn_type!')
    return pi, pi_reg, pi_dropout_mask_generator, pi_dropout_mask_phs,\
           q1, q1_reg, q1_dropout_mask_generator, q1_dropout_mask_phs, q1_pi, q1_pi_reg,\
           q2, q2_reg, q2_dropout_mask_generator, q2_dropout_mask_phs
