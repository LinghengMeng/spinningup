import numpy as np
import tensorflow as tf

class VariationalDense:
    """Variational Dense Layer Class"""
    def __init__(self, n_in, n_out, dropout_mask,
                 model_prob=0.9, model_lam=1e-2, activation=None, name="hidden"):
        self.model_prob = model_prob    # probability to keep units
        self.model_lam = model_lam      # l^2 / 2*tau
        self.dropout_mask = dropout_mask

        if activation is None:
            self.activation = tf.identity
        else:
            self.activation = activation

        kernel_initializer = tf.initializers.truncated_normal(mean=0.0, stddev=0.01)
        self.model_M = tf.get_variable("{}_M".format(name), initializer=kernel_initializer([n_in, n_out])) # variational parameters
        self.model_m = tf.get_variable("{}_b".format(name), initializer=tf.zeros([n_out]))

        self.model_W = tf.matmul(tf.diag(self.dropout_mask), self.model_M)

    def __call__(self, X):
        output = self.activation(tf.matmul(X, self.model_W) + self.model_m)
        return output

    @property
    def regularization(self):
        return self.model_lam * (
            self.model_prob * tf.reduce_sum(tf.square(self.model_M)) +
            tf.reduce_sum(tf.square(self.model_m))
        )

def generate_dropout_mask_placeholders(x_dim, hidden_sizes=(32,)):
    dropout_mask_placeholders = []
    for l, size in enumerate((x_dim, *hidden_sizes)):
        dropout_mask_placeholders.append(tf.placeholder(dtype=tf.float32, shape=(size,),
                                                        name='dropout_mask_{}'.format(l)))
    return dropout_mask_placeholders

class DropoutMaskGenerator:
    """Class used to generate dropout mask."""
    def __init__(self,x_dim, hidden_sizes=(32,), model_prob=0.9):
        self.x_dim = x_dim
        self.hidden_sizes = hidden_sizes
        self.model_prob = model_prob

    def generate_dropout_mask_placeholders(self):
        dropout_mask_placeholders = []
        for l, size in enumerate((self.x_dim, *self.hidden_sizes)):
            dropout_mask_placeholders.append(tf.placeholder(dtype=tf.float32, shape=(size,),
                                                            name='dropout_mask_{}'.format(l)))
        return dropout_mask_placeholders

    def generate_dropout_mask(self):
        new_dropout_masks = []
        for l, size in enumerate((self.x_dim, *self.hidden_sizes)):
            new_dropout_masks.append(np.random.binomial(1, self.model_prob, (size,)))
        return new_dropout_masks

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp_variational(x, dropout_mask_phs, hidden_sizes=(32,),
                    activation=tf.tanh, output_activation=None, dropout_rate=0.1):
    # Hidden layers
    regularization = 0
    for l, h in enumerate(hidden_sizes[:-1]):
        hidden_layer = VariationalDense(n_in=x.shape.as_list()[1],
                                        n_out=h,
                                        dropout_mask = dropout_mask_phs[l],
                                        model_prob=1.0-dropout_rate,
                                        model_lam=1e-2,
                                        activation=activation,
                                        name="h{}".format(l+1))
        x = hidden_layer(x)
        regularization += hidden_layer.regularization
    # Output layer
    out_layer = VariationalDense(n_in=x.shape.as_list()[1],
                                 n_out=hidden_sizes[-1],
                                 dropout_mask=dropout_mask_phs[-1],
                                 model_prob=1.0-dropout_rate,
                                 model_lam=1e-2,
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

"""
Random Network Distillation
"""
def random_net_distill(x_ph, a_ph,  hidden_sizes=(400,300), activation=tf.nn.relu,
                       output_activation=tf.tanh, action_space=None):
    act_dim = a_ph.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('rnd_targ_act'):
        rnd_targ_act = act_limit * mlp(x_ph, list(hidden_sizes) + [act_dim], activation, output_activation)
    with tf.variable_scope('rnd_pred_act'):
        rnd_pred_act = act_limit * mlp(x_ph, list(hidden_sizes) + [act_dim], activation, output_activation)
    with tf.variable_scope('rnd_targ_cri'):
        rnd_targ_cri = tf.squeeze(mlp(tf.concat([x_ph, a_ph], axis=-1), list(hidden_sizes) + [1], activation, None), axis=1)
    with tf.variable_scope('rnd_pred_cri'):
        rnd_pred_cri = tf.squeeze(mlp(tf.concat([x_ph, a_ph], axis=-1), list(hidden_sizes) + [1], activation, None),
                                  axis=1)
    return rnd_targ_act, rnd_pred_act, rnd_targ_cri, rnd_pred_cri

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
            q1 = tf.squeeze(q1, axis=1)
        with tf.variable_scope('q1', reuse=True):
            q1_pi, q1_pi_reg = mlp_variational(tf.concat([x, pi], axis=-1), q1_dropout_mask_phs, list(hidden_sizes) + [1],
                                                activation, None, dropout_rate)
            q1_pi = tf.squeeze(q1_pi, axis=1)
        with tf.variable_scope('q2'):
            q2_in_ph = tf.concat([x, a], axis=-1)
            q2_in_dim = q2_in_ph.shape.as_list()[1]
            q2_dropout_mask_generator = DropoutMaskGenerator(q2_in_dim, hidden_sizes, model_prob=1.0 - dropout_rate)
            q2_dropout_mask_phs = q2_dropout_mask_generator.generate_dropout_mask_placeholders()

            q2, q2_reg = mlp_variational(q2_in_ph, q2_dropout_mask_phs, list(hidden_sizes) + [1],
                                         activation, None, dropout_rate)
            q2 = tf.squeeze(q2, axis=1)
    else:
        raise ValueError('Please choose a proper nn_type!')

    return pi, pi_reg, pi_dropout_mask_generator, pi_dropout_mask_phs,\
           q1, q1_reg, q1_dropout_mask_generator, q1_dropout_mask_phs, q1_pi, q1_pi_reg,\
           q2, q2_reg, q2_dropout_mask_generator, q2_dropout_mask_phs
