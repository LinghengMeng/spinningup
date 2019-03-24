import numpy as np
import tensorflow as tf
import ray
import ray.experimental.tf_utils

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

class DropoutMaskGenerator:
    """Class used to generate dropout mask."""
    def __init__(self,x_dim, hidden_sizes=(32,), model_prob=0.9):
        self.x_dim = x_dim
        self.hidden_sizes = hidden_sizes
        self.model_prob = model_prob
    def generate_dropout_mask(self):
        new_dropout_masks = []
        for l, size in enumerate((self.x_dim, *self.hidden_sizes)):
            new_dropout_masks.append(np.random.binomial(1, self.model_prob, (size,)))
        return new_dropout_masks

@ray.remote
class PostSampler(object):
    def __init__(self,obs_dim, hidden_sizes=(400,300), activation=tf.nn.relu,
                     output_activation=tf.tanh, action_space=None,dropout_rate=0):
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.pi_dropout_mask_generator = DropoutMaskGenerator(obs_dim, hidden_sizes, model_prob=1.0 - dropout_rate)

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        # tf.reset_default_graph()
        with self.graph.as_default():
            self.pi_dropout_mask_phs = self._generate_dropout_mask_placeholders(obs_dim, hidden_sizes)
            self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim))
            with tf.variable_scope('post_sample_pi'):
                pi, self.pi_reg = mlp_variational(self.obs_ph, self.pi_dropout_mask_phs, list(hidden_sizes) + [act_dim],
                                                  activation,
                                                  output_activation, dropout_rate)
                self.pi = act_limit * pi
            self.init_vars = tf.global_variables_initializer()
            self.variables = ray.experimental.tf_utils.TensorFlowVariables(self.pi, self.sess)

        self.sess.run(self.init_vars)

    def _generate_dropout_mask_placeholders(self, x_dim, hidden_sizes=(32,)):
        dropout_mask_placeholders = []
        for l, size in enumerate((x_dim, *hidden_sizes)):
            dropout_mask_placeholders.append(tf.placeholder(dtype=tf.float32, shape=(size,),
                                                            name='dropout_mask_{}'.format(l)))
        return dropout_mask_placeholders

    def sample_action(self, weights, obs):
        # Set weights of current policy to post_sampler
        # Note: weights include variables for Adam optimizer from 'main/pi'
        var_weights_dict = {}
        for var_i, var_name in enumerate(self.variables.get_weights().keys()):
            var_weights_dict[var_name] = weights[var_i]
        self.variables.set_weights(var_weights_dict)

        pi_dropout_masks = self.pi_dropout_mask_generator.generate_dropout_mask()
        feed_dict={self.obs_ph: obs.reshape(1, -1)}
        for mask_i in range(len(self.pi_dropout_mask_phs)):
            feed_dict[self.pi_dropout_mask_phs[mask_i]] = pi_dropout_masks[mask_i]
        return self.sess.run(self.pi, feed_dict=feed_dict)[0]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

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
                     create_post_samplers=False, n_post=10, dropout_rate=0.1):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    obs_dim = x.shape.as_list()[-1]
    with tf.variable_scope('pi'):
        pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    # create n_post actor
    pi_post_samplers = []
    if create_post_samplers:
        pi_post_samplers = [PostSampler.remote(obs_dim, hidden_sizes,activation,output_activation,action_space, dropout_rate)
                            for i in range(n_post)]
    with tf.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, pi_post_samplers, q, q_pi