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

def q_mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    q_mu = tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
    q_sigma = tf.layers.dense(x, units=hidden_sizes[-1], activation=tf.nn.relu)
    q_mu = tf.squeeze(q_mu, axis=1)
    q_sigma = tf.squeeze(q_sigma, axis=1)
    q_dist = tf.distributions.Normal(loc=q_mu, scale=q_sigma)
    q = q_dist.sample([1])
    import pdb;
    pdb.set_trace()
    return q

def a_mlp(x, hidden_sizes=(32,), activation=tf.nn.relu, output_activation=tf.tanh):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    act_dim = hidden_sizes[-1]
    a_mu = tf.layers.dense(x, units=act_dim, activation=output_activation)
    a_sigma = tf.layers.dense(x, units=act_dim, activation=tf.sigmoid)
    conc_factor = 1
    a_dist = tf.distributions.Normal(a_mu, conc_factor*a_sigma)
    a = a_dist.sample([1])[0]
    a = tf.tanh(a)
    return a,a_mu,a_sigma

def sample_action(a_mu, a_alpha, a_beta, concen_factor=1):
    act_dim = len(a_mu)
    # Shift a_alpha to range [-10,0] so that exp(a_alpha) in [4e-5,1]
    a_alpha = (a_alpha-1)*5
    a_cov = np.zeros((act_dim, act_dim))
    beta_ind = 0
    for i in range(act_dim):
        a_cov[i,i] = np.exp(a_alpha[i])
        if i+1 < act_dim:
            for j in range(i+1, act_dim):
                tmp = np.exp((a_alpha[i]+a_alpha[j])/2) * a_beta[beta_ind]
                a_cov[i,j] = tmp
                a_cov[j,i] = tmp
    a_cov = concen_factor * a_cov
    return np.random.multivariate_normal(a_mu, a_cov, 1)[0], a_cov

def sample_action_op(a_mu, a_para, act_dim):
    # alpha and beta for information matrix: (act_dim * (act_dim+1)) / 2
    # a_alpha[0:act_dim] = alpha
    # a_alpha[act_dim:(act_dim * (act_dim+1)) / 2] = beta
    a_alpha = a_para[:, :act_dim]
    a_beta = a_para[:, act_dim:]
    a_info_matrix = np.zeros((a_para.shape[0], act_dim, act_dim))
    beta_ind = 0
    for i in range(act_dim):
        a_info_matrix[:, i, i] = tf.exp(a_alpha[:,i])
        if i+1 < act_dim:
            for j in range(i+1, act_dim):
                a_info_matrix[:, i, j] = tf.exp((a_alpha[:,i]+a_alpha[:,j])/2) * tf.tanh(a_beta[:, beta_ind])
                a_info_matrix[:, j, i] = tf.exp((a_alpha[:,i] + a_alpha[:,j]) / 2) * tf.tanh(a_beta[:, beta_ind])
                beta_ind += 1
    a_dist = tf.contrib.distributions.MultivariateNormalFullCovariance(a_mu, tf.linalg.inv(a_info_matrix))
    a = a_dist.sample([1])
    return a, a_info_matrix

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

"""
Actor-Critics
"""
def mlp_actor_critic(x, a_ph, hidden_sizes=(400,300), activation=tf.nn.relu,
                     output_activation=tf.tanh, action_space=None):
    act_dim = a_ph.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        # pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
        # import pdb; pdb.set_trace()
        pi, pi_mu, pi_sigma = a_mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
        pi = act_limit * pi
        pi_mu = act_limit * pi_mu
    with tf.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x, a_ph], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
        # q = q_mlp(tf.concat([x, a], axis=-1), list(hidden_sizes)+[1], activation, None)
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x, pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q', reuse=True):
        q_pi_mu = tf.squeeze(mlp(tf.concat([x, pi_mu], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
        # q_pi = q_mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None)
    return pi, pi_mu, pi_sigma, q, q_pi, q_pi_mu
