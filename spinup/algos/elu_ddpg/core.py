import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pdb
import sys


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float64, shape=(None,dim) if dim else (None,))

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
    # alpha, beta and rho for covariance matrix
    a_alpha = tf.layers.dense(x, units=act_dim, activation=tf.tanh)
    a_beta = tf.layers.dense(x, units=int(act_dim*(act_dim-1)/2), activation=tf.tanh)

    # Create MultivariateGaussian
    # Shift a_alpha and a_beta to (-8,0) so that e^a_alpha and e^a_beta and in (0,1).
    a_alpha = 4*a_alpha - 4
    a_cov = construct_covariance_matrix(a_alpha, a_beta, act_dim)

    a = tfp.edward2.MultivariateNormalFullCovariance(a_mu, a_cov)
    a = tf.tanh(a) # shift back to [-1,1]
    return a, a_mu, a_alpha, a_beta, a_cov

# def construct_covariance_matrix(sigma, rho, act_dim):
#     """
#     Construct covariance and correlation matrix
#     :param sigma:
#     :param rho:
#     :return: covariance and correlation
#     """
#
#     # 1. recontruct correlation matrix
#     w_rho = generate_weights(act_dim)
#     tol_value = tf.constant([np.sqrt(act_dim) * 1e-4], dtype=tf.float64)
#     dist = tf.multiply(tf.ones(tf.shape(sigma)[0], dtype=tf.float64), tf.sqrt(tf.cast(act_dim, tf.float64)))
#     diag_vec = tf.zeros(tf.shape(sigma), dtype=tf.float64)
#     A = tf.reshape(tf.matmul(rho, w_rho), [tf.shape(sigma)[0], act_dim, act_dim])
#     A = tf.linalg.set_diag(A, diag_vec)
#
#     loop_vars = (tol_value, dist, diag_vec, A)
#     # pdb.set_trace()
#     def loop_cond(tol_value, dist, diag_vec, A):
#         return tf.reduce_all(tf.greater(dist, tol_value))
#
#     def loop_body(tol_value, dist, diag_vec, A):
#         diag_delta = tf.log(tf.linalg.diag_part(tf.linalg.expm(A)))
#         diag_vec = tf.subtract(diag_vec, diag_delta)
#         A = tf.linalg.set_diag(A, diag_vec)
#         dist = tf.norm(diag_delta, axis=1)
#         return tol_value, dist, diag_vec, A
#
#     loop_vars = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=loop_vars,
#                               back_prop=False, return_same_structure=True)
#
#     C = tf.linalg.expm(loop_vars[-1])
#     C = tf.linalg.set_diag(C, tf.ones(tf.shape(sigma), dtype=tf.float64))
#
#     # 2. recontruct covariance matrix
#     tmp_sigma1 = tf.reshape(sigma,shape=[tf.shape(sigma)[0], act_dim, 1])
#     tmp_sigma2 = tf.reshape(sigma,shape=[tf.shape(sigma)[0], 1, act_dim])
#     tmp_sigma = tf.einsum('bij,bjk->bik', tmp_sigma1, tmp_sigma2)
#     S = tf.multiply(C, tmp_sigma, name='reconstruct_variance_mat')
#
#     # 3. find nearest positive definite matrix if is not positive definite
#     # pdb.set_trace()
#     epsilon = tf.constant(1e-6, dtype=tf.float64)
#     S_eigval, S_eigvec = tf.linalg.eigh(S)
#     val = tf.reshape(tf.maximum(S_eigval, epsilon), shape=[tf.shape(S)[0], 1, act_dim])
#     T = 1 / tf.einsum('bij,bjk->bik', tf.multiply(S_eigvec, S_eigvec), tf.transpose(val, perm=[0, 2, 1]))
#     T = tf.sqrt(
#         tf.linalg.set_diag(tf.zeros(tf.shape(S), dtype=tf.float64), tf.reshape(T, shape=[tf.shape(S)[0], act_dim])))
#     B = tf.einsum('bij,bjk->bik', tf.einsum('bij,bjk->bik', T, S_eigvec),
#                   tf.linalg.set_diag(tf.zeros(tf.shape(S), dtype=tf.float64),
#                                      tf.reshape(tf.sqrt(val), shape=[tf.shape(S)[0], act_dim])))
#     S_psd = tf.einsum('bij,bjk->bik', B, tf.transpose(B, perm=[0, 2, 1]))
#     tf.print(tf.linalg.eigvalsh(S_psd), output_stream=sys.stderr)
#     return S_psd, C
def construct_covariance_matrix(alpha, beta, act_dim):
    """
    Construct covariance and correlation matrix
    :param alpha:
    :param beta:
    :return: covariance and correlation
    """

    # 1. recontruct correlation matrix
    w_beta = generate_weights(act_dim)
    diag_vec = tf.ones(tf.shape(alpha), dtype=tf.float64)
    A = tf.reshape(tf.matmul(beta, w_beta), [tf.shape(alpha)[0], act_dim, act_dim])
    A = tf.linalg.set_diag(A, diag_vec)

    # 2. recontruct covariance matrix
    tmp_sigma1 = tf.reshape(tf.exp(alpha/2),shape=[tf.shape(alpha)[0], act_dim, 1])
    tmp_sigma2 = tf.reshape(tf.exp(alpha/2),shape=[tf.shape(alpha)[0], 1, act_dim])
    tmp_sigma = tf.einsum('bij,bjk->bik', tmp_sigma1, tmp_sigma2)
    C = tf.multiply(A, tmp_sigma, name='reconstruct_variance_mat')
    # Add a small positive number to avoid semi-positive covariance matrix
    C = C + 1e-8 * tf.eye(num_rows=act_dim, batch_shape=[tf.shape(C)[0]], dtype=tf.float64)
    return C

def generate_weights(act_dim):
    """Generate untrainable weights for creating covariance matrix."""
    k_rho = 0
    w_rho = np.zeros((int(act_dim*(act_dim-1)/2), act_dim**2))
    for i in range(act_dim):
        for j in range(act_dim):
            if i != j and i < j:
                w_rho[k_rho, [i*act_dim+j, j*act_dim+i]] = 1
                k_rho += 1
    return tf.convert_to_tensor(w_rho,dtype=tf.float64)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

"""
Actor-Critics
"""
def mlp_actor_critic(x, a_mu_ph, a_alpha_ph, a_beta_ph, hidden_sizes=(400,300), activation=tf.nn.relu,
                     output_activation=tf.tanh, action_space=None):
    act_dim = a_mu_ph.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        # pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
        # import pdb; pdb.set_trace()
        pi, pi_mu, pi_alpha, pi_beta, pi_cov = a_mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
        pi = act_limit * pi
        pi_mu = act_limit * pi_mu
    with tf.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x, a_mu_ph, a_alpha_ph, a_beta_ph], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
        # q = q_mlp(tf.concat([x, a], axis=-1), list(hidden_sizes)+[1], activation, None)
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x, pi_mu, pi_alpha, pi_beta], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
        # q_pi = q_mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None)
    return pi, pi_mu, pi_alpha, pi_beta, pi_cov, q, q_pi
