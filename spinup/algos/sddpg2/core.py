import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
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
    # sigma and rho for information matrix
    a_sigma = tf.layers.dense(x, units=act_dim, activation=tf.sigmoid)
    a_rho = tf.layers.dense(x, units=(act_dim*(act_dim-1))/2, activation=tf.tanh)
    # Create MultivariateGaussian
    a_rho = tf.multiply(a_rho, tf.constant([5], dtype=tf.float64))  # scale rho to make correlation in range [-1,1]
    a_cov, a_corr, a_corr_iter_number = construct_covariance_matrix(a_sigma, a_rho, act_dim)

    a = tfp.edward2.MultivariateNormalFullCovariance(a_mu, a_cov)
    a = tf.tanh(a) # shift back to [-1,1]
    return a, a_mu, a_sigma, a_rho, a_cov, a_corr_iter_number

def construct_covariance_matrix(sigma, rho, act_dim):
    """
    Construct covariance and correlation matrix
    :param sigma:
    :param rho:
    :return: covariance and correlation
    """

    # 1. recontruct correlation matrix
    tol_value = tf.constant([np.sqrt(act_dim) * 1e-4], dtype=tf.float64)
    iter_number = tf.constant([0])
    dist = tf.multiply(tf.ones(tf.shape(sigma)[0], dtype=tf.float64), tf.sqrt(tf.cast(act_dim, tf.float64)))
    diag_vec = tf.zeros(tf.shape(sigma), dtype=tf.float64)
    w_rho = generate_weights(act_dim)
    A = tf.reshape(tf.matmul(rho, w_rho), [tf.shape(sigma)[0], act_dim, act_dim])
    A = tf.linalg.set_diag(A, diag_vec)

    # TODO: speed up reconstructing correlation matrix
    loop_vars = (tol_value, iter_number, dist, diag_vec, A)
    # pdb.set_trace()
    def loop_cond(tol_value, iter_number, dist, diag_vec, A):
        # Crutial Note: Repeat body while the condition cond is true.
        cond = tf.greater(dist, tol_value)
        return tf.reduce_any(cond)

    def loop_body(tol_value, iter_number, dist, diag_vec, A):
        iter_number = tf.Print(iter_number, [iter_number], 'Printing iter_number:')
        iter_number = iter_number + 1
        diag_delta = tf.log(tf.linalg.diag_part(tf.linalg.expm(A)))
        diag_vec = tf.subtract(diag_vec, diag_delta)
        A = tf.linalg.set_diag(A, diag_vec)
        dist = tf.norm(diag_delta, axis=1)
        # dist = tf.Print(dist, [dist, iter_number], 'Printing dist:')
        return tol_value, iter_number, dist, diag_vec, A

    loop_vars = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=loop_vars,
                              parallel_iterations=1,
                              back_prop=False, return_same_structure=True)

    C = tf.linalg.expm(loop_vars[-1])
    C = tf.linalg.set_diag(C, tf.ones(tf.shape(sigma), dtype=tf.float64))

    # # 2. find nearest positive definite matrix if is not positive definite
    # C = nearest_correlation_matrix(C, threshold=1e-15)
    k = tol_value*2
    C = C + k * tf.eye(num_rows=act_dim, batch_shape=[tf.shape(C)[0]], dtype=tf.float64)

    # 3. recontruct covariance matrix
    tmp_sigma1 = tf.reshape(sigma,shape=[tf.shape(sigma)[0], act_dim, 1])
    tmp_sigma2 = tf.reshape(sigma,shape=[tf.shape(sigma)[0], 1, act_dim])
    tmp_sigma = tf.einsum('bij,bjk->bik', tmp_sigma1, tmp_sigma2)
    S = tf.multiply(C, tmp_sigma, name='reconstruct_variance_mat')

    return S, C, iter_number

def nearest_correlation_matrix(corr, threshold=1e-15):
    """Find the nearest correlation matrix that is positive (semi-)definite."""
    # TODO: it's confirmed time consuming op is "tf.linalg.eigh".
    threshold = tf.constant(threshold, dtype=tf.float64)
    evals, evecs = tf.linalg.eigh(corr)
    need_clip = tf.reduce_any(evals < 0, axis=1)
    tf.print(need_clip, output_stream=sys.stdout)
    def true_fn():
        shifted_evals = tf.tile(tf.reshape(tf.maximum(evals, threshold),
                                           shape=[tf.shape(corr)[0], 1, tf.shape(corr)[1]]),
                                [1, tf.shape(corr)[1], 1])
        corr_new = tf.einsum('bij,bjk->bik', tf.multiply(evecs, shifted_evals), tf.transpose(evecs, perm=[0, 2, 1]))
        return corr_new

    corr = tf.cond(tf.reduce_any(need_clip), true_fn=true_fn, false_fn=lambda : corr)

    # corr = tf.zeros(tf.shape(corr), dtype=tf.float64)
    # corr = tf.linalg.set_diag(corr, tf.ones([tf.shape(corr)[0],tf.shape(corr)[1]], dtype=tf.float64))

    return corr


# def construct_covariance_matrix(sigma, rho, act_dim):
#     """
#     Construct covariance and correlation matrix
#     :param sigma:
#     :param rho:
#     :return: covariance and correlation
#     """
#     w_rho = generate_weights(act_dim)
#     x_old = tf.ones(tf.shape(sigma),dtype=tf.float64)
#     x_new = tf.zeros(tf.shape(sigma), dtype=tf.float64)
#     precision = tf.constant([1e-20], dtype=tf.float64)
#     A = tf.reshape(tf.matmul(rho, w_rho), [tf.shape(sigma)[0], act_dim, act_dim])
#
#     loop_vars = (x_old, x_new, precision, A)
#
#     def loop_cond(x_old, x_new, precision, A):
#         # x_old = tf.floormod(x_old, precision)
#         # x_new = tf.floormod(x_new, precision)
#         return tf.reduce_all(tf.equal(x_old, x_new))
#
#     def loop_body(x_old, x_new, precision, A):
#         x_old = x_new
#         A = tf.linalg.set_diag(A, x_old)
#         x_new = x_old - tf.log(tf.linalg.diag_part(tf.linalg.expm(A)))
#         return x_old, x_new, precision, A
#
#     loop_vars = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=loop_vars,
#                               back_prop=False, return_same_structure=True)
#
#     # recontruct correlation matrix and covariance matrix
#     C = tf.linalg.expm(loop_vars[-1])
#     C = tf.linalg.set_diag(C, tf.ones(tf.shape(sigma),dtype=tf.float64))
#     tmp_sigma1 = tf.reshape(sigma,shape=[tf.shape(sigma)[0], act_dim, 1])
#     tmp_sigma2 = tf.reshape(sigma,shape=[tf.shape(sigma)[0], 1, act_dim])
#     tmp_sigma = tf.einsum('bij,bjk->bik', tmp_sigma1, tmp_sigma2)
#     S = tf.multiply(C, tmp_sigma)
#     return S, C

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
def mlp_actor_critic(x, a_ph, hidden_sizes=(400,300), activation=tf.nn.relu,
                     output_activation=tf.tanh, action_space=None):
    act_dim = a_ph.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        # pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
        # import pdb; pdb.set_trace()
        pi, pi_mu, pi_sigma, pi_rho, pi_cov, pi_corr_iter_number = a_mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
        pi = act_limit * pi
        pi_mu = act_limit * pi_mu
    with tf.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x, a_ph], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
        # q = q_mlp(tf.concat([x, a], axis=-1), list(hidden_sizes)+[1], activation, None)
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x, pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
        # q_pi = q_mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None)
    return pi, pi_mu, pi_sigma, pi_rho, pi_cov, pi_corr_iter_number, q, q_pi
