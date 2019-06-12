import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import tensorflow as tf
from spinup.algos.uncertainty_estimate.core import MLP, BeroulliDropoutMLP, BootstrappedEnsemble, get_vars, ReplayBuffer
from spinup.utils.logx import EpochLogger
import os.path as osp

"""
Investigate Uncertainty Estimation
"""
def uncertainty_estimate(seed=0, x_dim=2, y_dim = 2, hidden_sizes = [300, 300],
                         hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear,
                         x_low = -10, x_high = 10,
                         training_data_size=int(1e4), max_steps=int(1e6),
                         delayed_traininig=1000,
                         learning_rate=1e-3,
                         batch_size=100,
                         raw_batch_size=500, uncertainty_based_minibatch_sample=True,
                         replay_size=int(1e6),
                         BerDrop_n_post=50, bootstrapp_p=0.75,
                         logger_kwargs=dict()):
    """

    :param seed:
    :param x_dim:
    :param y_dim:
    :param hidden_sizes:
    :param x_low: lower bound of input
    :param x_high: upper bound of input
    :param max_steps:
    :param learning_rate:
    :param batch_size:
    :param replay_size:
    :param BerDrop_n_post: number of post samples for BernoulliDropoutMLP
    :param logger_kwargs:
    :return:
    """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Define input placeholder
    x_ph = tf.placeholder(dtype=tf.float32, shape=(None, x_dim))
    y_ph = tf.placeholder(dtype=tf.float32, shape=(None, y_dim))
    layer_sizes = hidden_sizes + [y_dim]

    # TODO: consider use an understandable target function rather than a random NN
    # 0. Create RandomTargetNetwork
    #       Note: initialize RNT weights far away from 0
    rtn = MLP(layer_sizes,
              kernel_initializer=tf.keras.initializers.random_uniform(minval=-0.8, maxval=0.8),
              bias_initializer=tf.keras.initializers.random_uniform(minval=-0.8, maxval=0.8),
              hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.sigmoid)
    rtn_y = rtn(x_ph) # target y

    # 1. Create MLP to learn RTN:
    #       which is only used for LazyBernoulliDropoutMLP.
    mlp_replay_buffer = ReplayBuffer(x_dim=x_dim, y_dim=y_dim, size=replay_size)
    kernel_regularizer = None#tf.keras.regularizers.l2(l=0.01)
    with tf.variable_scope('MLP'):
        mlp = MLP(layer_sizes, kernel_regularizer=kernel_regularizer,
                  hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.sigmoid)
        mlp_y = mlp(x_ph)

    mlp_loss = tf.reduce_mean((y_ph - mlp_y)**2) # mean-square-error
    mlp_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    mlp_train_op = mlp_optimizer.minimize(mlp_loss, var_list=get_vars('MLP'))

    # 2. Create BernoulliDropoutMLP：
    #       which is trained with dropout masks and regularization term
    with tf.variable_scope('BernoulliDropoutUncertaintyTrain'):
        bernoulli_dropout_mlp = BeroulliDropoutMLP(layer_sizes, weight_regularizer=1e-6, dropout_rate=0.05,
                                                   hidden_activation = tf.keras.activations.relu,
                                                   output_activation = tf.keras.activations.sigmoid)
        ber_drop_mlp_y = bernoulli_dropout_mlp(x_ph, training=True) # Must set training=True to use dropout mask

    ber_drop_mlp_reg_losses = tf.reduce_sum(
        tf.losses.get_regularization_losses(scope='BernoulliDropoutUncertaintyTrain'))
    ber_drop_mlp_loss = tf.reduce_mean(
        (y_ph - ber_drop_mlp_y) ** 2 + ber_drop_mlp_reg_losses)  # TODO: heteroscedastic loss
    ber_drop_mlp_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    ber_drop_mlp_train_op = ber_drop_mlp_optimizer.minimize(ber_drop_mlp_loss,
                                                            var_list=get_vars('BernoulliDropoutUncertaintyTrain'))
    # 3. Create lazy BernoulliDropoutMLP:
    #       which copys weights from MLP by
    #           sess.run(lazy_ber_drop_mlp_update)
    #       , then post sample predictions with dropout masks.
    with tf.variable_scope('LazyBernoulliDropoutUncertaintySample'):
        # define placeholder for parallel sampling
        #   batch x n_post x dim
        lazy_bernoulli_dropout_mlp = BeroulliDropoutMLP(layer_sizes, weight_regularizer=1e-6, dropout_rate=0.05,
                                                        hidden_activation=tf.keras.activations.relu,
                                                        output_activation=tf.keras.activations.sigmoid)
        lazy_ber_drop_mlp_y = lazy_bernoulli_dropout_mlp(x_ph, training=True) # Set training=True to sample with dropout masks
        lazy_ber_drop_mlp_update = tf.group([tf.assign(v_lazy_ber_drop_mlp, v_mlp)
                                             for v_mlp, v_lazy_ber_drop_mlp in
                                             zip(mlp.variables, lazy_bernoulli_dropout_mlp.variables)])

    # Create BootstrappedEnsembleNN
    with tf.variable_scope('BootstrappedEnsembleUncertainty'):
        boots_ensemble = BootstrappedEnsemble(ensemble_size=BerDrop_n_post, x_dim=x_dim, y_dim=y_dim, replay_size=replay_size,
                                              x_ph=x_ph, y_ph=y_ph, layer_sizes=layer_sizes,
                                              kernel_regularizer=kernel_regularizer,learning_rate=learning_rate)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def randomly_sampled_x(x_dim, x_low=-1, x_high=1):
        x = np.random.uniform(low=x_low, high=x_high, size=(1,x_dim))
        return x

    for step in range(int(max_steps)):
        x = randomly_sampled_x(x_dim, x_low=x_low, x_high=x_high)  # randomly select an input
        y = sess.run(rtn_y, feed_dict={x_ph: x.reshape(1, -1)})[0]  # Generate target for the selected input
        if step < int(training_data_size):
            # store the (input, target) to replay buffer
            mlp_replay_buffer.store(x, y)
            # add (x, y) to ensemble's replay buffer with probability bootstrapp_p
            boots_ensemble.add_to_replay_buffer(x, y, bootstrapp_p)

        # Post Sample and estimate uncertainty
        #   BernoulliDropoutMLP
        x_postSampling = np.matlib.repmat(x, BerDrop_n_post, 1) # repmat x for post sampling
        ber_drop_mlp_postSample = sess.run(ber_drop_mlp_y, feed_dict={x_ph: x_postSampling})

        ber_drop_mlp_pred = np.mean(ber_drop_mlp_postSample, axis=0)
        ber_drop_mlp_pred_error = np.linalg.norm((y - ber_drop_mlp_pred), ord=2)

        ber_drop_mlp_postSample_cov = np.cov(ber_drop_mlp_postSample, rowvar=False)
        ber_drop_mlp_postSample_unc = np.sum(np.diag(ber_drop_mlp_postSample_cov))

        #   LazyBernoulliDropoutMLP
        sess.run(lazy_ber_drop_mlp_update) # copy weights
        lazy_ber_drop_mlp_postSample = sess.run(lazy_ber_drop_mlp_y, feed_dict={x_ph: x_postSampling})

        lazy_ber_drop_mlp_pred = np.mean(lazy_ber_drop_mlp_postSample, axis=0)
        lazy_ber_drop_mlp_pred_error = np.linalg.norm((y - lazy_ber_drop_mlp_pred), ord=2)

        lazy_ber_drop_mlp_postSample_cov = np.cov(lazy_ber_drop_mlp_postSample, rowvar=False)
        lazy_ber_drop_mlp_postSample_unc = np.sum(np.diag(lazy_ber_drop_mlp_postSample_cov))

        #   BootstrappedEnsemble
        boots_ensemble_preds = boots_ensemble.prediction(sess, x)

        boots_ensemble_preds_pred = np.mean(boots_ensemble_preds, axis=0)
        boots_ensemble_pred_error = np.linalg.norm((y - boots_ensemble_preds_pred), ord=2)

        boots_ensemble_preds_cov = np.cov(boots_ensemble_preds, rowvar=False)
        boots_ensemble_preds_unc = np.sum(np.diag(boots_ensemble_preds_cov))

        # Training
        if step > raw_batch_size:
            if uncertainty_based_minibatch_sample:
                # Resample based on uncertainty rank
                raw_batch = mlp_replay_buffer.sample_batch(raw_batch_size)
                raw_batch_repmat = np.reshape(np.matlib.repmat(raw_batch['x'], 1, BerDrop_n_post),
                                              (raw_batch_size * BerDrop_n_post, x_dim))
                raw_batch_repmat_postSample = sess.run(ber_drop_mlp_y, feed_dict={x_ph: raw_batch_repmat})
                raw_batch_repmat_postSample_reshape = np.reshape(raw_batch_repmat_postSample,
                                                                 (raw_batch_size, BerDrop_n_post, y_dim))
                uncertainty_rank = np.zeros((raw_batch_size,))
                for i_batch in range(raw_batch_size):
                    uncertainty_rank[i_batch] = np.sum(
                        np.diag(np.cov(raw_batch_repmat_postSample_reshape[i_batch], rowvar=False)))
                # Find top_n highest uncertainty samples
                top_batch_size_highest_uncertainty_indices = np.argsort(uncertainty_rank)[-batch_size:]
                mlp_batch = {}
                mlp_batch['x'] = raw_batch['x'][top_batch_size_highest_uncertainty_indices]
                mlp_batch['y'] = raw_batch['y'][top_batch_size_highest_uncertainty_indices]
            else:
                mlp_batch = mlp_replay_buffer.sample_batch(batch_size)

            # Train MLP
            mlp_outs = sess.run([mlp_loss, mlp_train_op], feed_dict={x_ph: mlp_batch['x'], y_ph: mlp_batch['y']})

            # Train BernoulliDropoutMLP on the same batch with MLP
            ber_drop_outs = sess.run([ber_drop_mlp_loss, ber_drop_mlp_train_op], feed_dict={x_ph: mlp_batch['x'],
                                                                                   y_ph: mlp_batch['y']})

            # Train BootstrappedEnsemble
            boots_ensemble_loss = boots_ensemble.train(sess, raw_batch_size, batch_size, uncertainty_based_minibatch_sample)

            # Log data
            logger.log_tabular('Step', step)
            logger.log_tabular('MLPLoss', mlp_outs[0])

            logger.log_tabular('LazyBernDropPredError', lazy_ber_drop_mlp_pred_error)
            logger.log_tabular('LazyBernDropUnc', lazy_ber_drop_mlp_postSample_unc)
            for i, v in enumerate(lazy_ber_drop_mlp_postSample_cov.flatten(order='C')):
                logger.log_tabular('{}_{}'.format('lazy_ber_drop_mlp_postSample_cov', i), v)

            logger.log_tabular('BernDropPredError', ber_drop_mlp_pred_error)
            logger.log_tabular('BernDropUnc', ber_drop_mlp_postSample_unc)
            for i, v in enumerate(ber_drop_mlp_postSample_cov.flatten(order='C')):
                logger.log_tabular('{}_{}'.format('ber_drop_mlp_postSample_cov', i), v)
            logger.log_tabular('BernDropLoss', ber_drop_outs[0])

            logger.log_tabular('BootstrappedEnsemblePredError', boots_ensemble_pred_error)
            logger.log_tabular('BootstrappedEnsembleUnc', boots_ensemble_preds_unc)
            for i, v in enumerate(boots_ensemble_preds_cov.flatten(order='C')):
                logger.log_tabular('{}_{}'.format('boots_ensemble_preds_cov', i), v)
            logger.log_tabular('BootsEnsembleAvgLoss', boots_ensemble_loss.mean())

            if step % 1000 == 0:
                print_data = True
            else:
                print_data = False
            logger.dump_tabular(print_data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--exp_name', type=str, default='uncertainty_estimate')
    parser.add_argument('--x_dim', type=int, default=2)
    parser.add_argument('--y_dim', type=int, default=5)
    parser.add_argument('--BerDrop_n_post', type=int, default=100)
    parser.add_argument('--training_data_size', type=int, default=4e4)
    parser.add_argument('--uncertainty_based_minibatch_sample', action="store_true")
    parser.add_argument('--max_steps', type=int, default=4e4)
    parser.add_argument('--delayed_traininig', type=int, default=1000)
    args = parser.parse_args()

    # Set log data saving directory
    from spinup.utils.run_utils import setup_logger_kwargs
    data_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))),
                        'spinup_data')
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)

    uncertainty_estimate(seed=args.seed,
                         x_dim=args.x_dim, y_dim=args.y_dim,
                         BerDrop_n_post=args.BerDrop_n_post,
                         training_data_size=args.training_data_size,
                         uncertainty_based_minibatch_sample=args.uncertainty_based_minibatch_sample,
                         max_steps=args.max_steps,
                         delayed_traininig=args.delayed_traininig,
                         logger_kwargs=logger_kwargs)
