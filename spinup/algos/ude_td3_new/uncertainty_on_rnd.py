import numpy as np
import tensorflow as tf
import time

from spinup.algos.ude_td3_new.core import MLP, BeroulliDropoutMLP, BootstrappedEnsemble, get_vars, count_vars
from spinup.algos.ude_td3_new.replay_buffer import ReplayBuffer, RandomNetReplayBuffer
from spinup.utils.logx import EpochLogger, Logger

class UncertaintyOnRandomNetwork(object):
    def __init__(self, x_dim, y_dim, hidden_sizes, post_sample_size,
                 logger_kwargs, loger_file_name='unc_on_random_net.txt'):
        """

        :param x_dim: input size
        :param y_dim: output size
        :param hidden_sizes: hidden layer sizes
        """
        self.replay_size = int(1e6)
        self.learning_rate = 1e-3
        self.mlp_kernel_regularizer = None  # tf.keras.regularizers.l2(l=0.01)
        self.bernoulli_dropout_weight_regularizer = 1e-6
        self.dropout_rate = 0.05
        self.ensemble_size = post_sample_size
        self.post_sample_size = post_sample_size
        self.batch_size = 100
        self.bootstrapp_p = 0.75 # probability used to add to replay buffer

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.layer_sizes = hidden_sizes + [y_dim]

        self.x_ph = tf.placeholder(dtype=tf.float32, shape=(None, x_dim))
        self.y_ph = tf.placeholder(dtype=tf.float32, shape=(None, y_dim))

        ######################################################################################################
        # Define random target network
        #       Note: initialize RNT weights far away from 0 and keep fixed
        random_net = MLP(self.layer_sizes,
                         kernel_initializer=tf.keras.initializers.random_uniform(minval=-0.8, maxval=0.8),
                         bias_initializer=tf.keras.initializers.random_uniform(minval=-0.8, maxval=0.8),
                         hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.sigmoid)
        self.random_net_y = random_net(self.x_ph)  # target y

        ######################################################################################################
        # Define LazyBernoulliDropout MLP
        # 1. Create MLP to learn RTN: which is only used for LazyBernoulliDropoutMLP.
        self.mlp_replay_buffer = RandomNetReplayBuffer(self.x_dim, self.y_dim, size=self.replay_size)
        with tf.variable_scope('MLP'):
            mlp = MLP(self.layer_sizes, kernel_regularizer=self.mlp_kernel_regularizer,
                      hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.sigmoid)
            mlp_y = mlp(self.x_ph)

        self.mlp_loss = tf.reduce_mean((self.y_ph - mlp_y) ** 2)  # mean-square-error
        mlp_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.mlp_train_op = mlp_optimizer.minimize(self.mlp_loss, var_list=get_vars('MLP'))
        # 2. Create lazy BernoulliDropoutMLP:
        #       which copys weights from MLP by
        #           sess.run(lazy_ber_drop_mlp_update)
        #       , then post sample predictions with dropout masks.
        with tf.variable_scope('LazyBernoulliDropoutUncertaintySample'):
            # define placeholder for parallel sampling
            #   batch x n_post x dim
            lazy_bernoulli_dropout_mlp = BeroulliDropoutMLP(self.layer_sizes, weight_regularizer=1e-6,
                                                            dropout_rate=self.dropout_rate,
                                                            hidden_activation=tf.keras.activations.relu,
                                                            output_activation=tf.keras.activations.sigmoid)
            self.lazy_ber_drop_mlp_y = lazy_bernoulli_dropout_mlp(self.x_ph,
                                                                  training=True)  # Set training=True to sample with dropout masks
            self.lazy_ber_drop_mlp_update = tf.group([tf.assign(v_lazy_ber_drop_mlp, v_mlp)
                                                        for v_mlp, v_lazy_ber_drop_mlp in
                                                        zip(mlp.variables, lazy_bernoulli_dropout_mlp.variables)])

        ######################################################################################################
        # Define BernoulliDropout MLP：
        #       which is trained with dropout masks and regularization term
        with tf.variable_scope('BernoulliDropoutUncertaintyTrain'):
            bernoulli_dropout_mlp = BeroulliDropoutMLP(self.layer_sizes,
                                                       weight_regularizer=self.bernoulli_dropout_weight_regularizer,
                                                       dropout_rate=self.dropout_rate,
                                                       hidden_activation=tf.keras.activations.relu,
                                                       output_activation=tf.keras.activations.sigmoid)
            self.ber_drop_mlp_y = bernoulli_dropout_mlp(self.x_ph,
                                                        training=True)  # Must set training=True to use dropout mask

        ber_drop_mlp_reg_losses = tf.reduce_sum(
            tf.losses.get_regularization_losses(scope='BernoulliDropoutUncertaintyTrain'))
        self.ber_drop_mlp_loss = tf.reduce_mean(
            (self.y_ph - self.ber_drop_mlp_y) ** 2 + ber_drop_mlp_reg_losses)  # TODO: heteroscedastic loss
        ber_drop_mlp_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.ber_drop_mlp_train_op = ber_drop_mlp_optimizer.minimize(self.ber_drop_mlp_loss,
                                                                     var_list=get_vars(
                                                                         'BernoulliDropoutUncertaintyTrain'))
        ######################################################################################################
        # Define BootstrappedEnsemble
        # Create BootstrappedEnsembleNN
        with tf.variable_scope('BootstrappedEnsembleUncertainty'):
            self.boots_ensemble = BootstrappedEnsemble(ensemble_size=self.ensemble_size,
                                                       x_dim=self.x_dim, y_dim=self.y_dim,
                                                       replay_size=self.replay_size,
                                                       x_ph=self.x_ph, y_ph=self.y_ph,
                                                       layer_sizes=self.layer_sizes,
                                                       kernel_regularizer=self.mlp_kernel_regularizer,
                                                       learning_rate=self.learning_rate)
        ######################################################################################################
        # Define logger
        self.uncertainty_on_random_net_logger = Logger(output_fname=loger_file_name, **logger_kwargs)

    def get_predError_and_uncerEstimate_on_policy_based_input(self, input, sess, t, start_time):
        x = input
        # Generate target for the selected input
        y = sess.run(self.random_net_y, feed_dict={self.x_ph: x.reshape(1, -1)})[0]

        # store the (input, target) to replay buffer
        self.mlp_replay_buffer.store(x, y)
        # add (x, y) to ensemble's replay buffer with probability bootstrapp_p
        self.boots_ensemble.add_to_replay_buffer(x, y, self.bootstrapp_p)
        if t<self.batch_size:
            lazy_ber_drop_mlp_pred_error = 0
            ber_drop_mlp_pred_error = 0
            boots_ensemble_pred_error = 0
            lazy_ber_drop_mlp_postSample_unc = 0
            ber_drop_mlp_postSample_unc = 0
            boots_ensemble_preds_unc = 0
            lazy_ber_drop_mlp_loss = 0
            ber_drop_mlp_loss = 0
            boots_ensemble_loss = 0
        else:
            ###########################################################################################
            # Post Sample and estimate uncertainty
            x_postSampling = np.matlib.repmat(x, self.post_sample_size, 1)  # repmat x for post sampling

            #   LazyBernoulliDropoutMLP
            lazy_ber_drop_mlp_postSample = sess.run(self.lazy_ber_drop_mlp_y, feed_dict={self.x_ph: x_postSampling})

            lazy_ber_drop_mlp_pred = np.mean(lazy_ber_drop_mlp_postSample, axis=0)
            lazy_ber_drop_mlp_pred_error = np.linalg.norm((y - lazy_ber_drop_mlp_pred), ord=2)

            lazy_ber_drop_mlp_postSample_cov = np.cov(lazy_ber_drop_mlp_postSample, rowvar=False)
            lazy_ber_drop_mlp_postSample_unc = np.sum(np.diag(lazy_ber_drop_mlp_postSample_cov))

            #   BernoulliDropoutMLP
            ber_drop_mlp_postSample = sess.run(self.ber_drop_mlp_y, feed_dict={self.x_ph: x_postSampling})

            ber_drop_mlp_pred = np.mean(ber_drop_mlp_postSample, axis=0)
            ber_drop_mlp_pred_error = np.linalg.norm((y - ber_drop_mlp_pred), ord=2)

            ber_drop_mlp_postSample_cov = np.cov(ber_drop_mlp_postSample, rowvar=False)
            ber_drop_mlp_postSample_unc = np.sum(np.diag(ber_drop_mlp_postSample_cov))

            #   BootstrappedEnsemble
            boots_ensemble_preds = self.boots_ensemble.prediction(sess, x)

            boots_ensemble_preds_pred = np.mean(boots_ensemble_preds, axis=0)
            boots_ensemble_pred_error = np.linalg.norm((y - boots_ensemble_preds_pred), ord=2)

            boots_ensemble_preds_cov = np.cov(boots_ensemble_preds, rowvar=False)
            boots_ensemble_preds_unc = np.sum(np.diag(boots_ensemble_preds_cov))

            ########################################################################################
            # train
            lazy_ber_drop_mlp_loss, ber_drop_mlp_loss, boots_ensemble_loss = self._train(sess)

        ########################################################################################
        # log data
        self.uncertainty_on_random_net_logger.log_tabular('Step', t)
        self.uncertainty_on_random_net_logger.log_tabular('LBDPredError', lazy_ber_drop_mlp_pred_error)
        self.uncertainty_on_random_net_logger.log_tabular('BDPredError', ber_drop_mlp_pred_error)
        self.uncertainty_on_random_net_logger.log_tabular('BEPredError', boots_ensemble_pred_error)
        self.uncertainty_on_random_net_logger.log_tabular('LBDUnc', lazy_ber_drop_mlp_postSample_unc)
        self.uncertainty_on_random_net_logger.log_tabular('BDUnc', ber_drop_mlp_postSample_unc)
        self.uncertainty_on_random_net_logger.log_tabular('BEUnc', boots_ensemble_preds_unc)
        self.uncertainty_on_random_net_logger.log_tabular('LBDLoss', lazy_ber_drop_mlp_loss)
        self.uncertainty_on_random_net_logger.log_tabular('BDLoss', ber_drop_mlp_loss)
        self.uncertainty_on_random_net_logger.log_tabular('BELoss', boots_ensemble_loss)
        self.uncertainty_on_random_net_logger.log_tabular('Time', time.time() - start_time)
        self.uncertainty_on_random_net_logger.dump_tabular(print_data=False)
        return [lazy_ber_drop_mlp_pred_error, ber_drop_mlp_pred_error, boots_ensemble_pred_error,
                lazy_ber_drop_mlp_postSample_unc, ber_drop_mlp_postSample_unc, boots_ensemble_preds_unc,
                lazy_ber_drop_mlp_loss, ber_drop_mlp_loss, boots_ensemble_loss]

    def _train(self, sess):
        # Train MLP
        mlp_batch = self.mlp_replay_buffer.sample_batch(self.batch_size)
        mlp_outs = sess.run([self.mlp_loss, self.mlp_train_op], feed_dict={self.x_ph: mlp_batch['x'],
                                                                 self.y_ph: mlp_batch['y']})
        sess.run(self.lazy_ber_drop_mlp_update)
        # Train BernoulliDropoutMLP on the same batch with MLP
        ber_drop_outs = sess.run([self.ber_drop_mlp_loss, self.ber_drop_mlp_train_op],
                                 feed_dict={self.x_ph: mlp_batch['x'], self.y_ph: mlp_batch['y']})
        # Train BootstrappedEnsemble
        boots_ensemble_loss = self.boots_ensemble.train(sess, self.batch_size)
        return mlp_outs[0], ber_drop_outs[0], boots_ensemble_loss.mean()


