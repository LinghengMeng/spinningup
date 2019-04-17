import tensorflow as tf
from spinup.algos.ude_td3.core import get_vars
from spinup.utils.logx import Logger
import numpy as np

class UncertaintyModule(object):
    """This class is to provide functions to investigate dropout-based uncertainty change trajectories."""

    def __init__(self, act_dim, obs_dim, n_post_action,
                 obs_set_size, track_obs_set_unc_frequency,
                 pi, x_ph,
                 pi_dropout_mask_phs, pi_dropout_mask_generator,
                 logger_kwargs,
                 tf_var_scope_main='main', tf_var_scope_unc='uncertainty',
                 uncertainty_type='dropout'):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.n_post_action = n_post_action
        # policy
        self.pi = pi
        self.x_ph = x_ph
        # dropout
        self.pi_dropout_mask_phs = pi_dropout_mask_phs
        self.pi_dropout_mask_generator = pi_dropout_mask_generator

        self.obs_set_size = obs_set_size
        self.obs_set_is_empty = True
        self.track_obs_set_unc_frequency = track_obs_set_unc_frequency

        self.tf_var_scope_main = tf_var_scope_main
        self.tf_var_scope_unc = tf_var_scope_unc

        self.uncertainty_logger = Logger(output_fname='{}_uncertainty.txt'.format(uncertainty_type),
                                         **logger_kwargs)
        self.sample_logger = Logger(output_fname='{}_sample_observation.txt'.format(uncertainty_type),
                                    **logger_kwargs)

    def uncertainty_policy_update(self, sess):
        """Update uncertainty policy to current policy"""
        sess.run(tf.group([tf.assign(v_unc, v_main)
                  for v_main, v_unc in zip(get_vars(self.tf_var_scope_main), get_vars(self.tf_var_scope_unc))]))

    def sample_obs_set_from_replay_buffer(self, replay_buffer):
        """Sample an obs set from replay buffer."""
        self.obs_set = replay_buffer.sample_batch(self.obs_set_size)['obs1']
        self.obs_set_is_empty = False
        # Save sampled observations
        for i, o in enumerate(self.obs_set):
            self.sample_logger.log_tabular('Observation', i)
            # import pdb; pdb.set_trace()
            for dim, o_i in enumerate(o):
                self.sample_logger.log_tabular('o_{}'.format(dim), o_i)
            self.sample_logger.dump_tabular(print_data=False)

    def calculate_obs_set_uncertainty(self, sess, epoch, step):
        self.uncertainty_logger.log_tabular('Epoch', epoch)
        self.uncertainty_logger.log_tabular('Step', step)
        for obs_i, obs in enumerate(self.obs_set):
            # import pdb
            # pdb.set_trace()
            a_post = self.get_post_samples(obs, sess)
            a_cov = np.cov(a_post, rowvar=False)
            if a_post.shape[1] != 1:
                a_unc_cov = np.sum(np.triu(a_cov))  # summation of upper triangle of an array as uncertainty
                a_unc_var = np.sum(np.diag(a_cov))
            else:
                a_unc_cov = np.sum(a_cov)
                a_unc_var = np.sum(a_cov)  # summation of upper triangle of an array as uncertainty
            self.uncertainty_logger.log_tabular('Obs{}_cov'.format(obs_i), a_unc_cov)
            self.uncertainty_logger.log_tabular('Obs{}_var'.format(obs_i), a_unc_var)
        self.uncertainty_logger.dump_tabular(print_data=False)

    def get_post_samples(self, obs, sess):
        """Return a post sample matrix for an observation."""
        return np.zeros(self.n_post_action, self.act_dim)

class DropoutUncertaintyModule(UncertaintyModule):
    """This class is to provide functions to investigate dropout-based uncertainty change trajectories."""
    def __init__(self, act_dim, obs_dim, n_post_action,
                 obs_set_size, track_obs_set_unc_frequency,
                 pi, x_ph,
                 pi_dropout_mask_phs, pi_dropout_mask_generator,
                 logger_kwargs,
                 tf_var_scope_main='main', tf_var_scope_unc='uncertainty'):
        super().__init__(act_dim, obs_dim, n_post_action,
                         obs_set_size, track_obs_set_unc_frequency,
                         pi, x_ph,
                         pi_dropout_mask_phs, pi_dropout_mask_generator,
                         logger_kwargs, tf_var_scope_main, tf_var_scope_unc, 'dropout')
        self.dropout_masks_set = {i:pi_dropout_mask_generator.generate_dropout_mask() for i in range(n_post_action)}

    def uncertainty_dropout_masks_update(self):
        """Update uncertainty dropout_masks."""
        self.dropout_masks_set = {i: self.pi_dropout_mask_generator.generate_dropout_mask() for i in
                                  range(self.n_post_action)}

    def get_post_samples(self, obs, sess):
        """Return a post sample matrix for an observation."""
        feed_dictionary = {self.x_ph: obs.reshape(1, -1)}
        a_post = np.zeros((self.n_post_action, self.act_dim))
        for post_i in range(self.n_post_action):
            # import pdb; pdb.set_trace()
            # pi_dropout_masks = self.pi_dropout_mask_generator.generate_dropout_mask()
            for mask_i in range(len(self.pi_dropout_mask_phs)):
                feed_dictionary[self.pi_dropout_mask_phs[mask_i]] = self.dropout_masks_set[post_i][mask_i]
            # import pdb; pdb.set_trace()
            a_post[post_i] = sess.run(self.pi, feed_dict=feed_dictionary)[0]
        return a_post

class ObsSampleUncertaintyModule(UncertaintyModule):
    """Class for investigating state-sampling-based uncertainty estimation."""
    def __init__(self, act_dim, obs_dim, n_post_action,
                 obs_set_size, track_obs_set_unc_frequency,
                 pi, x_ph,
                 pi_dropout_mask_phs, pi_dropout_mask_generator,
                 logger_kwargs, sample_obs_std=1):
        super().__init__(act_dim, obs_dim, n_post_action,
                         obs_set_size, track_obs_set_unc_frequency,
                         pi, x_ph,
                         pi_dropout_mask_phs, pi_dropout_mask_generator,
                         logger_kwargs, 'obs_sample')
        self.sample_obs_std = sample_obs_std

    def get_post_samples(self, obs, sess):
        """Return a post sample matrix for an observation."""
        feed_dictionary = {}
        for mask_i in range(len(self.pi_dropout_mask_phs)):
            feed_dictionary[self.pi_dropout_mask_phs[mask_i]] = np.ones(self.pi_dropout_mask_phs[mask_i].shape.as_list())

        # Sample states
        if self.act_dim == 1:
            feed_dictionary[self.x_ph] = np.random.normal(obs,
                                                          self.sample_obs_std,
                                                          size=(self.n_post_action,self.obs_dim))
        else:
            feed_dictionary[self.x_ph] = np.random.multivariate_normal(obs,
                                                                       self.sample_obs_std*np.identity(self.obs_dim),
                                                                       self.n_post_action)
            # import pdb; pdb.set_trace()
        a_post = sess.run(self.pi, feed_dict=feed_dictionary)
        return a_post


