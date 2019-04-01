from spinup.utils.logx import Logger
import numpy as np

class UncertaintyModule(object):
    """This class is to provide functions to investigate uncertainty change trajectories."""
    def __init__(self, act_dim, n_post_action,
                 obs_set_size, track_obs_set_unc_frequency,
                 pi, x_ph,
                 pi_dropout_mask_phs, pi_dropout_mask_generator,
                 logger_kwargs):
        self.act_dim = act_dim
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

        self.logger = Logger(output_fname='uncertainty.txt', **logger_kwargs)

    def sample_obs_set_from_replay_buffer(self, replay_buffer):
        """Sample an obs set from replay buffer."""
        self.obs_set = replay_buffer.sample_batch(self.obs_set_size)['obs1']
        self.obs_set_is_empty = False

    def calculate_obs_set_uncertainty(self, sess,epoch, step):
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('Step', step)
        for obs_i, obs in enumerate(self.obs_set):
            # import pdb
            # pdb.set_trace()
            a_post = self.get_post_samples(obs, sess)
            a_cov = np.cov(a_post, rowvar=False)
            a_unc_cov = np.sum(np.triu(a_cov))  # summation of upper triangle of an array as uncertainty
            a_unc_var = np.sum(np.diag(a_cov))  # summation of upper triangle of an array as uncertainty
            self.logger.log_tabular('Obs{}_cov'.format(obs_i), a_unc_cov)
            self.logger.log_tabular('Obs{}_var'.format(obs_i), a_unc_var)
        self.logger.dump_tabular()

    def get_post_samples(self, obs, sess):
        """Return a post sample matrix for an observation."""
        feed_dictionary = {self.x_ph: obs.reshape(1, -1)}
        a_post = np.zeros((self.n_post_action, self.act_dim))
        for post_i in range(self.n_post_action):
            # import pdb; pdb.set_trace()
            pi_dropout_masks = self.pi_dropout_mask_generator.generate_dropout_mask()
            for mask_i in range(len(self.pi_dropout_mask_phs)):
                feed_dictionary[self.pi_dropout_mask_phs[mask_i]] = pi_dropout_masks[mask_i]
            a_post[post_i] = sess.run(self.pi, feed_dict=feed_dictionary)[0]
        return a_post
