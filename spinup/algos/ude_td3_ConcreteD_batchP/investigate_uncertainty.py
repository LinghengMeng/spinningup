import tensorflow as tf
from spinup.algos.ude_td3_batchP.core import get_vars, mlp_actor_critic, random_net_distill
from spinup.utils.logx import Logger
import numpy as np

# class UncertaintyModule(object):
#     """This class is to provide functions to investigate dropout-based uncertainty change trajectories."""
#
#     def __init__(self, act_dim, obs_dim, n_post_action,
#                  obs_set_size, track_obs_set_unc_frequency,
#                  pi, x_ph, a_ph,
#                  pi_dropout_mask_phs, pi_dropout_mask_generator,
#                  rnd_targ_act, rnd_pred_act,
#                  rnd_targ_cri, rnd_pred_cri,
#                  logger_kwargs,
#                  tf_var_scope_main='main', tf_var_scope_target='target', tf_var_scope_unc='uncertainty',
#                  uncertainty_type='dropout'):
#         self.act_dim = act_dim
#         self.obs_dim = obs_dim
#         self.n_post_action = n_post_action
#         # policy
#         self.pi = pi
#         self.x_ph = x_ph
#         self.a_ph = a_ph
#         # dropout
#         self.pi_dropout_mask_phs = pi_dropout_mask_phs
#         self.pi_dropout_mask_generator = pi_dropout_mask_generator
#         # rnd
#         self.rnd_targ_act = rnd_targ_act
#         self.rnd_pred_act = rnd_pred_act
#         self.rnd_targ_cri = rnd_targ_cri
#         self.rnd_pred_cri = rnd_pred_cri
#
#         self.obs_set_size = obs_set_size
#         self.obs_set_is_empty = True
#         self.track_obs_set_unc_frequency = track_obs_set_unc_frequency
#
#         self.tf_var_scope_main = tf_var_scope_main
#         self.tf_var_scope_target = tf_var_scope_target
#         self.tf_var_scope_unc = tf_var_scope_unc
#
#         self.uncertainty_logger = Logger(output_fname='{}_uncertainty.txt'.format(uncertainty_type),
#                                          **logger_kwargs)
#         self.sample_logger = Logger(output_fname='{}_sample_observation.txt'.format(uncertainty_type),
#                                     **logger_kwargs)
#     # TODO: target policy
#     def uncertainty_policy_update_targ(self,sess):
#         """Update uncertainty policy to current policy"""
#         sess.run(tf.group([tf.assign(v_unc, v_main)
#                            for v_main, v_unc in
#                            zip(get_vars(self.tf_var_scope_target), get_vars(self.tf_var_scope_unc))]))
#
#     def uncertainty_policy_update(self, sess):
#         """Update uncertainty policy to current policy"""
#         sess.run(tf.group([tf.assign(v_unc, v_main)
#                   for v_main, v_unc in zip(get_vars(self.tf_var_scope_main), get_vars(self.tf_var_scope_unc))]))
#
#     def sample_obs_set_from_replay_buffer(self, replay_buffer):
#         """Sample an obs set from replay buffer."""
#         self.obs_set = replay_buffer.sample_batch(self.obs_set_size)['obs1']
#         self.obs_set_is_empty = False
#         # Save sampled observations
#         for i, o in enumerate(self.obs_set):
#             self.sample_logger.log_tabular('Observation', i)
#             # import pdb; pdb.set_trace()
#             for dim, o_i in enumerate(o):
#                 self.sample_logger.log_tabular('o_{}'.format(dim), o_i)
#             self.sample_logger.dump_tabular(print_data=False)
#
#     def calculate_obs_set_uncertainty(self, sess, epoch, step):
#         self.uncertainty_logger.log_tabular('Epoch', epoch)
#         self.uncertainty_logger.log_tabular('Step', step)
#         for obs_i, obs in enumerate(self.obs_set):
#             # Calculate uncertainty
#             a_post = self.get_post_samples(obs, sess, step)
#             a_cov = np.cov(a_post, rowvar=False)
#             for unc_i, unc_v in enumerate(np.array(a_cov).flatten(order='C')):
#                 self.uncertainty_logger.log_tabular('Obs{}_unc_{}'.format(obs_i, unc_i), unc_v)
#             # Calculate RND prediction error
#             rnd_targ, rnd_pred, rnd_pred_error = self.calculate_actor_RND_pred_error(obs, sess)
#             for rnd_i in range(self.act_dim):
#                 self.uncertainty_logger.log_tabular('Obs{}_rnd_t_{}'.format(obs_i, rnd_i), rnd_targ[rnd_i])
#                 self.uncertainty_logger.log_tabular('Obs{}_rnd_p_{}'.format(obs_i, rnd_i), rnd_pred[rnd_i])
#             self.uncertainty_logger.log_tabular('Obs{}_rnd_error'.format(obs_i), rnd_pred_error)
#         self.uncertainty_logger.dump_tabular(print_data=False)
#
#     def calculate_actor_RND_pred_error(self, obs, sess):
#         feed_dictionary = {self.x_ph: obs.reshape(1, -1)}
#         targ, pred = sess.run([self.rnd_targ_act, self.rnd_pred_act], feed_dict=feed_dictionary)
#         pred_error = np.sqrt(np.sum((pred-targ)**2))
#         return targ[0], pred[0], pred_error
#
#     def calculate_critic_RND_pred_error(self, obs, act, sess):
#         feed_dictionary = {self.x_ph: obs.reshape(1, -1), self.a_ph: act.reshape(1, -1)}
#         targ, pred = sess.run([self.rnd_targ_cri, self.rnd_pred_cri], feed_dict=feed_dictionary)
#         pred_error = np.sqrt(np.sum(pred-targ)**2)
#         return targ[0], pred[0], pred_error
#
#     def get_post_samples(self, obs, sess):
#         """Return a post sample matrix for an observation."""
#         return np.zeros(self.n_post_action, self.act_dim)

class DropoutUncertaintyModule:
    """This class is to provide functions to investigate dropout-based uncertainty change trajectories."""
    def __init__(self, act_dim, obs_dim, n_post_action,
                 obs_set_size, track_obs_set_unc_frequency,
                 x_ph, a_ph, ac_kwargs, dropout_rate,
                 logger_kwargs,
                 tf_var_scope_main='main', tf_var_scope_target='target',
                 tf_var_scope_rnd='random_net_distill'):

        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.n_post_action = n_post_action

        self.obs_set_size = obs_set_size
        self.obs_set_is_empty = True
        self.track_obs_set_unc_frequency = track_obs_set_unc_frequency

        self.tf_var_scope_main = tf_var_scope_main
        self.tf_var_scope_target = tf_var_scope_target
        self.tf_var_scope_rnd = tf_var_scope_rnd
        self.tf_var_scope_main_unc = 'main_uncertainty'
        self.tf_var_scope_target_unc = 'target_uncertainty'
        self.tf_var_scope_rnd_unc = 'rnd_uncertainty'

        # Create Actor-critic and RND to load weights for post sampling
        with tf.variable_scope(self.tf_var_scope_main_unc):
            self.x_ph = x_ph
            self.a_ph = a_ph
            # Actor-critic
            self.pi, _, self.pi_dropout_mask_generator, self.pi_dropout_mask_phs, \
            self.q1, _, self.q1_dropout_mask_generator, self.q1_dropout_mask_phs, self.q1_pi, _, \
            self.q2, _, self.q2_dropout_mask_generator, self.q2_dropout_mask_phs = mlp_actor_critic(x_ph, a_ph, **ac_kwargs,
                                                                                                    dropout_rate=dropout_rate)
        with tf.variable_scope(self.tf_var_scope_rnd_unc):
            # import pdb; pdb.set_trace()
            # Random Network Distillation
            self.rnd_targ_act, \
            self.rnd_pred_act, _, \
            self.rnd_pred_act_dropout_mask_generator, self.rnd_pred_act_dropout_mask_phs, \
            self.rnd_targ_cri, \
            self.rnd_pred_cri, _, \
            self.rnd_pred_cri_dropout_mask_generator, self.rnd_pred_cri_dropout_mask_phs = random_net_distill(x_ph, a_ph,
                                                                                                              **ac_kwargs,
                                                                                                              dropout_rate=dropout_rate)
        self.dropout_masks_set_pi = self.pi_dropout_mask_generator.generate_dropout_mask(n_post_action)
        self.dropout_masks_set_q1 = self.q1_dropout_mask_generator.generate_dropout_mask(n_post_action)
        self.dropout_masks_set_q2 = self.q2_dropout_mask_generator.generate_dropout_mask(n_post_action)
        self.dropout_masks_set_rnd_act = self.rnd_pred_act_dropout_mask_generator.generate_dropout_mask(n_post_action)
        self.dropout_masks_set_rnd_cri = self.rnd_pred_cri_dropout_mask_generator.generate_dropout_mask(n_post_action)



        self.uncertainty_logger = Logger(output_fname='dropout_uncertainty.txt',
                                         **logger_kwargs)
        self.sample_logger = Logger(output_fname='dropout_sample_observation.txt',
                                    **logger_kwargs)

        self.delayed_dropout_masks_update = False
        self.delayed_dropout_masks_update_freq = 1000

    # Load weights
    def update_weights_of_target_unc(self, sess):
        """Update uncertainty policy to current policy"""
        sess.run(tf.group([tf.assign(v_unc, v_targ)
                           for v_targ, v_unc in
                           zip(get_vars(self.tf_var_scope_target), get_vars(self.tf_var_scope_target_unc))]))

    def update_weights_of_main_unc(self, sess):
        """Update uncertainty policy to current policy"""
        sess.run(tf.group([tf.assign(v_unc, v_main)
                           for v_main, v_unc in
                           zip(get_vars(self.tf_var_scope_main), get_vars(self.tf_var_scope_main_unc))]))

    def update_weights_of_rnd_unc(self, sess):
        """Update uncertainty policy to current policy"""
        sess.run(tf.group([tf.assign(v_unc, v_rnd)
                           for v_rnd, v_unc in
                           zip(get_vars(self.tf_var_scope_rnd), get_vars(self.tf_var_scope_rnd_unc))]))

    # Update dropout masks
    def uncertainty_pi_dropout_masks_update(self):
        """Update uncertainty dropout_masks."""
        self.dropout_masks_set_pi = self.pi_dropout_mask_generator.generate_dropout_mask(self.n_post_action)

    def uncertainty_q_dropout_masks_update(self):
        """Update uncertainty dropout_masks."""
        self.dropout_masks_set_q1 = self.q1_dropout_mask_generator.generate_dropout_mask(self.n_post_action)
        self.dropout_masks_set_q2 = self.q2_dropout_mask_generator.generate_dropout_mask(self.n_post_action)

    def uncertainty_rnd_dropout_masks_update(self):
        """Update uncertainty dropout_masks."""
        self.dropout_masks_set_rnd_act = self.rnd_pred_act_dropout_mask_generator.generate_dropout_mask(self.n_post_action)
        self.dropout_masks_set_rnd_cri = self.rnd_pred_cri_dropout_mask_generator.generate_dropout_mask(self.n_post_action)

    # Get post samples
    def get_post_samples_act(self, obs, sess, step_index):
        """Return a post sample of actions for an observation."""
        feed_dictionary = {self.x_ph: obs.reshape(1, -1)}
        a_post = np.zeros((self.n_post_action, self.act_dim))
        if not self.delayed_dropout_masks_update:
            self.uncertainty_pi_dropout_masks_update()
        elif step_index % self.delayed_dropout_masks_update_freq:
            self.uncertainty_pi_dropout_masks_update()
        else:
            pass

        for mask_i in range(len(self.pi_dropout_mask_phs)):
            feed_dictionary[self.pi_dropout_mask_phs[mask_i]] = self.dropout_masks_set_pi[mask_i]
        # import pdb; pdb.set_trace()
        a_post = sess.run(self.pi, feed_dict=feed_dictionary)[:,0,:]
        return a_post
    
    def get_post_samples_q(self, obs, act, sess, step_index):
        """Return a post sample for a (observation, action) pair."""
        feed_dictionary = {self.x_ph: obs.reshape(1, -1), self.a_ph: act.reshape(1, -1)}
        q1_post = np.zeros((self.n_post_action, ))
        q2_post = np.zeros((self.n_post_action, ))
        if not self.delayed_dropout_masks_update:
            self.uncertainty_q_dropout_masks_update()
        elif step_index % self.delayed_dropout_masks_update_freq:
            self.uncertainty_q_dropout_masks_update()
        else:
            pass

        for mask_i in range(len(self.q1_dropout_mask_phs)):
            feed_dictionary[self.q1_dropout_mask_phs[mask_i]] = self.dropout_masks_set_q1[mask_i]
            feed_dictionary[self.q2_dropout_mask_phs[mask_i]] = self.dropout_masks_set_q2[mask_i]
        # import pdb; pdb.set_trace()
        q1_post = sess.run(self.q1, feed_dict=feed_dictionary)[:,0]
        q2_post = sess.run(self.q2, feed_dict=feed_dictionary)[:,0]
        return q1_post, q2_post

    def get_post_samples_rnd_act(self, obs, sess, step_index):
        """Return a post sample of action predictions for an observation."""
        feed_dictionary = {self.x_ph: obs.reshape(1, -1)}
        rnd_a_post = np.zeros((self.n_post_action, self.act_dim))
        if not self.delayed_dropout_masks_update:
            self.uncertainty_pi_dropout_masks_update()
        elif step_index % self.delayed_dropout_masks_update_freq:
            self.uncertainty_rnd_dropout_masks_update()
        else:
            pass

        for mask_i in range(len(self.rnd_pred_act_dropout_mask_phs)):
            feed_dictionary[self.rnd_pred_act_dropout_mask_phs[mask_i]] = self.dropout_masks_set_rnd_act[mask_i]
        rnd_a_post = sess.run(self.rnd_pred_act, feed_dict=feed_dictionary)[:, 0, :]
        return rnd_a_post

    def get_post_samples_rnd_cri(self, obs, act, sess, step_index):
        """Return a post sample of q predictions for a (observation, action) pair."""
        feed_dictionary = {self.x_ph: obs.reshape(1, -1), self.a_ph: act.reshape(1, -1)}
        rnd_q_post = np.zeros((self.n_post_action,))
        if not self.delayed_dropout_masks_update:
            self.uncertainty_rnd_dropout_masks_update()
        elif step_index % self.delayed_dropout_masks_update_freq:
            self.uncertainty_rnd_dropout_masks_update()
        else:
            pass

        for mask_i in range(len(self.rnd_pred_cri_dropout_mask_phs)):
            feed_dictionary[self.rnd_pred_cri_dropout_mask_phs[mask_i]] = self.dropout_masks_set_rnd_cri[mask_i]
        # import pdb; pdb.set_trace()
        rnd_q_post = sess.run(self.rnd_pred_cri, feed_dict=feed_dictionary)[:, 0]
        return rnd_q_post

    # Sample observations to track
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

    # Calculate uncertainty of tracked observations
    def calculate_obs_set_uncertainty(self, sess, epoch, step):
        self.uncertainty_logger.log_tabular('Epoch', epoch)
        self.uncertainty_logger.log_tabular('Step', step)
        for obs_i, obs in enumerate(self.obs_set):
            # Calculate uncertainty
            a_post = self.get_post_samples_act(obs, sess, step)
            a_cov = np.cov(a_post, rowvar=False)
            for unc_i, unc_v in enumerate(np.array(a_cov).flatten(order='C')):
                self.uncertainty_logger.log_tabular('Obs{}_unc_{}'.format(obs_i, unc_i), unc_v)
            # Calculate RND prediction error
            rnd_targ, rnd_pred, rnd_pred_error = self.calculate_actor_RND_pred_error(obs, sess)
            # import pdb; pdb.set_trace()
            for rnd_i in range(self.act_dim):
                self.uncertainty_logger.log_tabular('Obs{}_rnd_t_{}'.format(obs_i, rnd_i), rnd_targ[rnd_i])
                self.uncertainty_logger.log_tabular('Obs{}_rnd_p_{}'.format(obs_i, rnd_i), rnd_pred[rnd_i])
            self.uncertainty_logger.log_tabular('Obs{}_rnd_error'.format(obs_i), rnd_pred_error)
        self.uncertainty_logger.dump_tabular(print_data=False)

    # Calculate RND prediction errors
    def calculate_actor_RND_pred_error(self, obs, sess):
        """Calculate prediction error without dropout"""
        feed_dictionary = {self.x_ph: obs.reshape(1, -1)}
        for mask_i in range(len(self.rnd_pred_act_dropout_mask_phs)):
            feed_dictionary[self.rnd_pred_act_dropout_mask_phs[mask_i]] = np.ones([1, self.rnd_pred_act_dropout_mask_phs[mask_i].shape.as_list()[1]])
        targ, pred = sess.run([self.rnd_targ_act, self.rnd_pred_act], feed_dict=feed_dictionary)
        pred_error = np.sqrt(np.sum((pred[0]-targ)**2))
        return targ[0], pred[0][0], pred_error

    def calculate_critic_RND_pred_error(self, obs, act, sess):
        """Calculate prediction error without dropout"""
        feed_dictionary = {self.x_ph: obs.reshape(1, -1), self.a_ph: act.reshape(1, -1)}
        for mask_i in range(len(self.rnd_pred_cri_dropout_mask_phs)):
            feed_dictionary[self.rnd_pred_cri_dropout_mask_phs[mask_i]] = np.ones([1, self.rnd_pred_cri_dropout_mask_phs[mask_i].shape.as_list()[1]])
        targ, pred = sess.run([self.rnd_targ_cri, self.rnd_pred_cri], feed_dict=feed_dictionary)
        pred_error = np.sqrt(np.sum(pred[0]-targ)**2)
        return targ[0], pred[0], pred_error

# class ObsSampleUncertaintyModule(UncertaintyModule):
#     """Class for investigating state-sampling-based uncertainty estimation."""
#     def __init__(self, act_dim, obs_dim, n_post_action,
#                  obs_set_size, track_obs_set_unc_frequency,
#                  pi, x_ph,
#                  pi_dropout_mask_phs, pi_dropout_mask_generator,
#                  logger_kwargs, sample_obs_std=1):
#         super().__init__(act_dim, obs_dim, n_post_action,
#                          obs_set_size, track_obs_set_unc_frequency,
#                          pi, x_ph,
#                          pi_dropout_mask_phs, pi_dropout_mask_generator,
#                          logger_kwargs, 'obs_sample')
#         self.sample_obs_std = sample_obs_std
#
#     def get_post_samples(self, obs, sess):
#         """Return a post sample matrix for an observation."""
#         feed_dictionary = {}
#         for mask_i in range(len(self.pi_dropout_mask_phs)):
#             feed_dictionary[self.pi_dropout_mask_phs[mask_i]] = np.ones(self.pi_dropout_mask_phs[mask_i].shape.as_list())
#
#         # Sample states
#         if self.act_dim == 1:
#             feed_dictionary[self.x_ph] = np.random.normal(obs,
#                                                           self.sample_obs_std,
#                                                           size=(self.n_post_action,self.obs_dim))
#         else:
#             feed_dictionary[self.x_ph] = np.random.multivariate_normal(obs,
#                                                                        self.sample_obs_std*np.identity(self.obs_dim),
#                                                                        self.n_post_action)
#             # import pdb; pdb.set_trace()
#         a_post = sess.run(self.pi, feed_dict=feed_dictionary)
#         return a_post


