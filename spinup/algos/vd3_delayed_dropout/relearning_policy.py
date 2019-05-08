"""
Relearning policy with Supervised Learning.

Definition of Relearning:
https://www.britannica.com/science/memory-psychology/Retrieval#ref986140
"""
import pandas as pd
import json

# Load configuration
config_file_path = r"C:\Users\pc-admin\Github Repos\spinningup\data\2019-04-03_ud3_test_experience_saving\2019-04-03_22-43-40-ud3_test_experience_saving_s3\config.json"
with open(config_file_path) as config_file:
    config = json.load(config_file)

# Load experiences
experience_file_path = r"C:\Users\pc-admin\Github Repos\spinningup\data\2019-04-03_ud3_test_experience_saving\2019-04-03_22-43-40-ud3_test_experience_saving_s3\experiences.txt"
experience_df = pd.read_csv(experience_file_path, sep=' ')


# # Create policy
# # Inputs to computation graph
# x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)
#
# # Main outputs from computation graph
# with tf.variable_scope('main'):
#     pi, pi_reg, pi_dropout_mask_generator, pi_dropout_mask_phs,\
#     q, q_reg, q_dropout_mask_generator, q_dropout_mask_phs,\
#     q_pi, q_pi_reg = actor_critic(x_ph, a_ph, **ac_kwargs, dropout_rate=dropout_rate)
#
#     # Initialize uncertainty module
#     obs_set_size = 10
#     track_obs_set_unc_frequency = 100 # every 100 steps
#     pi_unc_module = UncertaintyModule(act_dim, n_post_action,
#                                       obs_set_size, track_obs_set_unc_frequency,
#                                       pi, x_ph, pi_dropout_mask_phs, pi_dropout_mask_generator,
#                                       logger_kwargs)
#
# # Target policy network
# with tf.variable_scope('target'):
#     pi_targ, _, pi_targ_dropout_mask_generator, pi_dropout_mask_phs_targ, \
#     _, _, _, _, _, _ = actor_critic(x2_ph, a_ph, **ac_kwargs, dropout_rate=dropout_rate)
#
# # Target Q networks
# with tf.variable_scope('target', reuse=True):
#     if target_policy_smooth == True:
#         # Target policy smoothing, by adding clipped noise to target actions
#         epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
#         epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
#         a2 = pi_targ + epsilon
#         a2 = tf.clip_by_value(a2, -act_limit, act_limit)
#     else:
#         a2 = pi_targ
#
#     # Target Q-values, using action from target policy
#     _, _, _, _, \
#     q_targ, _, q_targ_dropout_mask_generator, q_dropout_mask_phs_targ, q_pi_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs, dropout_rate=0)

