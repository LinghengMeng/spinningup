import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.vd3_delayed_dropout import core
from spinup.algos.vd3_delayed_dropout.core import get_vars
from spinup.algos.vd3_delayed_dropout.investigate_uncertainty import DropoutUncertaintyModule,ObsSampleUncertaintyModule
from spinup.utils.logx import EpochLogger, Logger
from multiprocessing import Pool

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""

VD3 (Variational Delayed DDPG)

"""
def vd3_delayed_dropout(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
                        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
                        polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
                        without_start_steps=False,
                        without_delay_train=True,
                        batch_size=100, start_steps=10000,
                        act_noise=0.1, target_noise=0.2, noise_clip=0.5,
                        n_post_q=1, n_post_action = 10, uncertainty_method='dropout',
                        sample_action_with_dropout = True, dropout_rate=0.1,
                        action_choose_method = 'random_sample',
                        uncertainty_noise_type = 'std_noise',
                        a_var_clip_max = 1, a_var_clip_min = 0.1,
                        a_std_clip_max = 1, a_std_clip_min = 0.1,
                        concentration_factor = 0.2, minimum_exploration_level=0,
                        policy_delay=2,target_policy_smooth = False,
                        max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target 
            policy.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    # TODO: Test no start steps
    if without_start_steps:
        start_steps = batch_size

    # EpochLogger: save epoch info
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    # ExperienceLogger: save experiences for supervised learning
    experience_logger = Logger(output_fname='experiences.txt', **logger_kwargs)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, pi_reg, pi_dropout_mask_generator, pi_dropout_mask_phs,\
        q, q_reg, q_dropout_mask_generator, q_dropout_mask_phs,\
        q_pi, q_pi_reg = actor_critic(x_ph, a_ph, **ac_kwargs, dropout_rate=dropout_rate)

        # TODO: calculate uncertainty based on target policy
        # Initialize uncertainty module
        obs_set_size = 10
        track_obs_set_unc_frequency = 100 # every 100 steps
        if uncertainty_method == 'dropout':
            pi_unc_module = DropoutUncertaintyModule(act_dim, obs_dim, n_post_action,
                                                     obs_set_size, track_obs_set_unc_frequency,
                                                     pi, x_ph, pi_dropout_mask_phs, pi_dropout_mask_generator,
                                                     logger_kwargs)
        elif uncertainty_method == 'gaussian_obs_sample':
            pi_unc_module = ObsSampleUncertaintyModule(act_dim, obs_dim, n_post_action,
                                                       obs_set_size, track_obs_set_unc_frequency,
                                                       pi, x_ph, pi_dropout_mask_phs, pi_dropout_mask_generator,
                                                       logger_kwargs)
        else:
            raise ValueError('Please choose a proper uncertainty_method!')

    # Target policy network
    with tf.variable_scope('target'):
        pi_targ, _, pi_targ_dropout_mask_generator, pi_dropout_mask_phs_targ, \
        _, _, _, _, _, _ = actor_critic(x2_ph, a_ph, **ac_kwargs, dropout_rate=dropout_rate)

        # # Initialize uncertainty module
        # obs_set_size = 10
        # track_obs_set_unc_frequency = 100  # every 100 steps
        # pi_unc_module = DropoutUncertaintyModule(act_dim, obs_dim, n_post_action,
        #                                          obs_set_size, track_obs_set_unc_frequency,
        #                                          pi_targ, x2_ph,
        #                                          pi_dropout_mask_phs_targ, pi_targ_dropout_mask_generator,
        #                                          logger_kwargs)

    # Target Q networks
    with tf.variable_scope('target', reuse=True):
        if target_policy_smooth == True:
            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
            epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = tf.clip_by_value(a2, -act_limit, act_limit)
        else:
            a2 = pi_targ

        # Target Q-values, using action from target policy
        _, _, _, pi_dropout_mask_phs_targ2, \
        q_targ, _, q_targ_dropout_mask_generator, q_dropout_mask_phs_targ, q_pi_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs, dropout_rate=0)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q functions, using Clipped Double-Q targets

    q_post_ph = tf.placeholder(dtype=tf.float32, shape=(batch_size,n_post_q))

    q_targ_min = tf.reduce_min(q_post_ph, 1)
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_targ_min)

    # q_targ_mean = tf.reduce_mean(q_post_ph, 1)
    # backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * q_targ_mean)

    # TD3 losses
    pi_loss = -tf.reduce_mean(q_pi)
    # q_loss = (tf.reduce_sum((q-backup)**2)+q_reg) / batch_size
    q_loss = tf.reduce_mean((q - backup) ** 2)

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q': q})

    def get_action(o, noise_scale, pi_unc_module):
        a_var_uncertainty = 0
        a_var_uncertainty_clipped = 0
        a_std_uncertainty = 0
        a_std_uncertainty_clipped = 0

        feed_dictionary = {x_ph: o.reshape(1, -1)}
        if sample_action_with_dropout:
            # Generate post samples
            # TODO: Tried to use ray implementing parallel post sampling but no speed up in one machine.
            # Non-parallel post sample
            a_post = pi_unc_module.get_post_samples(o, sess)
            # import pdb; pdb.set_trace()
            # a_post = np.zeros((n_post_action, act_dim))
            # for post_i in range(n_post_action):
            #     # import pdb; pdb.set_trace()
            #     pi_dropout_masks = pi_dropout_mask_generator.generate_dropout_mask()
            #     for mask_i in range(len(pi_dropout_mask_phs)):
            #         feed_dictionary[pi_dropout_mask_phs[mask_i]] = pi_dropout_masks[mask_i]
            #
            #     a_post[post_i] = sess.run(pi, feed_dict=feed_dictionary)[0]

            # TODO: var and std must been scaled or clipped.
            #  Otherwise, a huge variance will always cause action out of act_lim and
            #  then be clipped to -1 or 1.
            #  We also need to set a lower bound to enforce a minimum exploration
            a_mean = np.mean(a_post, axis=0)
            a_median = np.median(a_post, axis=0)

            a_var = np.var(a_post, axis=0)
            a_var_clipped = np.clip(a_var, a_var_clip_min, a_var_clip_max)
            a_var_noise = a_var_clipped * np.random.randn(act_dim)

            a_std = np.std(a_post, axis=0)
            a_std_clipped = np.clip(a_std, a_std_clip_min, a_std_clip_max)
            a_std_noise = a_std_clipped * np.random.randn(act_dim)
            # TODO: Save uncertainty for each dimension of action.
            # Calculate uncertainty
            a_var_uncertainty = np.mean(a_var)
            a_var_uncertainty_clipped = np.mean(a_var_clipped)
            a_std_uncertainty = np.mean(a_std)
            a_std_uncertainty_clipped = np.mean(a_std_clipped)

            # TODO: clip noise within a range. Maybe not necessary.
            if uncertainty_noise_type == 'var_noise':
                noise = a_var_noise
            elif uncertainty_noise_type == 'std_noise':
                noise = a_std_noise
            else:
                raise ValueError('Please choose a proper noise_type.')
            a = np.zeros((act_dim,))
            if action_choose_method == 'random_sample':
                # Method 1: randomly sample one from post sampled actions
                a = a_post[np.random.choice(n_post_action)]
            elif action_choose_method == 'gaussian_sample':
                # Method 2: estimate mean and std, then sample from a Gaussian distribution
                for a_i in range(act_dim):
                    a[i] = np.random.normal(a_mean[a_i], a_std_clipped[a_i], 1)
            elif action_choose_method == 'mean_of_samples':
                a = a_mean
            elif action_choose_method == 'median_of_sample':
                pass
            elif action_choose_method == 'mean_and_variance_based_noise':
                a = a_mean+noise
            elif action_choose_method == 'median_and_variance_based_noise':
                a = a_median+noise
            elif action_choose_method == 'prediction_and_variance_based_noise':
                for mask_i in range(len(pi_dropout_mask_phs)):
                    feed_dictionary[pi_dropout_mask_phs[mask_i]] = np.ones(pi_dropout_mask_phs[mask_i].shape.as_list())
                a_prediction = sess.run(pi, feed_dict=feed_dictionary)[0]
                a = a_prediction + noise
            elif action_choose_method == 'gaussian_sample_with_scaling_and_lower_bound':
                # TODO: store this covariance for investigate
                # a = np.random.multivariate_normal(a_mean, a_cov_shaped, 1)[0]
                # a = np.random.multivariate_normal(a_median, a_cov_shaped, 1)[0]
                for mask_i in range(len(pi_dropout_mask_phs)):
                    feed_dictionary[pi_dropout_mask_phs[mask_i]] = np.ones(pi_dropout_mask_phs[mask_i].shape.as_list())
                a_prediction = sess.run(pi, feed_dict=feed_dictionary)[0]

                if act_dim >1:
                    a_cov = np.cov(a_post, rowvar=False)
                    a_cov_shaped = concentration_factor * a_cov + minimum_exploration_level * np.ones(a_cov.shape)
                    a = np.random.multivariate_normal(a_prediction, a_cov_shaped, 1)[0]
                else:
                    a_var_shaped = concentration_factor * a_var + minimum_exploration_level * np.ones(a_var.shape)
                    a = np.random.normal(a_prediction, a_var_shaped, 1)[0]
            else:
                pass
        else:
            for mask_i in range(len(pi_dropout_mask_phs)):
                feed_dictionary[pi_dropout_mask_phs[mask_i]] = np.ones(pi_dropout_mask_phs[mask_i].shape.as_list())
            a = sess.run(pi, feed_dict=feed_dictionary)[0]
            a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit), \
               a_var_uncertainty, a_var_uncertainty_clipped, a_std_uncertainty, a_std_uncertainty_clipped

    def get_action_test(o):
        """Get deterministic action without noise and dropout."""
        feed_dictionary = {x_ph: o.reshape(1, -1)}
        for mask_i in range(len(pi_dropout_mask_phs)):
            feed_dictionary[pi_dropout_mask_phs[mask_i]] = np.ones(pi_dropout_mask_phs[mask_i].shape.as_list())
        a = sess.run(pi, feed_dict=feed_dictionary)[0]
        return np.clip(a, -act_limit, act_limit)

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action_test(o))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)



    start_time = time.time()
    o, r, d, ep_ret, ep_len, \
    ep_a_var_uncertainty,ep_a_var_uncertainty_clipped, \
    ep_a_std_uncertainty, ep_a_std_uncertainty_clipped = env.reset(), 0, False, 0, 0, 0, 0, 0, 0
    total_steps = steps_per_epoch * epochs

    # No dropout for training phase: set all dropout masks to one
    feed_dict_train = {}
    for mask_i in range(len(pi_dropout_mask_phs)):
        feed_dict_train[pi_dropout_mask_phs[mask_i]] = np.ones(pi_dropout_mask_phs[mask_i].shape.as_list())
    for mask_i in range(len(q_dropout_mask_phs)):
        feed_dict_train[q_dropout_mask_phs[mask_i]] = np.ones(q_dropout_mask_phs[mask_i].shape.as_list())
    for mask_i in range(len(pi_dropout_mask_phs_targ)):
        feed_dict_train[pi_dropout_mask_phs_targ[mask_i]] = np.ones(pi_dropout_mask_phs_targ[mask_i].shape.as_list())
    for mask_i in range(len(q_dropout_mask_phs_targ)):
        feed_dict_train[q_dropout_mask_phs_targ[mask_i]] = np.ones(q_dropout_mask_phs_targ[mask_i].shape.as_list())

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps:
            a, a_var_uncertainty, a_var_uncertainty_clipped, \
            a_std_uncertainty, a_std_uncertainty_clipped = get_action(o, act_noise, pi_unc_module)
        else:
            a = env.action_space.sample()
            # TODO:
            a_var_uncertainty = 0
            a_var_uncertainty_clipped = 0
            a_std_uncertainty = 0
            a_std_uncertainty_clipped = 0

        # Sample an observation set to track their uncertainty trajectories
        if t > start_steps:
            if pi_unc_module.obs_set_is_empty:
                pi_unc_module.sample_obs_set_from_replay_buffer(replay_buffer)

            if t % pi_unc_module.track_obs_set_unc_frequency == 0:
                pi_unc_module.calculate_obs_set_uncertainty(sess, t // steps_per_epoch, t)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        ep_a_var_uncertainty += a_var_uncertainty
        ep_a_var_uncertainty_clipped += a_var_uncertainty_clipped
        ep_a_std_uncertainty += a_std_uncertainty
        ep_a_std_uncertainty_clipped += a_std_uncertainty_clipped

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Save experience
        def log_experiences(o, a, r, o2, d, t, steps_per_epoch, start_time):
            experience_logger.log_tabular('Epoch', t // steps_per_epoch)
            experience_logger.log_tabular('Step', t)
            for i, o_i in enumerate(o):
                experience_logger.log_tabular('o_{}'.format(i), o_i)
            for i, a_i in enumerate(a):
                experience_logger.log_tabular('a_{}'.format(i), a_i)
            experience_logger.log_tabular('r', r)
            for i, o2_i in enumerate(o2):
                experience_logger.log_tabular('o2_{}'.format(i), o2_i)
            experience_logger.log_tabular('d', d)
            experience_logger.log_tabular('Time', time.time() - start_time)
            experience_logger.dump_tabular(print_data=False)

        log_experiences(o, a, r, o2, d, t, steps_per_epoch, start_time)
        #import pdb; pdb.set_trace()

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if without_delay_train:
            batch = replay_buffer.sample_batch(batch_size)
            feed_dict_train[x_ph] = batch['obs1']
            feed_dict_train[x2_ph] = batch['obs2']
            feed_dict_train[a_ph] = batch['acts']
            feed_dict_train[r_ph] = batch['rews']
            feed_dict_train[d_ph] = batch['done']
            # Delayed dropout_mask update
            # dropout_delay = 1
            # if j % dropout_delay == 0:
            #     pi_dropout_masks = sess.run(pi_new_dropout_masks)
            #     q_dropout_masks = sess.run(q_new_dropout_masks)
            #
            #     pi_targ_dropout_masks = sess.run(pi_new_dropout_masks_targ)
            #     q_targ_dropout_masks = sess.run(q_new_dropout_masks_targ)

            # TODO: estimate uncertainty of q-value.
            #   Run q nn multiple times
            q_post = np.zeros((batch_size, n_post_q))
            for i in range(n_post_q):
                q_post[:, i] = sess.run(q_targ, feed_dict_train)
            feed_dict_train[q_post_ph] = q_post

            # import pdb; pdb.set_trace()
            q_step_ops = [q_loss, q, train_q_op]
            outs = sess.run(q_step_ops, feed_dict_train)
            logger.store(LossQ=outs[0], QVals=outs[1])

            # Delayed policy update
            outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict_train)
            logger.store(LossPi=outs[0])

        if d or (ep_len == max_ep_len):
            """
            Perform all TD3 updates at the end of the trajectory
            (in accordance with source code of TD3 published by
            original authors).
            """
            if not without_delay_train:
                for j in range(ep_len):
                    batch = replay_buffer.sample_batch(batch_size)
                    feed_dict_train[x_ph] = batch['obs1']
                    feed_dict_train[x2_ph] = batch['obs2']
                    feed_dict_train[a_ph] = batch['acts']
                    feed_dict_train[r_ph] = batch['rews']
                    feed_dict_train[d_ph] = batch['done']
                    # Delayed dropout_mask update
                    # dropout_delay = 1
                    # if j % dropout_delay == 0:
                    #     pi_dropout_masks = sess.run(pi_new_dropout_masks)
                    #     q_dropout_masks = sess.run(q_new_dropout_masks)
                    #
                    #     pi_targ_dropout_masks = sess.run(pi_new_dropout_masks_targ)
                    #     q_targ_dropout_masks = sess.run(q_new_dropout_masks_targ)

                    # TODO: estimate uncertainty of q-value.
                    #   Run q nn multiple times
                    # TODO： here might be a bug that q_targ should be q_pi_targ for backup value
                    q_post = np.zeros((batch_size, n_post_q))
                    for i in range(n_post_q):
                        q_post[:, i] = sess.run(q_targ, feed_dict_train)
                    feed_dict_train[q_post_ph] = q_post

                    # import pdb; pdb.set_trace()
                    q_step_ops = [q_loss, q, train_q_op]
                    outs = sess.run(q_step_ops, feed_dict_train)
                    logger.store(LossQ=outs[0], QVals=outs[1])

                    if j % policy_delay == 0:
                        # Delayed policy update
                        outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict_train)
                        logger.store(LossPi=outs[0])

            logger.store(EpRet=ep_ret, EpLen=ep_len,
                         EpVarUncertainty=ep_a_var_uncertainty,
                         EpVarUncertaintyClipped=ep_a_var_uncertainty_clipped,
                         EpStdUncertainty=ep_a_std_uncertainty,
                         EpStdUncertaintyClipped=ep_a_std_uncertainty_clipped)
            o, r, d, ep_ret, ep_len, \
            ep_a_var_uncertainty, ep_a_var_uncertainty_clipped, \
            ep_a_std_uncertainty, ep_a_std_uncertainty_clipped = env.reset(), 0, False, 0, 0, 0, 0, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('EpVarUncertainty', with_min_and_max=True)
            logger.log_tabular('EpVarUncertaintyClipped', with_min_and_max=True)
            logger.log_tabular('EpStdUncertainty', with_min_and_max=True)
            logger.log_tabular('EpStdUncertaintyClipped', with_min_and_max=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='vd3')
    parser.add_argument('--without_start_steps', action='store_true')
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--without_delay_train', action='store_true')
    parser.add_argument('--batch_size', type=float, default=100)

    parser.add_argument('--n_post_action', type=int, default=10)
    parser.add_argument('--uncertainty_method', choices=['dropout','gaussian_obs_sample'],
                        default='gaussian_obs_sample')
    parser.add_argument('--sample_action_with_dropout', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--action_choose_method', choices=['random_sample',
                                                           'gaussian_sample',
                                                           'mean_and_variance_based_noise',
                                                           'median_and_variance_based_noise',
                                                           'prediction_and_variance_based_noise',
                                                           'gaussian_sample_with_scaling_and_lower_bound'],
                        default='gaussian_sample_with_scaling_and_lower_bound')
    parser.add_argument('--uncertainty_noise_type', type=str, choices=['var_noise', 'std_noise'],
                        default='var_noise')
    parser.add_argument('--a_var_clip_max', type=float, default=1.0)
    parser.add_argument('--a_var_clip_min', type=float, default=0.1)
    parser.add_argument('--a_std_clip_max', type=float, default=1.0)
    parser.add_argument('--a_std_clip_min', type=float, default=0.1)
    parser.add_argument('--concentration_factor', type=float, default=0.1)
    parser.add_argument('--minimum_exploration_level', type=float, default=0)

    parser.add_argument('--target_policy_smooth', action='store_true')
    parser.add_argument('--act_noise', type=float, default=0.1)
    # parser.add_argument('--hardcopy_targ', type=str, default='True')
    args = parser.parse_args()

    # if args.hardcopy_targ == 'True':
    #     polyak = 0

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, datestamp=True)
    
    vd3_delayed_dropout(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
                        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                        gamma=args.gamma,
                        without_start_steps=args.without_start_steps,
                        start_steps=args.start_steps,
                        without_delay_train=args.without_delay_train,
                        batch_size=args.batch_size, seed=args.seed, epochs=args.epochs,
                        n_post_action = args.n_post_action,
                        uncertainty_method = args.uncertainty_method,
                        sample_action_with_dropout=args.sample_action_with_dropout,
                        dropout_rate=args.dropout_rate,
                        action_choose_method=args.action_choose_method,
                        uncertainty_noise_type=args.uncertainty_noise_type,
                        a_var_clip_max=args.a_var_clip_max, a_var_clip_min=args.a_var_clip_min,
                        a_std_clip_max=args.a_std_clip_max, a_std_clip_min=args.a_std_clip_min,
                        concentration_factor=args.concentration_factor,
                        minimum_exploration_level=args.minimum_exploration_level,
                        target_policy_smooth=args.target_policy_smooth,
                        act_noise = args.act_noise,
                        logger_kwargs=logger_kwargs)

# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env HalfCheetah-v2 --seed 3 --l 2 --sample_action_with_dropout --dropout_rate 0.1 --action_choose_method median_and_variance_based_noise --exp_name vd3_two_layers_dropout_0_1_median_and_variance_based_noise
# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env Ant-v2 --seed 3 --l 2 --sample_action_with_dropout --dropout_rate 0.1 --action_choose_method median_and_variance_based_noise --exp_name vd3_Ant_v2_two_layers_dropout_0_1_median_and_variance_based_noise

# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env HalfCheetah-v2 --seed 3 --l 2 --exp_name vd3_two_layers_withoutdropout_no_target_policy_smooth
# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env Ant-v2 --seed 3 --l 2 --sample_action_with_dropout --dropout_rate 0.1 --action_choose_method median_and_variance_based_noise --exp_name vd3_Ant_v2_two_layers_dropout_0_1_median_and_variance_based_noise

#*************************************************************************#

# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env HalfCheetah-v2 --seed 3 --l 2 --sample_action_with_dropout --dropout_rate 0.1 --action_choose_method median_and_variance_based_noise --exp_name vd3_two_layers_dropout_0_1_median_no_target_policy_smooth

# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env HalfCheetah-v2 --seed 3 --l 2 --exp_name vd3_two_layers_withoutdropout_no_target_policy_smooth

# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env HalfCheetah-v2 --seed 3 --l 2 --sample_action_with_dropout --dropout_rate 0.1 --action_choose_method prediction_and_variance_based_noise --exp_name vd3_two_layers_dropout_0_1_prediction_no_target_policy_smooth

# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env HalfCheetah-v2 --seed 3 --l 2 --act_noise 0.2 --exp_name vd3_two_layers_withoutdropout_0_2_act_noise_no_target_policy_smooth
# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env HalfCheetah-v2 --seed 3 --l 2 --act_noise 0.3 --exp_name vd3_two_layers_withoutdropout_0_3_act_noise_no_target_policy_smooth

# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env HalfCheetah-v2 --seed 3 --l 2 --sample_action_with_dropout --dropout_rate 0.1 --action_choose_method median_and_variance_based_noise --a_var_clip_max 0.5 --a_var_clip_min 0.05 --exp_name vd3_two_layers_dropout_0_1_median_no_target_policy_smooth_05_5_varclip

# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env HalfCheetah-v2 --seed 3 --l 2 --sample_action_with_dropout --dropout_rate 0.1 --action_choose_method median_and_variance_based_noise --exp_name vd3_two_layers_dropout_0_1_median_no_target_policy_smooth_std_noise

# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env HalfCheetah-v2 --seed 3 --l
# 2 --epochs 200 --sample_action_with_dropout --dropout_rate 0.1 --action_choose_method median_and_variance_based_noise --exp_name ud3_HalfCheetah_3se
# ed_2layer_200Ep_drop_1_median_var

# python spinup/algos/vd3_delayed_dropout/vd3_delayed_dropout.py --env HalfCheetah-v2 --seed 3 --l 2 --epochs 200 --sample_action_with_dropout --dropout_rate 0.1 --action_choose_method gaussian_sample_with_scaling_and_lower_bound --exp_name ud3_HalfCheetah_3seed_2layer_200Ep_drop_1_gaussian_sample_prediction_0_1_scale_0_01_lower