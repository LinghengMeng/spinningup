import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.ude_td3 import core
from spinup.algos.ude_td3.core import get_vars
from spinup.algos.ude_td3.investigate_uncertainty import DropoutUncertaintyModule,ObsSampleUncertaintyModule
from spinup.utils.logx import EpochLogger, Logger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size,
                 logger_fname='experiences_log.txt', **logger_kwargs):
        # ExperienceLogger: save experiences for supervised learning
        logger_kwargs['output_fname'] = logger_fname
        self.experience_logger = Logger(**logger_kwargs)
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done,
              uncertainty, step_index, steps_per_epoch, start_time):
        # Save experiences in disk
        self.log_experiences(obs, act, rew, next_obs, done,
                             uncertainty, step_index, steps_per_epoch, start_time)
        # Save experiences in memory
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

    def log_experiences(self, obs, act, rew, next_obs, done,
                        uncertainty, step_index, steps_per_epoch, start_time):
        self.experience_logger.log_tabular('Epoch', step_index // steps_per_epoch)
        self.experience_logger.log_tabular('Step', step_index)
        # Log observation
        for i, o_i in enumerate(obs):
            self.experience_logger.log_tabular('o_{}'.format(i), o_i)
        # Log action
        for i, a_i in enumerate(act):
            self.experience_logger.log_tabular('a_{}'.format(i), a_i)
        # Log reward
        self.experience_logger.log_tabular('r', rew)
        # Log next observation
        for i, o2_i in enumerate(next_obs):
            self.experience_logger.log_tabular('o2_{}'.format(i), o2_i)
        # TODO: save uncertainty
        # Log uncertainty: flatten in row-major order
        for i, unc_i in enumerate(np.array(uncertainty).flatten(order='C')):
            self.experience_logger.log_tabular('unc_{}'.format(i), unc_i)
        # Log done
        self.experience_logger.log_tabular('d', done)
        self.experience_logger.log_tabular('Time', time.time() - start_time)
        self.experience_logger.dump_tabular(print_data=False)

"""

UDE-TD3 (Uncertainty Driven Exploration Twin Delayed DDPG)

"""
def ude_td3(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
            steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
            polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
            without_start_steps=True, batch_size=100, start_steps=10000,
            without_delay_train=False,
            act_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2,
            max_ep_len=1000, logger_kwargs=dict(), save_freq=1,
            n_post_action=10,
            uncertainty_method='dropout',
            sample_obs_std=1,
            uncertainty_driven_exploration=False,
            uncertainty_policy_delay=5000,
            dropout_rate=0.1,
            concentration_factor=0.1,
            minimum_exploration_level=0,
            ):
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
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

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
        pi, _, pi_dropout_mask_generator, pi_dropout_mask_phs, \
        q1, _, q1_dropout_mask_generator, q1_dropout_mask_phs, q1_pi, _, \
        q2, _, q2_dropout_mask_generator, q2_dropout_mask_phs = actor_critic(x_ph, a_ph, **ac_kwargs, dropout_rate=0)


    with tf.variable_scope('uncertainty'):
        pi_unc, _, pi_dropout_mask_generator_unc, pi_dropout_mask_phs_unc, \
        q1_unc, _, q1_dropout_mask_generator_unc, q1_dropout_mask_phs_unc, q1_pi_unc, _, \
        q2_unc, _, q2_dropout_mask_generator_unc, q2_dropout_mask_phs_unc = actor_critic(x_ph, a_ph, **ac_kwargs, dropout_rate=dropout_rate)

        # Initialize uncertainty module
        obs_set_size = 10
        track_obs_set_unc_frequency = 100  # every 100 steps
        if uncertainty_method == 'dropout':
            pi_unc_module = DropoutUncertaintyModule(act_dim, obs_dim, n_post_action,
                                                     obs_set_size, track_obs_set_unc_frequency,
                                                     pi_unc, x_ph,
                                                     pi_dropout_mask_phs_unc, pi_dropout_mask_generator_unc,
                                                     logger_kwargs,
                                                     tf_var_scope_main='main', tf_var_scope_unc='uncertainty')
        elif uncertainty_method == 'gaussian_obs_sample':
            pi_unc_module = ObsSampleUncertaintyModule(act_dim, obs_dim, n_post_action,
                                                       obs_set_size, track_obs_set_unc_frequency,
                                                       pi_unc, x_ph,
                                                       pi_dropout_mask_phs_unc, pi_dropout_mask_generator_unc,
                                                       logger_kwargs, sample_obs_std)
        else:
            raise ValueError('Please choose a proper uncertainty_method!')

    # Target policy network
    with tf.variable_scope('target'):
        pi_targ, _, pi_dropout_mask_generator_targ, pi_dropout_mask_phs_targ, \
        _, _, _, _, _, _, \
        _, _, _, _ = actor_critic(x2_ph, a_ph, **ac_kwargs, dropout_rate=0)
    
    # Target Q networks
    with tf.variable_scope('target', reuse=True):
        # TODO: add with_out_policy_smoothing
        # Target policy smoothing, by adding clipped noise to target actions
        epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2 = tf.clip_by_value(a2, -act_limit, act_limit)

        # Target Q-values, using action from target policy
        _, _, _, _, \
        q1_targ, _, q1_dropout_mask_generator_targ, q1_dropout_mask_phs_targ, _, _, \
        q2_targ, _, q2_dropout_mask_generator_targ, q2_dropout_mask_phs_targ = actor_critic(x2_ph, a2, **ac_kwargs, dropout_rate=0)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
                                 logger_fname='experiences_log.txt', **logger_kwargs)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q functions, using Clipped Double-Q targets
    min_q_targ = tf.minimum(q1_targ, q2_targ)
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*min_q_targ)

    # TD3 losses
    pi_loss = -tf.reduce_mean(q1_pi)
    q1_loss = tf.reduce_mean((q1-backup)**2)
    q2_loss = tf.reduce_mean((q2-backup)**2)
    q_loss = q1_loss + q2_loss

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
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q1': q1, 'q2': q2})

    def get_action_train(o, noise_scale, pi_unc_module, step_index):
        feed_dictionary = {x_ph: o.reshape(1, -1)}
        if uncertainty_driven_exploration:
            # 1. Generate Prediction
            for mask_i in range(len(pi_dropout_mask_phs)):
                feed_dictionary[pi_dropout_mask_phs[mask_i]] = np.ones(pi_dropout_mask_phs[mask_i].shape.as_list())
            a_prediction = sess.run(pi, feed_dict=feed_dictionary)[0]

            # 2. Generate post samples in Non-parallel way
            #    (Tried to use ray implementing parallel post sampling but no speed up in one machine.)
            a_post = pi_unc_module.get_post_samples(o, sess, step_index)

            # 3. Generate uncertainty-driven exploratory action
            a = np.zeros((act_dim,))
            if act_dim >1:
                a_cov = np.cov(a_post, rowvar=False)
                a_cov_shaped = concentration_factor * a_cov
                # TODO: only keep one
                a = np.random.multivariate_normal(a_prediction, a_cov_shaped, 1)[0]
                a = a_prediction
                uncertainty = a_cov
            else:
                a_std = np.std(a_post, axis=0)
                a_std_shaped = concentration_factor * a_std + minimum_exploration_level * np.ones(a_std.shape)
                # TODO: only keep one
                a = np.random.normal(a_prediction, a_std_shaped, 1)[0]
                a = a_prediction
                uncertainty = np.var(a_post, axis=0)
        else:
            for mask_i in range(len(pi_dropout_mask_phs)):
                feed_dictionary[pi_dropout_mask_phs[mask_i]] = np.ones(pi_dropout_mask_phs[mask_i].shape.as_list())
            a = sess.run(pi, feed_dict=feed_dictionary)[0]
            a += noise_scale * np.random.randn(act_dim)
            uncertainty = 0
        unc_based_reward = np.mean(np.abs(uncertainty))
        # unc_based_reward = np.sum(np.abs(uncertainty))
        return np.clip(a, -act_limit, act_limit), uncertainty, unc_based_reward

    def get_action_test(o):
        """Get deterministic action without exploration."""
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
    o, r, d, ep_ret, ep_len, ep_uncertainty = env.reset(), 0, False, 0, 0, 0
    total_steps = steps_per_epoch * epochs

    # No dropout for training phase: set all dropout masks to one
    feed_dict_train = {}
    for mask_i in range(len(pi_dropout_mask_phs)):
        feed_dict_train[pi_dropout_mask_phs[mask_i]] = np.ones(pi_dropout_mask_phs[mask_i].shape.as_list())
    for mask_i in range(len(q1_dropout_mask_phs)):
        feed_dict_train[q1_dropout_mask_phs[mask_i]] = np.ones(q1_dropout_mask_phs[mask_i].shape.as_list())
    for mask_i in range(len(q2_dropout_mask_phs)):
        feed_dict_train[q2_dropout_mask_phs[mask_i]] = np.ones(q2_dropout_mask_phs[mask_i].shape.as_list())
    for mask_i in range(len(pi_dropout_mask_phs_targ)):
        feed_dict_train[pi_dropout_mask_phs_targ[mask_i]] = np.ones(pi_dropout_mask_phs_targ[mask_i].shape.as_list())
    for mask_i in range(len(q1_dropout_mask_phs_targ)):
        feed_dict_train[q1_dropout_mask_phs_targ[mask_i]] = np.ones(q1_dropout_mask_phs_targ[mask_i].shape.as_list())
    for mask_i in range(len(q2_dropout_mask_phs_targ)):
        feed_dict_train[q2_dropout_mask_phs_targ[mask_i]] = np.ones(q2_dropout_mask_phs_targ[mask_i].shape.as_list())

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps:
            a, uncertainty, unc_based_reward = get_action_train(o, act_noise, pi_unc_module, step_index=t)
        else:
            a = env.action_space.sample()
            # TODO：keep the same dimension with real covariance
            if uncertainty_driven_exploration:
                uncertainty = np.zeros((act_dim, act_dim))
            else:
                uncertainty = 0
            unc_based_reward = 0

        # Sample an observation set to track their uncertainty trajectories
        if t > start_steps:
            if pi_unc_module.obs_set_is_empty:
                pi_unc_module.sample_obs_set_from_replay_buffer(replay_buffer)

            if t % pi_unc_module.track_obs_set_unc_frequency == 0:
                pi_unc_module.calculate_obs_set_uncertainty(sess, t // steps_per_epoch, t)

            # Update uncertainty policy to current policy
            if t % uncertainty_policy_delay == 0:
                pi_unc_module.uncertainty_policy_update(sess)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        ep_uncertainty += np.sum(uncertainty)

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r+unc_based_reward, o2, d, uncertainty, t, steps_per_epoch, start_time)

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
            q_step_ops = [q_loss, q1, q2, train_q_op]
            outs = sess.run(q_step_ops, feed_dict_train)
            logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

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

                    q_step_ops = [q_loss, q1, q2, train_q_op]
                    outs = sess.run(q_step_ops, feed_dict_train)
                    logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                    if j % policy_delay == 0:
                        # Delayed policy update
                        outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict_train)
                        logger.store(LossPi=outs[0])

            logger.store(EpRet=ep_ret, EpLen=ep_len,
                         EpUncertainty=ep_uncertainty)
            o, r, d, ep_ret, ep_len, ep_uncertainty = env.reset(), 0, False, 0, 0, 0

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
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('EpUncertainty', with_min_and_max=True)
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
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--without_start_steps', action='store_true')
    parser.add_argument('--without_delay_train', action='store_true')
    parser.add_argument('--exp_name', type=str, default='ude_td3')

    parser.add_argument('--n_post_action', type=int, default=10)
    parser.add_argument('--uncertainty_method', choices=['dropout', 'gaussian_obs_sample'],
                        default='dropout')
    parser.add_argument('--sample_obs_std', type=float, default=1)
    parser.add_argument('--uncertainty_driven_exploration', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--uncertainty_policy_delay', type=int, default=5000)
    parser.add_argument('--concentration_factor', type=float, default=0.5)
    parser.add_argument('--minimum_exploration_level', type=float, default=0)

    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, datestamp=True)
    # import pdb; pdb.set_trace()
    ude_td3(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
            gamma=args.gamma, seed=args.seed, epochs=args.epochs,
            replay_size=args.replay_size,
            without_start_steps=args.without_start_steps,
            without_delay_train=args.without_delay_train,
            n_post_action=args.n_post_action,
            uncertainty_method=args.uncertainty_method,
            sample_obs_std=args.sample_obs_std,
            uncertainty_driven_exploration=args.uncertainty_driven_exploration,
            uncertainty_policy_delay=args.uncertainty_policy_delay,
            dropout_rate=args.dropout_rate,
            concentration_factor=args.concentration_factor,
            minimum_exploration_level=args.minimum_exploration_level,
            logger_kwargs=logger_kwargs)
