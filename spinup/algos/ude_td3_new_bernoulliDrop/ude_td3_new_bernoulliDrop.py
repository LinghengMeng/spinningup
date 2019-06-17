import numpy as np
import numpy.matlib
import tensorflow as tf
import gym
import time
from spinup.algos.ude_td3_new_bernoulliDrop import core
from spinup.algos.ude_td3_new_bernoulliDrop.core import MLP, BeroulliDropoutMLP, get_vars, count_vars
from spinup.utils.logx import EpochLogger, Logger
import os.path as osp

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
              step_index, steps_per_epoch, start_time, **kwargs):
        # Save experiences in disk
        self.log_experiences(obs, act, rew, next_obs, done,
                             step_index, steps_per_epoch, start_time, **kwargs)
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
                        step_index, steps_per_epoch, start_time, **kwargs):
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
        # Log other data
        for key, value in kwargs.items():
            for i, v in enumerate(np.array(value).flatten(order='C')):
                self.experience_logger.log_tabular('{}_{}'.format(key, i), v)
        # Log done
        self.experience_logger.log_tabular('d', done)
        self.experience_logger.log_tabular('Time', time.time() - start_time)
        self.experience_logger.dump_tabular(print_data=False)

"""

uncertainty driven exploration TD3 (Twin Delayed DDPG)

"""
def ude_td3_new_bernoulliDrop(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
                steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
                polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
                without_start_steps=True, batch_size=100, start_steps=10000,
                without_delay_train=False,
                uncertainty_driven_exploration=True, dropout_rate=0.05, n_post=100, concentration_factor=0.5,
                uncertainty_policy_delay=5000,
                act_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2,
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

    hidden_sizes = list(ac_kwargs['hidden_sizes'])

    # Main actor-critic
    with tf.variable_scope('main'):
        actor = MLP(hidden_sizes+[act_dim], hidden_activation=tf.keras.activations.relu,
                    output_activation=tf.keras.activations.tanh)
        # critic_1 = MLP(hidden_sizes + [1], hidden_activation=tf.keras.activations.relu,
        #                output_activation=tf.keras.activations.linear)
        # critic_2 = MLP(hidden_sizes + [1], hidden_activation=tf.keras.activations.relu,
        #                output_activation=tf.keras.activations.linear)
        critic_1 = BeroulliDropoutMLP(hidden_sizes + [1], weight_regularizer=1e-6, dropout_rate=dropout_rate,
                                      hidden_activation=tf.keras.activations.relu,
                                      output_activation=tf.keras.activations.linear)
        critic_2 = BeroulliDropoutMLP(hidden_sizes + [1], weight_regularizer=1e-6, dropout_rate=dropout_rate,
                                      hidden_activation=tf.keras.activations.relu,
                                      output_activation=tf.keras.activations.linear)

        pi = act_limit*actor(x_ph)
        q1 = tf.squeeze(critic_1(tf.concat([x_ph,a_ph], axis=-1)), axis=1)
        q2 = tf.squeeze(critic_2(tf.concat([x_ph, a_ph], axis=-1)), axis=1)
        q1_pi = tf.squeeze(critic_1(tf.concat([x_ph, pi], axis=-1)), axis=1)
    
    # Target actor-critic
    with tf.variable_scope('target'):
        actor_targ = MLP(hidden_sizes + [act_dim], hidden_activation=tf.keras.activations.relu,
                         output_activation=tf.keras.activations.tanh)
        critic_1_targ = MLP(hidden_sizes + [1], hidden_activation=tf.keras.activations.relu,
                            output_activation=tf.keras.activations.linear)
        critic_2_targ = MLP(hidden_sizes + [1], hidden_activation=tf.keras.activations.relu,
                            output_activation=tf.keras.activations.linear)

        pi_targ = act_limit*actor_targ(x2_ph)
        # Target policy smoothing, by adding clipped noise to target actions
        epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2 = tf.clip_by_value(a2, -act_limit, act_limit)
        # a2 = pi_targ

        q1_targ = tf.squeeze(critic_1_targ(tf.concat([x2_ph, a2], axis=-1)), axis=1)
        q2_targ = tf.squeeze(critic_2_targ(tf.concat([x2_ph, a2], axis=-1)), axis=1)

    # Create LazyBernoulliDropoutMLP:
    #       which copys weights from MLP by
    #           sess.run(lazy_ber_drop_mlp_update)
    #       , then post sample predictions with dropout masks.
    with tf.variable_scope('LazyBernoulliDropoutUncertaintySample'):
        # define placeholder for parallel sampling
        #   batch x n_post x dim
        lazy_bernoulli_dropout_actor = BeroulliDropoutMLP(hidden_sizes + [act_dim], weight_regularizer=1e-6, dropout_rate=dropout_rate,
                                                        hidden_activation=tf.keras.activations.relu,
                                                        output_activation=tf.keras.activations.sigmoid)
        lazy_ber_drop_pi = lazy_bernoulli_dropout_actor(x_ph, training=True)  # Set training=True to sample with dropout masks
        lazy_ber_drop_actor_update = tf.group([tf.assign(v_lazy_ber_drop_mlp, v_mlp)
                                               for v_mlp, v_lazy_ber_drop_mlp in
                                               zip(actor.variables, lazy_bernoulli_dropout_actor.variables)])

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
                                 logger_fname='experiences_log.txt', **logger_kwargs)

    # Count variables
    print('\nNumber of parameters: \t pi: {:d}, \t q1: {:d}, \t q2: {:d}, \t total: {:d}\n'.format(count_vars(actor.variables),
                                                                                                   count_vars(critic_1.variables),
                                                                                                   count_vars(critic_2.variables),
                                                                                                   count_vars(get_vars('main'))))

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
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=actor.variables)
    train_q_op = q_optimizer.minimize(q_loss, var_list=(critic_1.variables + critic_2.variables))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)
    sess.run(lazy_ber_drop_actor_update)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q1': q1, 'q2': q2})

    def get_uncertainty_driven_explore_action(o):
        o_post_samples = np.matlib.repmat(o.reshape(1,-1), n_post, 1)  # repmat x for post sampling

        # 1. Generate action Prediction
        a_pred = sess.run(pi, feed_dict={x_ph: o.reshape(1, -1)})[0]

        # 2. Generate post sampled actions
        a_post = sess.run(lazy_ber_drop_pi, feed_dict={x_ph: o_post_samples})

        a = np.zeros((act_dim,))
        if act_dim > 1:
            a_cov = np.cov(a_post, rowvar=False)
            a_cov_shaped = concentration_factor * a_cov
            a = np.random.multivariate_normal(a_pred, a_cov_shaped, 1)[0]
            unc_a = a_cov
        else:
            a_std = np.std(a_post, axis=0)
            a_std_shaped = concentration_factor * a_std
            a = np.random.normal(a_pred, a_std_shaped, 1)[0]
            unc_a = a_std
        # import pdb; pdb.set_trace()
        # TODO: logdet as intrinsic reward
        return np.clip(a, -act_limit, act_limit)

    def get_gaussian_noise_explore_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def get_action_test(o):
        """Get deterministic action without exploration."""
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1, -1)})[0]
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
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps:
            if uncertainty_driven_exploration:
                if t%uncertainty_policy_delay==0:
                    sess.run(lazy_ber_drop_actor_update)
                a = get_uncertainty_driven_explore_action(o)
            else:
                a = get_gaussian_noise_explore_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a,  r, o2, d, t, steps_per_epoch, start_time)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if without_delay_train:
            batch = replay_buffer.sample_batch(batch_size)
            feed_dict = {x_ph: batch['obs1'],
                         x2_ph: batch['obs2'],
                         a_ph: batch['acts'],
                         r_ph: batch['rews'],
                         d_ph: batch['done']
                         }
            q_step_ops = [q_loss, q1, q2, train_q_op]
            outs = sess.run(q_step_ops, feed_dict)
            logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

            # Delayed policy update
            outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
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
                    feed_dict = {x_ph: batch['obs1'],
                                 x2_ph: batch['obs2'],
                                 a_ph: batch['acts'],
                                 r_ph: batch['rews'],
                                 d_ph: batch['done']
                                }
                    q_step_ops = [q_loss, q1, q2, train_q_op]
                    outs = sess.run(q_step_ops, feed_dict)
                    logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                    if j % policy_delay == 0:
                        # Delayed policy update
                        outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                        logger.store(LossPi=outs[0])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

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
    parser.add_argument('--without_start_steps', action='store_true')
    parser.add_argument('--without_delay_train', action='store_true')
    parser.add_argument('--exp_name', type=str, default='ude_td3_new')
    parser.add_argument('--uncertainty_driven_exploration',  action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.05)
    parser.add_argument('--n_post', type=int, default=100)
    parser.add_argument('--concentration_factor', type=float, default=0.5)
    parser.add_argument('--uncertainty_policy_delay', type=int, default=5000)
    parser.add_argument('--act_noise', type=float, default=0.1)
    args = parser.parse_args()

    # Set log data saving directory
    from spinup.utils.run_utils import setup_logger_kwargs
    data_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))),
                        'spinup_data')
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)

    ude_td3_new_bernoulliDrop(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                without_start_steps=args.without_start_steps,
                without_delay_train=args.without_delay_train,
                gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                uncertainty_driven_exploration=args.uncertainty_driven_exploration,
                              dropout_rate=args.dropout_rate,
                n_post=args.n_post,
                concentration_factor=args.concentration_factor,
                uncertainty_policy_delay=args.uncertainty_policy_delay,
                act_noise=args.act_noise,
                logger_kwargs=logger_kwargs)

# python ./spinup/algos/td3/td3.py --env HalfCheetah-v2 --seed 3 --l 2 --exp_name td3_two_layers
# python ./spinup/algos/td3/td3.py --env Ant-v2 --seed 3 --l 2 --exp_name td3_Ant_v2_two_layers