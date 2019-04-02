import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.ddpg_stage import core
from spinup.algos.ddpg_stage.core import get_vars
from spinup.utils.logx import EpochLogger

import copy

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
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

Deep Deterministic Policy Gradient (DDPG)

"""
def ddpg_stage(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), nn_type='simple_dense', seed=0,
                steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
                polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
                without_start_steps=True, batch_size=100, start_steps=10000,
                without_delay_train=False,
                act_noise=0.1, policy_delay=2, max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
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
            ``q``        (batch,)          | Gives the current estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to DDPG.

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
    with tf.variable_scope('online_main'):
        pi, q, q_pi = actor_critic(x_ph, a_ph, **ac_kwargs, nn_type=nn_type)
    
    # Target networks
    with tf.variable_scope('online_target'):
        # Note that the action placeholder going to actor_critic here is 
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        pi_targ, _, q_pi_targ  = actor_critic(x2_ph, a_ph, **ac_kwargs, nn_type=nn_type)

    # Staged Main
    with tf.variable_scope('staged_main'):
        staged_pi, staged_q, staged_q_pi = actor_critic(x_ph, a_ph, **ac_kwargs, nn_type=nn_type)

    # Staged Target
    with tf.variable_scope('staged_target'):
        staged_pi_targ, _, staged_q_pi_targ = actor_critic(x2_ph, a_ph, **ac_kwargs, nn_type=nn_type)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    staged_replay_buffer = copy.deepcopy(replay_buffer)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['online_main/pi', 'online_main/q', 'online_main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

    staged_var_counts = tuple(core.count_vars(scope) for scope in ['staged_main/pi', 'staged_main/q', 'staged_main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n' % staged_var_counts)

    # Bellman backup for Q function
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_pi_targ)
    staged_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*staged_q_pi_targ)

    # DDPG losses
    pi_loss = -tf.reduce_mean(q_pi)
    q_loss = tf.reduce_mean((q-backup)**2)

    staged_pi_loss = -tf.reduce_mean(staged_q_pi)
    staged_q_loss = tf.reduce_mean((staged_q-staged_backup)**2)

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('online_main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('online_main/q'))

    staged_pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    staged_q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    staged_train_pi_op = staged_pi_optimizer.minimize(staged_pi_loss, var_list=get_vars('staged_main/pi'))
    staged_train_q_op = staged_q_optimizer.minimize(staged_q_loss, var_list=get_vars('staged_main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('online_main'), get_vars('online_target'))])

    staged_target_update = tf.group([tf.assign(v_targ, polyak*v_targ+(1-polyak)*v_main)
                                     for v_main, v_targ in zip(get_vars('staged_main'),
                                                               get_vars('staged_target'))])

    # Copy staged pi and q to online pi and q
    copy_staged_to_online = tf.group([tf.assign(v_online, v_staged)
                                      for v_online, v_staged in zip(get_vars('online_main'),
                                                                    get_vars('staged_main'))])
    # Reinitialize staged pi and q
    # import pdb; pdb.set_trace()
    reinit_staged = tf.variables_initializer(staged_q_optimizer.variables()
                                             + staged_pi_optimizer.variables()
                                             + get_vars('staged_main'))

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('online_main'), get_vars('online_target'))])

    staged_target_init = tf.group([tf.assign(v_targ, v_main)
                                   for v_main, v_targ in zip(get_vars('staged_main'),
                                                             get_vars('staged_target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)
    sess.run(staged_target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q': q})

    def get_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
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
            a = get_action(o, act_noise)
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

        # Store experience to replay buffer, but no adding to staged_replay_buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if without_delay_train:
            # Train online policy
            batch = replay_buffer.sample_batch(batch_size)
            feed_dict = {x_ph: batch['obs1'],
                         x2_ph: batch['obs2'],
                         a_ph: batch['acts'],
                         r_ph: batch['rews'],
                         d_ph: batch['done']
                         }

            # Q-learning update
            outs = sess.run([q_loss, q, train_q_op], feed_dict)
            logger.store(LossQ=outs[0], QVals=outs[1])

            # Policy update
            outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
            logger.store(LossPi=outs[0])
            # import pdb; pdb.set_trace()
            # TODO:
            staged_training_steps = 10000
            if t >= staged_training_steps:
                if t % staged_training_steps == 0:
                    staged_replay_buffer = copy.deepcopy(replay_buffer)
                    import pdb; pdb.set_trace()
                    sess.run(copy_staged_to_online)
                    sess.run(reinit_staged)
                    sess.run(staged_target_init)

                # Train staged policy
                staged_batch = staged_replay_buffer.sample_batch(batch_size)
                staged_feed_dict = {x_ph: staged_batch['obs1'],
                                    x2_ph: staged_batch['obs2'],
                                    a_ph: staged_batch['acts'],
                                    r_ph: staged_batch['rews'],
                                    d_ph: staged_batch['done']
                                    }

                staged_outs = sess.run([staged_q_loss, staged_q, staged_train_q_op], staged_feed_dict)
                logger.store(StagedLossQ=staged_outs[0], StagedQVals=staged_outs[1])
                staged_outs = sess.run([staged_pi_loss, staged_train_pi_op, staged_target_update], staged_feed_dict)
                logger.store(StagedLossPi=staged_outs[0])
            else:
                logger.store(StagedLossQ=0, StagedQVals=np.zeros((batch_size,)))
                logger.store(StagedLossPi=0)

        if d or (ep_len == max_ep_len):
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
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

                    # Q-learning update
                    outs = sess.run([q_loss, q, train_q_op], feed_dict)
                    logger.store(LossQ=outs[0], QVals=outs[1])

                    # Policy update
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
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('StagedQVals', with_min_and_max=True)
            logger.log_tabular('StagedLossPi', average_only=True)
            logger.log_tabular('StagedLossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--nn_type', choices=['tf_dense', 'simple_dense'], default='simple_dense')
    parser.add_argument('--without_start_steps', action='store_true')

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--without_delay_train', action='store_true')
    parser.add_argument('--exp_name', type=str, default='ddpg_staged')
    parser.add_argument('--hardcopy_target_nn', action="store_true", help='Target network update method: hard copy')

    parser.add_argument("--exploration-strategy", type=str, choices=["action_noise", "epsilon_greedy"],
                        default='epsilon_greedy', help='action_noise or epsilon_greedy')
    parser.add_argument("--epsilon-max", type=float, default=1.0, help='maximum of epsilon')
    parser.add_argument("--epsilon-min", type=float, default=.01, help='minimum of epsilon')
    parser.add_argument("--epsilon-decay", type=float, default=.001, help='epsilon decay')

    parser.add_argument("--data_dir", type=str, default=None)

    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, datestamp=True)

    # if args.hardcopy_target_nn:
    #     polyak = 0

    ddpg_stage(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                nn_type=args.nn_type,
                without_start_steps = args.without_start_steps,
                without_delay_train=args.without_delay_train,
                gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                logger_kwargs=logger_kwargs)

# python  spinup/algos/ddpg/ddpg.py --env HalfCheetah-v2 --seed 3 --l 2 --exp_name ddpg_two_layers_delay_policy
# python  spinup/algos/ddpg/ddpg.py --env Ant-v2 --seed 3 --l 2 --exp_name ddpg_Ant_v2_two_layers_delay_policy
# python  spinup/algos/ddpg/ddpg.py --env Humanoid-v2 --seed 3 --l 2 --exp_name ddpg_Humanoid_v2_two_layers_delay_policy