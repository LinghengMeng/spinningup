import os
import os.path as osp
import time
import gym
from gym.wrappers import Monitor
import numpy as np
import tensorflow as tf
import atari_py
from spinup.algos.dqn.gym_wrappers import MainGymWrapper
from spinup.utils.logx import EpochLogger, Logger
from spinup.algos.dqn.replay_buffer import ReplayBuffer
from spinup.algos.dqn import core
from spinup.algos.dqn.core import DQN

FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)


def make_train_and_test_envs(env_name, max_ep_len,
                             output_dir, record_video_every=100, save_video=True):
    # Add env Monitor wrapper to record video
    monitor_path_train = os.path.join(output_dir, 'videos_train')
    monitor_path_test = os.path.join(output_dir, 'videos_test')
    if not os.path.exists(monitor_path_train):
        os.makedirs(monitor_path_train)
    if not os.path.exists(monitor_path_test):
        os.makedirs(monitor_path_test)

    env_train = gym.make(env_name)
    env_train._max_episode_steps = max_ep_len
    env_test = gym.make(env_name)
    env_test._max_episode_steps = max_ep_len
    if save_video:
        env_train = Monitor(env_train, directory=monitor_path_train,
                            video_callable=lambda count: count % record_video_every == 0,
                            resume=True)
        env_test = Monitor(env_test, directory=monitor_path_test,
                           video_callable=lambda count: count % record_video_every == 0,
                           resume=True)
    # MainGymWrapper preprocesses state
    env_train, env_test = MainGymWrapper.wrap(env_train), MainGymWrapper.wrap(env_test)
    return env_train, env_test

def dqn(seed, game_name, render,
        gamma=0.99, q_lr=0.00025,
        start_steps=50000, epochs=1000, steps_per_epoch=5000, max_ep_len=5000,
        replay_size=int(1e6), batch_size=32, target_net_update_frequency = 10000,
        epsilon_max = 1, epsilon_min = 0.1, epsilon_decay_steps = 1000000,
        record_video_every=50,
        logger_kwargs=dict()):
    """

    :param seed:
    :param game_name:
    :param render:
    :param logger_kwargs:
    :param max_ep_len: the maximum episode length set to 5 minutes of real-time play (18,000 frames)
    :return:
    """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env_name = game_name + "Deterministic-v4"  # Handles frame skipping (4) at every iteration
    env, test_env = make_train_and_test_envs(env_name, max_ep_len,
                                             logger_kwargs['output_dir'], record_video_every)
    obs_dim = FRAMES_IN_OBSERVATION*FRAME_SIZE*FRAME_SIZE
    act_dim = env.action_space.n

    a_ph, r_ph, d_ph = core.placeholders(None, None, None)
    x_ph = tf.placeholder(dtype=tf.float32, shape=(None, FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE))
    x2_ph = tf.placeholder(dtype=tf.float32, shape=(None, FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE))

    # Define main DQN
    with tf.variable_scope('main'):
        dqn_model = DQN(act_dim)
        q = dqn_model(x_ph)
        pi = tf.argmax(q, axis=1)

    # Define target DQN
    with tf.variable_scope('target'):
        dqn_model_targ = DQN(act_dim)
        q_targ = dqn_model_targ(x2_ph)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
                                 logger_fname='experiences_log.txt', **logger_kwargs)

    # Bellman backup for Q functions
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*tf.reduce_max(q_targ, axis=1))

    # Train operation
    q_loss = tf.reduce_mean((q_targ - q)**2)
    q_optimizer = tf.train.RMSPropOptimizer(learning_rate=q_lr)
    train_q_op = q_optimizer.minimize(q_loss, var_list=dqn_model.variables)

    # Update target q-network
    target_update = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(dqn_model.variables, dqn_model_targ.variables)])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_update)

    # Get action
    def get_action(o, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, act_dim, size=1)
        return sess.run(pi, feed_dict={x_ph: np.expand_dims(np.asarray(o), axis=0)})[0]

    def get_action_test(o):
        """Get deterministic action without exploration."""
        return sess.run(pi, feed_dict={x_ph: np.expand_dims(np.asarray(o), axis=0)})[0]

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

    # Initialize epsilon
    epsilon_decay = (epsilon_max-epsilon_min)/epsilon_decay_steps
    epsilon = epsilon_max

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        if t%1000==0:
            print('Step: {}'.format(t))
        if render:
            env.render()

        if t > start_steps:
            a = get_action(o, epsilon)
            epsilon = max(epsilon_min, epsilon-epsilon_decay)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, t, steps_per_epoch, start_time)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if t > batch_size:
            # Train
            batch = replay_buffer.sample_batch(batch_size=batch_size)
            feed_dict = {x_ph: batch['obs1'],
                         x2_ph: batch['obs2'],
                         a_ph: batch['acts'],
                         r_ph: batch['rews'],
                         d_ph: batch['done']}
            q_step_ops = [q_loss, q, train_q_op]
            outs = sess.run(q_step_ops, feed_dict=feed_dict)
            logger.store(LossQ=outs[0], QVals=outs[1])

        # Update target q-net

        if t % target_net_update_frequency == 0:
            sess.run(target_update)

        if d or (ep_len == max_ep_len):
            # Log episode data
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs - 1):
            #     logger.save_state({'env': env}, None)

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
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    available_games = list((''.join(x.capitalize() or '_' for x in word.split('_')) for word in atari_py.list_games()))
    parser.add_argument("-g", "--game",
                        help="Choose from available games: " + str(available_games) + ". Default is 'breakout'.",
                        default="SpaceInvaders")
    parser.add_argument("-m", "--mode",
                        help="Choose from available modes: ddqn_train, ddqn_test, ge_train, ge_test. Default is 'ddqn_training'.",
                        default="ddqn_training")
    parser.add_argument("-r", "--render", help="Choose if the game should be rendered. Default is 'False'.",
                        default=False, type=bool)
    parser.add_argument("-tsl", "--total_step_limit",
                        help="Choose how many total steps (frames visible by agent) should be performed. Default is '5000000'.",
                        default=5000000, type=int)
    parser.add_argument("-trl", "--total_run_limit",
                        help="Choose after how many runs we should stop. Default is None (no limit).", default=None,
                        type=int)
    parser.add_argument("-c", "--clip", help="Choose whether we should clip rewards to (0, 1) range. Default is 'True'",
                        default=True, type=bool)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='dqn_atari')
    args = parser.parse_args()

    print('Selected game: {}'.format(args.game))
    print('Selected mode: {}'.format(args.mode))
    print('Should render: {}'.format(args.render))
    print('Should clip: {}'.format(args.clip))
    print('Total step limit: {}'.format(args.total_step_limit))
    print('Total run limit: {}'.format(args.total_run_limit))

    # Set log data saving directory
    from spinup.utils.run_utils import setup_logger_kwargs

    data_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))),
                        'spinup_data')
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)
    dqn(seed=args.seed, game_name=args.game, render=args.render, logger_kwargs=logger_kwargs)
