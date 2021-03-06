import numpy as np
import tensorflow as tf
import pybulletgym
import roboschool
import gym
import time
from spinup.algos.sac_n_step_singleQ import core
from spinup.algos.sac_n_step_singleQ.core import get_vars
from spinup.utils.logx import EpochLogger
import os.path as osp


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
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
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def sample_batch_n_step(self, batch_size=32, n_step=1):
        """
        return training batch for n-step experiences
        :param batch_size:
        :param n_step:
        :return: dict:
            'obs1': batch_size x n_step x obs_dim
            'obs2': batch_size x n_step x obs_dim
            'acts': batch_size x n_step x act_dim
            'rews': batch_size x n_step
            'done': batch_size x n_step
        """
        idxs = np.random.randint(0, self.size - (n_step - 1), size=batch_size)
        batch_obs1 = np.zeros([batch_size, n_step, self.obs_dim])
        batch_obs2 = np.zeros([batch_size, n_step, self.obs_dim])
        batch_acts = np.zeros([batch_size, n_step, self.act_dim])
        batch_rews = np.zeros([batch_size, n_step])
        batch_done = np.zeros([batch_size, n_step])
        for i in range(n_step):
            batch_obs1[:, i, :] = self.obs1_buf[idxs + i]
            batch_obs2[:, i, :] = self.obs2_buf[idxs + i]
            batch_acts[:, i, :] = self.acts_buf[idxs + i]
            batch_rews[:, i] = self.rews_buf[idxs + i]
            batch_done[:, i] = self.done_buf[idxs + i]
        # Set all done after the fist met one to 1
        done_index = np.asarray(np.where(batch_done == 1))
        for d_i in range(done_index.shape[1]):
            x, y = done_index[:, d_i]
            batch_done[x, y:] = 1
        batch_done = np.hstack((np.zeros((batch_size, 1)), batch_done))
        return dict(obs1=batch_obs1[:, 0, :],
                    obs2=batch_obs2[:, -1, :],
                    acts=batch_acts[:, 0, :],
                    rews=batch_rews,
                    done=batch_done)


"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""


def sac_n_step(env_name, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
               steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
               polyak=0.995, lr=1e-3, alpha=0.2,
               n_step=5, batch_size=100, start_steps=10000,
               without_delay_train=False,
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
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to SAC.

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

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = gym.make(env_name), gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph = core.placeholders(obs_dim, act_dim, obs_dim)
    # r_ph = tf.placeholder(dtype=tf.float32, shape=(None, n_step))
    # d_ph = tf.placeholder(dtype=tf.float32, shape=(None, n_step))
    r_ph = tf.placeholder(dtype=tf.float32, shape=(None, None))
    d_ph = tf.placeholder(dtype=tf.float32, shape=(None, None))
    n_step_ph = tf.placeholder(dtype=tf.float32, shape=())

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Target value network
    with tf.variable_scope('target'):
        _, _, _, _, _, _, _, v_targ = actor_critic(x2_ph, a_ph, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n') % var_counts)


    # Targets for Q and V regression
    # q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_targ)
    # q_backup = tf.stop_gradient(
    #     tf.reduce_sum(tf.multiply([gamma ** (i) for i in range(n_step)] * (1 - d_ph), r_ph), axis=1)
    #     + gamma ** n_step * (1 - d_ph[:, -1]) * v_targ)
    q_backup = tf.stop_gradient(
        tf.reduce_sum(tf.multiply(tf.pow(gamma, tf.range(0, n_step_ph))
                                  * (1 - tf.slice(d_ph, [0, 0], [batch_size, n_step])), r_ph), axis=1)
        + gamma ** n_step_ph * (1 - tf.reshape(tf.slice(d_ph, [0, n_step], [batch_size, 1]), [-1])) * v_targ)
    v_backup = tf.stop_gradient(q1_pi - alpha * logp_pi)

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
    v_loss = 0.5 * tf.reduce_mean((v_backup - v) ** 2)
    value_loss = q1_loss + v_loss

    # Policy train op
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q') + get_vars('main/v')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, v_loss, q1, v, logp_pi,
                train_pi_op, train_value_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                          outputs={'mu': mu, 'pi': pi, 'q1': q1, 'v': v})

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1, -1)})[0]

    def test_agent(n=10):
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
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
        use the learned policy. 
        """
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if t > batch_size and without_delay_train:
            # batch = replay_buffer.sample_batch(batch_size)
            batch = replay_buffer.sample_batch_n_step(batch_size, n_step=n_step)
            feed_dict = {x_ph: batch['obs1'],
                         x2_ph: batch['obs2'],
                         a_ph: batch['acts'],
                         r_ph: batch['rews'],
                         d_ph: batch['done'],
                         n_step_ph: n_step
                         }
            outs = sess.run(step_ops, feed_dict)
            logger.store(LossPi=outs[0], LossQ1=outs[1],
                         LossV=outs[2], Q1Vals=outs[3],
                         VVals=outs[4], LogPi=outs[5])

        if d or (ep_len == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            if not without_delay_train:
                for j in range(ep_len):
                    # batch = replay_buffer.sample_batch(batch_size)
                    batch = replay_buffer.sample_batch_n_step(batch_size, n_step=n_step)
                    feed_dict = {x_ph: batch['obs1'],
                                 x2_ph: batch['obs2'],
                                 a_ph: batch['acts'],
                                 r_ph: batch['rews'],
                                 d_ph: batch['done'],
                                 n_step_ph: n_step
                                 }
                    outs = sess.run(step_ops, feed_dict)
                    logger.store(LossPi=outs[0], LossQ1=outs[1],
                                 LossV=outs[2], Q1Vals=outs[3],
                                 VVals=outs[4], LogPi=outs[5])

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
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
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
    parser.add_argument('--n_step', type=int, default=5)
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--without_delay_train', action='store_true')
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--exp_name', type=str, default='sac_singleQ_2L_200Ep')
    parser.add_argument('--data_dir', type=str, default='spinup_data')
    args = parser.parse_args()

    # Set log data saving directory
    from spinup.utils.run_utils import setup_logger_kwargs

    data_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))),
                        args.data_dir)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)

    sac_n_step(env_name=args.env, actor_critic=core.mlp_actor_critic,
               ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
               n_step=args.n_step, replay_size=args.replay_size,
               gamma=args.gamma, seed=args.seed, epochs=args.epochs,
               without_delay_train=args.without_delay_train,
               start_steps=args.start_steps,
               logger_kwargs=logger_kwargs)