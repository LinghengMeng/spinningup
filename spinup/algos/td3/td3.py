import os
import numpy as np
import tensorflow as tf
import gym
# import roboschool
import pybulletgym
import time
from spinup.algos.td3 import core
from spinup.algos.td3.core import MLP
from spinup.utils.logx import EpochLogger
import os.path as osp

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

TD3 (Twin Delayed DDPG)

"""
def td3(env_name, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(),
        actor_hidden_layers=[300, 300], critic_hidden_layers=[300, 300],
        reward_scale = 1, seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
        without_start_steps=True, batch_size=100, start_steps=10000,
        without_delay_train=False, without_target_policy_smoothing=False,
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

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = gym.make(env_name), gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    actor_hidden_sizes = actor_hidden_layers
    critic_hidden_sizes = critic_hidden_layers
    actor_hidden_activation = tf.keras.activations.relu
    actor_output_activation = tf.keras.activations.tanh
    critic_hidden_activation = tf.keras.activations.relu
    critic_output_activation = tf.keras.activations.linear

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        actor = MLP(layer_sizes=actor_hidden_sizes + [act_dim],
                    hidden_activation=actor_hidden_activation, output_activation=actor_output_activation)
        critic1 = MLP(layer_sizes=critic_hidden_sizes + [1],
                      hidden_activation=critic_hidden_activation, output_activation=critic_output_activation)
        critic2 = MLP(layer_sizes=critic_hidden_sizes + [1],
                      hidden_activation=critic_hidden_activation, output_activation=critic_output_activation)
        pi = act_limit * actor(x_ph)
        q1 = tf.squeeze(critic1(tf.concat([x_ph, a_ph], axis=-1)), axis=1)
        q1_pi = tf.squeeze(critic1(tf.concat([x_ph, pi], axis=-1)), axis=1)
        q2 = tf.squeeze(critic2(tf.concat([x_ph, a_ph], axis=-1)), axis=1)
        q2_pi = tf.squeeze(critic2(tf.concat([x_ph, pi], axis=-1)), axis=1)
    
    # Target policy network
    with tf.variable_scope('target'):
        actor_targ = MLP(layer_sizes=actor_hidden_sizes + [act_dim],
                         hidden_activation=actor_hidden_activation, output_activation=actor_output_activation)
        pi_targ = act_limit * actor_targ(x2_ph)
        if without_target_policy_smoothing:
            a2 = pi_targ
        else:
            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
            epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = tf.clip_by_value(a2, -act_limit, act_limit)

        critic1_targ = MLP(layer_sizes=critic_hidden_sizes + [1],
                           hidden_activation=critic_hidden_activation, output_activation=critic_output_activation)
        critic2_targ = MLP(layer_sizes=critic_hidden_sizes + [1],
                           hidden_activation=critic_hidden_activation, output_activation=critic_output_activation)
        # Target Q-values, using action from target policy
        q1_targ = tf.squeeze(critic1_targ(tf.concat([x2_ph, a2], axis=-1)), axis=1)
        q2_targ = tf.squeeze(critic2_targ(tf.concat([x2_ph, a2], axis=-1)), axis=1)


    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Bellman backup for Q functions, using Clipped Double-Q targets
    # TD3 losses
    min_q_targ = tf.minimum(q1_targ, q2_targ)
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*min_q_targ)

    # MSE
    q1_loss = 0.5 * tf.reduce_mean((backup-q1) ** 2)
    q2_loss = 0.5 * tf.reduce_mean((backup-q2) ** 2)
    pi_loss = -tf.reduce_mean(q1_pi)
    q_loss = q1_loss + q2_loss

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=actor.variables)
    train_q_op = q_optimizer.minimize(q_loss, var_list=critic1.variables+critic2.variables)
    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(actor.variables + critic1.variables + critic2.variables,
                                                        actor_targ.variables + critic1_targ.variables + critic2_targ.variables)])
    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(actor.variables + critic1.variables + critic2.variables,
                                                        actor_targ.variables + critic1_targ.variables + critic2_targ.variables)])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    # logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q1': q1, 'q2': q2})

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

        # Store experience to replay buffer
        replay_buffer.store(o, a, reward_scale*r, o2, d)

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
            # Save actor-critic model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                model_save_dir = os.path.join(logger.output_dir, 'checkpoints')
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                actor.save_weights(os.path.join(model_save_dir, 'epoch{}_actor'.format(epoch)))
                critic1.save_weights(os.path.join(model_save_dir, 'epoch{}_critic1'.format(epoch)))
                critic2.save_weights(os.path.join(model_save_dir, 'epoch{}_critic2'.format(epoch)))

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
    parser.add_argument('--actor_hidden_layers', nargs='+', type=int, default=[300, 300])
    parser.add_argument('--critic_hidden_layers', nargs='+', type=int, default=[300, 300])
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--without_delay_train', action='store_true')
    parser.add_argument('--without_target_policy_smoothing', action='store_true')
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--reward_scale', type=float, default=1)
    parser.add_argument('--act_noise', type=float, default=0.1)
    parser.add_argument("--data_dir", type=str, default='spinup_data')
    args = parser.parse_args()

    # Set log data saving directory
    from spinup.utils.run_utils import setup_logger_kwargs

    data_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))),
                        args.data_dir)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)
    
    td3(env_name=args.env,
        actor_hidden_layers=args.actor_hidden_layers, critic_hidden_layers=args.critic_hidden_layers,
        reward_scale=args.reward_scale,
        start_steps=args.start_steps,
        replay_size=args.replay_size,
        without_delay_train=args.without_delay_train,
        without_target_policy_smoothing=args.without_target_policy_smoothing,
        gamma=args.gamma, seed=args.seed, epochs=args.epochs, save_freq=args.save_freq,
        act_noise=args.act_noise,
        logger_kwargs=logger_kwargs)
