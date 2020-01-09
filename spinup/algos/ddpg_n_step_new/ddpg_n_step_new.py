import numpy as np
import pandas as pd
import tensorflow as tf


import pybulletgym
import gym
import time
from spinup.algos.ddpg_n_step_new import core
from spinup.algos.ddpg_n_step_new.core import get_vars, MLP
from spinup.utils.logx import EpochLogger
import os.path as osp

import multiprocessing
from multiprocessing import Process, Pool

import tkinter as tk
from tkinter import simpledialog
ROOT = tk.Tk()
ROOT.withdraw()

from collections import deque

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_sim_state_buf = []
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, next_obs_sim_state):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.obs2_sim_state_buf.append(next_obs_sim_state)
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
        idxs = np.random.randint(0, self.size-(n_step-1), size=batch_size)
        batch_obs1 = np.zeros([batch_size, n_step, self.obs_dim])
        batch_obs2_sim_state = []
        batch_obs_restore = []
        batch_obs_restore_sim_state = []
        batch_obs2 = np.zeros([batch_size, n_step, self.obs_dim])
        batch_acts = np.zeros([batch_size, n_step, self.act_dim])
        batch_rews = np.zeros([batch_size, n_step])
        batch_done = np.zeros([batch_size, n_step])
        for i in range(n_step):
            batch_obs1[:, i, :] = self.obs1_buf[idxs+i]
            batch_obs2[:, i, :] = self.obs2_buf[idxs + i]
            batch_acts[:, i, :] = self.acts_buf[idxs + i]
            batch_rews[:, i] = self.rews_buf[idxs + i]
            batch_done[:, i] = self.done_buf[idxs + i]

        # Simulation state corresponds to obs2 for restoring
        for i in idxs:
            batch_obs_restore_sim_state.append([self.obs2_sim_state_buf[i+s_i] for s_i in range(n_step)])

        # Set all done after the fist met one to 1
        done_index = np.asarray(np.where(batch_done==1))
        for d_i in range(done_index.shape[1]):
            x, y=done_index[:, d_i]
            batch_done[x, y:] = 1
        batch_done = np.hstack((np.zeros((batch_size, 1)), batch_done))
        return dict(obs1=batch_obs1[:,0,:],
                    obs2=batch_obs2[:,:,:],
                    obs2_sim_state=batch_obs_restore_sim_state,
                    acts=batch_acts[:,0,:],
                    rews=batch_rews,
                    done=batch_done)


def pybulletenv_get_state(env):
    body_num = env.env._p.getNumBodies()
    # body_info = [env.env._p.getBodyInfo(body_i) for body_i in range(body_num)]
    floor_id, robot_id = 0, 1

    robot_base_pos_ori = env.env._p.getBasePositionAndOrientation(robot_id)
    robot_base_vel = env.env._p.getBaseVelocity(robot_id)

    joint_num = env.env._p.getNumJoints(robot_id)
    joint_state = []
    for joint_i in range(joint_num):
        joint_state.append(env.env._p.getJointState(robot_id, joint_i))

    state = {'body_num': body_num,
             'robot_base_pos_ori': robot_base_pos_ori, 'robot_base_vel': robot_base_vel,
             'joint_num': joint_num, 'joint_state': joint_state}
    return state


def pybulletenv_set_state(env, state):
    body_num = env.env._p.getNumBodies()
    floor_id, robot_id = 0, 1
    joint_num = env.env._p.getNumJoints(robot_id)
    if body_num != state['body_num'] and joint_num != state['body_num']:
        print('Set state error.')
    # restore state
    env.env._p.resetBasePositionAndOrientation(robot_id,
                                               state['robot_base_pos_ori'][0],
                                               state['robot_base_pos_ori'][1])
    env.env._p.resetBaseVelocity(robot_id, state['robot_base_vel'])
    for j_i, j_s in enumerate(state['joint_state']):
        env.env._p.resetJointState(robot_id, j_i, j_s[0], j_s[1])
    return env


def env_expansion_process(actor_weights, env_name, restore_obs, restore_obs_sim, done, gamma):
    import tensorflow as tf
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    # Create policy
    x_ph = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim))
    hidden_sizes = [300, 300]
    actor_hidden_activation = tf.keras.activations.relu
    actor_output_activation = tf.keras.activations.tanh
    actor = MLP(layer_sizes=hidden_sizes + [act_dim],
                hidden_activation=actor_hidden_activation, output_activation=actor_output_activation)
    pi = act_limit * actor(x_ph)
    # Load policy weights
    actor.set_weights(actor_weights)

    env.reset()
    # restore simulator state
    if 'PyBulletEnv' in env_name:
        pybulletenv_set_state(env, restore_obs_sim)
    elif 'Roboschool' in env_name:
        print('Roboschool is used.')
    else:
        env.sim.set_state(restore_obs_sim)  # MuJuco

    # expand
    o2 = restore_obs
    exp_dis_r = 0
    step = 0
    while not done:
        o2, r, done, _ = env.step(sess.run(pi, feed_dict={x_ph:o2.reshape(1,-1)})[0])
        exp_dis_r += (gamma**step)*r
        step += 1
    return exp_dis_r

"""

Deep Deterministic Policy Gradient (DDPG)

"""
def ddpg_n_step_new(env_name, render_env=False, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
                    steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
                    n_step=1, backup_method='mixed_n_step',
                    without_delay_train=False,
                    polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
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

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    if 'PyBulletEnv' in env_name:
        print('PyBulletEnv is used.')
    elif 'Roboschool' in env_name:
        print('Roboschool is used.')
    else:
        print('MuJoCo is used.')

    env, test_env = gym.make(env_name), gym.make(env_name)
    batch_envs = [gym.make(env_name) for tmp_i in range(batch_size)]
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim))
    a_ph = tf.placeholder(dtype=tf.float32, shape=(None, act_dim))
    x2_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, obs_dim))
    r_ph = tf.placeholder(dtype=tf.float32, shape=(None, None))
    d_ph = tf.placeholder(dtype=tf.float32, shape=(None, None))
    n_step_ph = tf.placeholder(dtype=tf.float32, shape=())

    hidden_sizes = list(ac_kwargs['hidden_sizes'])
    actor_hidden_activation = tf.keras.activations.relu
    actor_output_activation = tf.keras.activations.tanh
    critic_hidden_activation = tf.keras.activations.relu
    critic_output_activation = tf.keras.activations.linear

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        actor = MLP(layer_sizes=hidden_sizes + [act_dim],
                    hidden_activation=actor_hidden_activation, output_activation=actor_output_activation)
        critic = MLP(layer_sizes=hidden_sizes + [1],
                     hidden_activation=critic_hidden_activation, output_activation=critic_output_activation)
        pi = act_limit * actor(x_ph)
        q = tf.squeeze(critic(tf.concat([x_ph, a_ph], axis=-1)), axis=1)
        q_pi = tf.squeeze(critic(tf.concat([x_ph, pi], axis=-1)), axis=1)
    
    # Target networks
    with tf.variable_scope('target'):
        # Note that the action placeholder going to actor_critic here is 
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        actor_targ = MLP(layer_sizes=hidden_sizes + [act_dim],
                         hidden_activation=actor_hidden_activation, output_activation=actor_output_activation)
        critic_targ = MLP(layer_sizes=hidden_sizes + [1],
                          hidden_activation=critic_hidden_activation, output_activation=critic_output_activation)
        n_step_bootstrapped_q = []
        for n_step_i in range(n_step):
            next_obs_tmp = tf.reshape(tf.slice(x2_ph, [0, n_step_i, 0], [batch_size, 1, obs_dim]), [batch_size, obs_dim])
            pi_targ_tmp = act_limit * actor_targ(next_obs_tmp)
            q_pi_targ = tf.squeeze(critic_targ(tf.concat([next_obs_tmp, pi_targ_tmp], axis=-1)), axis=1)
            n_step_bootstrapped_q.append(q_pi_targ)
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Bellman backup for Q function
    backups = []
    for n_step_i in range(1, n_step+1):
        backup_tmp = tf.stop_gradient(
            tf.reduce_sum(tf.multiply(tf.pow(gamma, tf.range(0, n_step_i, dtype=tf.float32))
                                      * (1 - tf.slice(d_ph, [0, 0], [batch_size, n_step_i])),
                                      tf.slice(r_ph, [0, 0], [batch_size, n_step_i])), axis=1)
            + gamma ** n_step_i * (1 - tf.reshape(tf.slice(d_ph, [0, n_step_i], [batch_size, 1]), [-1])) *
            n_step_bootstrapped_q[n_step_i - 1])
        backups.append(backup_tmp)

    # backup_n_step = backups[-1]
    backup_avg_n_step = tf.stop_gradient(tf.reduce_mean(tf.stack(backups, axis=1), axis=1))
    backup_min_n_step = tf.stop_gradient(tf.reduce_min(tf.stack(backups, axis=1), axis=1))

    if backup_method == 'avg_n_step':
        backup_n_step = backup_avg_n_step
    elif backup_method == '1_step':
        backup_n_step = backups[0]
    elif backup_method == '2_step':
        backup_n_step = backups[1]
    elif backup_method == '3_step':
        backup_n_step = backups[2]
    elif backup_method == '4_step':
        backup_n_step = backups[3]
    elif backup_method == '5_step':
        backup_n_step = backups[4]
    elif backup_method == 'min_n_step':
        backup_n_step = backup_min_n_step
    else:
        pass

    # DDPG losses
    pi_loss = -tf.reduce_mean(q_pi)
    q_loss = tf.reduce_mean((q-backup_n_step)**2)

    # TOdo: log gradient to see how it change with overestimation problem, and we can clip gradient if overestimation
    #   happended

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    train_pi_op = pi_optimizer.minimize(loss=pi_loss, var_list=actor.variables)

    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_q_op = q_optimizer.minimize(loss=q_loss, var_list=critic.variables)

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(actor.variables+critic.variables,
                                                        actor_targ.variables+critic_targ.variables)])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(actor.variables + critic.variables,
                                                      actor_targ.variables + critic_targ.variables)])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

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
    batch_env_reset = [batch_env.reset() for batch_env in batch_envs]
    old_ep_ret = 0
    max_ep_ret = 0
    ep_ret_history = deque(maxlen=5)

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
        if render_env:
            env.render()
        o2, r, d, _ = env.step(a)

        if 'PyBulletEnv' in env_name:
            o2_sim_state = pybulletenv_get_state(env)
        elif 'Roboschool' in env_name:
            print('Roboschool is used.')
        else:
            o2_sim_state = env.sim.get_state()  # MuJuco

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, o2_sim_state)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if without_delay_train:
            # batch = replay_buffer.sample_batch(batch_size)
            batch = replay_buffer.sample_batch_n_step(batch_size, n_step=n_step)
            feed_dict = {x_ph: batch['obs1'],
                         x2_ph: batch['obs2'],
                         a_ph: batch['acts'],
                         r_ph: batch['rews'],
                         d_ph: batch['done'],
                         n_step_ph: n_step
                         }

            # Q-learning update
            outs = sess.run([q_loss, q, train_q_op], feed_dict)
            logger.store(LossQ=outs[0], QVals=outs[1])

            # # Policy update
            # if t % policy_delay == 0:
            # Delayed policy update
            outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
            logger.store(LossPi=outs[0])

        if d or (ep_len == max_ep_len):
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
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
                                 n_step_ph: n_step}


                    """
                    ###############################################
                    """
                    start_time_on_policy = time.time()




                    # reset simulator to the state (s') after taking (s, a, r)
                    # for i, batch_env_i in enumerate(batch_envs):
                    #     if 'PyBulletEnv' in env_name:
                    #         pybulletenv_set_state(batch_env_i, restore_obs_sim[i])
                    #     elif 'Roboschool' in env_name:
                    #         print('Roboschool is used.')
                    #     else:
                    #         batch_env_i.sim.set_state(restore_obs_sim[i])  # MuJuco

                    # # n-step online expansion + bootstrapped value
                    # n_step_next_obs = np.zeros([batch_size, n_step, obs_dim])
                    # obs_restore = batch['batch_obs_restore']
                    # n_step_batch_r = batch['rews'][:,0]
                    # n_step_batch_d = batch['done'][:,0:2]
                    #
                    # # Todo: expand until the end of an episod
                    # # for i in range(1000):
                    #
                    #
                    # for step_i in range(1, n_step):
                    #     n_step_next_obs[:, step_i-1, :] = obs_restore
                    #     action = sess.run(pi, feed_dict={x_ph:obs_restore})
                    #     obs_resotre_next = []
                    #     batch_r = []
                    #     batch_d = []
                    #     for i, batch_env_i in enumerate(batch_envs):
                    #         tmp_o2, tmp_r, tmp_d, _ = batch_env_i.step(action[i])
                    #         obs_resotre_next.append(tmp_o2)
                    #         batch_r.append(tmp_r)
                    #         batch_d.append(tmp_d)
                    #         if tmp_d:
                    #             tmp_o2 = batch_env_i.reset()
                    #             obs_resotre_next[-1] = tmp_o2
                    #     obs_restore = np.asarray(obs_resotre_next)
                    #     n_step_batch_r = np.append(n_step_batch_r, batch_r)
                    #     n_step_batch_d = np.append(n_step_batch_d, batch_d)
                    #
                    # n_step_batch_r = np.reshape(n_step_batch_r, (batch_size, n_step), order='F')
                    # n_step_batch_d = np.reshape(n_step_batch_d, (batch_size, n_step+1), order='F')
                    # n_step_next_obs[:, - 1, :] = obs_restore
                    #
                    # # Set all done after the fist met one to 1
                    # batch_done_index = np.asarray(np.where(n_step_batch_d == 1))
                    # for d_i in range(batch_done_index.shape[1]):
                    #     x, y = batch_done_index[:, d_i]
                    #     n_step_batch_d[x, y:] = 1
                    #
                    # bootstrapped_q = sess.run(n_step_bootstrapped_q, feed_dict={x2_ph: n_step_next_obs})
                    # bootstrapped_q = np.stack(bootstrapped_q, axis=1)
                    #
                    # n_step_backup_on_policy = np.zeros((batch_size, n_step))
                    # reward_discounted = np.zeros((batch_size, n_step))
                    # for n_i in range(n_step):
                    #     discount_n_i = gamma**n_i
                    #     reward_discounted[:, n_i] = discount_n_i * (1-n_step_batch_d[:, n_i]) * n_step_batch_r[:, n_i]
                    #     bootstrapped_n_i_discounted = gamma ** (n_i+1) * (1-n_step_batch_d[:, n_i+1]) * bootstrapped_q[:, n_i]
                    #     n_step_backup_on_policy[:, n_i] = np.sum(reward_discounted[:, :(n_i+1)], axis=1) + bootstrapped_n_i_discounted
                    #
                    # logger.store(**{'QBackupOnPolicy{}Step'.format(n_step_i+1):n_step_backup_on_policy[:,n_step_i] for n_step_i in range(n_step)})
                    # logger.store(QBackupOnPolicyMixedNStep=np.mean(n_step_backup_on_policy, axis=1))

                    """
                    ###############################################
                    """
                    # Q-learning update
                    outs = sess.run([q_loss, q, backups, backup_avg_n_step, backup_min_n_step, train_q_op], feed_dict)
                    logger.store(LossQ=outs[0], QVals=outs[1])
                    logger.store(
                        **{'QBackup{}Step'.format(n_step_i+1): outs[2][n_step_i] for n_step_i in range(n_step)})
                    logger.store(QBackupAvgNStep=outs[3], QBackupMinNStep=outs[4])

                    outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                    logger.store(LossPi=outs[0])

            # n-step offline + online expansion
            exp_batch_size = 50
            exp_batch = replay_buffer.sample_batch_n_step(exp_batch_size, n_step=n_step)
            exp_n_step = 0
            restore_obs = exp_batch['obs2'][:, exp_n_step, :]
            restore_obs_sim = pd.DataFrame(exp_batch['obs2_sim_state']).iloc[:, exp_n_step].tolist()
            restore_act = exp_batch['acts']
            done = exp_batch['done'][:, exp_n_step + 1]

            actor_weights = actor.get_weights()
            arguments = [(actor_weights, env_name, restore_obs[b_i], restore_obs_sim[b_i], done[b_i], gamma) for
                         b_i in range(exp_batch_size)]

            # multiprocessing.cpu_count()
            pool = Pool(processes=3)
            process_outs = pool.starmap(env_expansion_process, arguments)

            discounted_accumulated_reward = np.asarray(process_outs)
            dis_r = np.tile(np.asarray([gamma ** i for i in range(exp_n_step + 1)]).reshape([1, -1]),
                            [exp_batch_size, 1])
            ground_truth_q = dis_r * exp_batch['rews'][:, exp_n_step].reshape([exp_batch_size, -1]) + gamma ** (
                        exp_n_step + 1) * discounted_accumulated_reward
            predicted_q = sess.run(q, feed_dict={x_ph: exp_batch['obs1'], a_ph: exp_batch['acts']})
            logger.store(PredictedQ=predicted_q, GroundTruthQ=ground_truth_q)

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs-1):
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
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            for n_step_i in range(n_step):
                logger.log_tabular('QBackup{}Step'.format(n_step_i+1), with_min_and_max=True)
                # logger.log_tabular('QBackupOnPolicy{}Step'.format(n_step_i + 1), with_min_and_max=True)
            logger.log_tabular('QBackupAvgNStep', with_min_and_max=True)
            logger.log_tabular('QBackupMinNStep', with_min_and_max=True)
            logger.log_tabular('PredictedQ', with_min_and_max=True)
            logger.log_tabular('GroundTruthQ', with_min_and_max=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--render_env', action="store_true")
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--steps_per_epoch', type=int, default=5000)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--n_step', type=int, default=5)
    parser.add_argument('--backup_method', type=str,
                        choices=['avg_n_step', 'min_n_step', '1_step', '2_step', '3_step', '4_step', '5_step'],
                        default='avg_n_step')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--without_delay_train', action='store_true')
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--hardcopy_target_nn', action="store_true", help='Target network update method: hard copy')
    parser.add_argument('--act_noise',type=float, default=0.1)
    parser.add_argument("--exploration-strategy", type=str, choices=["action_noise", "epsilon_greedy"],
                        default='epsilon_greedy', help='action_noise or epsilon_greedy')
    parser.add_argument("--epsilon-max", type=float, default=1.0, help='maximum of epsilon')
    parser.add_argument("--epsilon-min", type=float, default=.01, help='minimum of epsilon')
    parser.add_argument("--epsilon-decay", type=float, default=.001, help='epsilon decay')

    parser.add_argument("--data_dir", type=str, default=None)

    args = parser.parse_args()

    # Set log data saving directory
    from spinup.utils.run_utils import setup_logger_kwargs
    data_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))),
                        'spinup_data')
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)

    # if args.hardcopy_target_nn:
    #     polyak = 0

    ddpg_n_step_new(env_name=args.env, render_env=args.render_env,
                    act_noise=args.act_noise,
                    actor_critic=core.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                    gamma=args.gamma, seed=args.seed,replay_size=args.replay_size,
                    n_step=args.n_step, backup_method=args.backup_method,
                    without_delay_train=args.without_delay_train,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    steps_per_epoch=args.steps_per_epoch, start_steps=args.start_steps,
                    logger_kwargs=logger_kwargs)
