import os.path as osp
import numpy as np
import tensorflow as tf
import roboschool
import pybulletgym
import gym
import spinup.algos.ddpg_n_step.modified_envs
import time
from spinup.algos.ddpg_multihead_n_step import core
from spinup.algos.ddpg_multihead_n_step.core import get_vars, MLP, MultiHeadMLP
from spinup.utils.logx import EpochLogger

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
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
        # Set all done after the fist met one to 1
        done_index = np.asarray(np.where(batch_done==1))
        for d_i in range(done_index.shape[1]):
            x, y=done_index[:, d_i]
            batch_done[x, y:] = 1
        batch_done = np.hstack((np.zeros((batch_size, 1)), batch_done))
        return dict(obs1=batch_obs1[:,0,:],
                    obs2=batch_obs2[:,-1,:],
                    acts=batch_acts[:,0,:],
                    rews=batch_rews,
                    done=batch_done)

    def sample_batch_multihead_n_step(self, batch_size=32,
                                      n_step_end=1):
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
        idxs = np.random.randint(0, self.size - (n_step_end - 1), size=batch_size)
        batch_obs1 = np.zeros([batch_size, n_step_end, self.obs_dim])
        batch_obs2 = np.zeros([batch_size, n_step_end, self.obs_dim])
        batch_acts = np.zeros([batch_size, n_step_end, self.act_dim])
        batch_rews = np.zeros([batch_size, n_step_end])
        batch_done = np.zeros([batch_size, n_step_end])
        for i in range(n_step_end):
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
        # TODO: obs2
        # import pdb; pdb.set_trace()
        return dict(obs1=batch_obs1[:, 0, :],
                    obs2=batch_obs2[:, :n_step_end, :],
                    acts=batch_acts[:, 0, :],
                    rews=batch_rews,
                    done=batch_done)

"""

Multi-head n-step Deep Deterministic Policy Gradient (DDPG)

"""
def ddpg_multihead_n_step(env_name,
                          actor_hidden_layers=[300, 300],
                          critic_shared_hidden_layers=[300],
                          critic_separated_head_hidden_layers=[300],
                          seed=0, dropout_rate = 0,
                          steps_per_epoch=5000, epochs=100, replay_size=int(1e6),
                          reward_scale = 1,
                          multi_head_multi_step_size = [1, 2, 3, 4, 5],
                          actor_omit_top_k_Q = 2, actor_omit_low_k_Q = 1,
                          critic_omit_top_k_Q = 2, critic_omit_low_k_Q = 1,
                          multihead_q_std_penalty = 0.2,
                          separate_action_and_prediction = False,
                          multi_head_bootstrapping = False,
                          target_policy_smoothing=True, target_noise = 0.2, noise_clip = 0.5,
                          random_n_step=False, random_n_step_low=1, random_n_step_high=5,
                          gamma=0.99, without_delay_train=False, obs_noise_scale=0,
                          nonstationary_env=False,
                          gravity_change_pattern = 'gravity_averagely_equal',
                          gravity_cycle = 1000, gravity_base = -9.81,
                          polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
                          act_noise=0.1, random_action_baseline=False,
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

    env, test_env = gym.make(env_name), gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture

    # Inputs to computation graph
    multi_head_size = len(multi_head_multi_step_size)

    x_ph = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim))
    a_ph = tf.placeholder(dtype=tf.float32, shape=(None, act_dim))
    # TODO: use different mini-batch
    x2_ph = tf.placeholder(dtype=tf.float32, shape=(None, max(multi_head_multi_step_size), obs_dim))
    r_ph = tf.placeholder(dtype=tf.float32, shape=(None, None))
    d_ph = tf.placeholder(dtype=tf.float32, shape=(None, None))
    n_step_ph = tf.placeholder(dtype=tf.float32, shape=())

    actor_hidden_sizes = actor_hidden_layers
    actor_hidden_activation = tf.keras.activations.relu
    actor_output_activation = tf.keras.activations.tanh
    critic_shared_hidden_sizes = critic_shared_hidden_layers
    critic_head_hidden_sizes = critic_separated_head_hidden_layers
    critic_hidden_activation = tf.keras.activations.relu
    critic_output_activation = tf.keras.activations.linear

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        actor = MLP(layer_sizes=actor_hidden_sizes+[act_dim],
                    hidden_activation=actor_hidden_activation, output_activation=actor_output_activation)
        multihead_critic = MultiHeadMLP(shared_hidden_layer_sizes=critic_shared_hidden_sizes,
                                        multi_head_layer_sizes=[critic_head_hidden_sizes+[1] for i in range(multi_head_size)],
                                        hidden_activation=critic_hidden_activation,
                                        output_activation=critic_output_activation)
        # Set training=False to ignore dropout masks
        pi = act_limit * actor(x_ph, training=False)
        multihead_q = [tf.squeeze(head_out, axis=1) for head_out in multihead_critic(tf.concat([x_ph,a_ph], axis=-1))]
        multihead_q_pi = [tf.squeeze(head_out, axis=1) for head_out in multihead_critic(tf.concat([x_ph, pi], axis=-1))]

    # Target networks
    with tf.variable_scope('target'):
        # Note that the action placeholder going to actor_critic here is 
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        actor_targ = MLP(layer_sizes=actor_hidden_sizes+[act_dim],
                         hidden_activation=actor_hidden_activation, output_activation=actor_output_activation)
        multihead_critic_targ = MultiHeadMLP(shared_hidden_layer_sizes=critic_shared_hidden_sizes,
                                             multi_head_layer_sizes=[critic_head_hidden_sizes+[1] for i in range(multi_head_size)],
                                             hidden_activation=critic_hidden_activation,
                                             output_activation=critic_output_activation)

        # Set training=False to ignore dropout for backup target value
        # Crucial: feed target networks with different next n-step observation
        multihead_q_pi_targ = []
        # for head_i in range(multi_head_size):
        for h_i, n_step in enumerate(multi_head_multi_step_size):
            print('Head-{}: {}-step'.format(h_i, n_step))
            head_x2_ph = tf.squeeze(tf.slice(x2_ph, [0, n_step-1,0], [batch_size, 1, obs_dim]), axis=1)

            _ = actor_targ(head_x2_ph) # just for copy parameter
            if separate_action_and_prediction:
                head_pi_targ = act_limit * actor(head_x2_ph)
            else:
                head_pi_targ = act_limit * actor_targ(head_x2_ph)

            if target_policy_smoothing:
                # Target policy smoothing, by adding clipped noise to target actions
                epsilon = tf.random_normal(tf.shape(head_pi_targ), stddev=target_noise)
                epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
                head_pi_targ = head_pi_targ + epsilon
                head_pi_targ = tf.clip_by_value(head_pi_targ, -act_limit, act_limit)

            # TODO: test multi-head bootstrapping with StdQPenalty
            if multi_head_bootstrapping:
                # all heads calculate n-step bootstrapping,
                #  omit overestimation and underestimation of n-step bootstrapped Q
                after_omit_overestimation = tf.math.top_k(
                    -tf.squeeze(tf.stack(multihead_critic_targ(tf.concat([head_x2_ph, head_pi_targ], axis=-1)), axis=2),
                                axis=1), multi_head_size - critic_omit_top_k_Q)[0]
                after_omit_underestimation = tf.math.top_k(-after_omit_overestimation,
                                                           multi_head_size - critic_omit_top_k_Q - critic_omit_low_k_Q)[0]
                multihead_q_pi_targ.append(tf.reduce_mean(after_omit_underestimation, axis=1))
            else:
                multihead_q_pi_targ.append(
                    tf.squeeze(multihead_critic_targ(tf.concat([head_x2_ph, head_pi_targ], axis=-1))[h_i], axis=1))


    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q function
    multihead_q_loss_list = []
    multihead_q_pi_loss_list = []
    multihead_backup_list = []
    for h_i, n_step in enumerate(multi_head_multi_step_size):
        head_q = multihead_q[h_i]
        head_q_pi_targ = multihead_q_pi_targ[h_i]
        head_q_pi = multihead_q_pi[h_i]

        head_backup = tf.stop_gradient(tf.reduce_sum(tf.multiply(tf.pow(gamma, tf.range(0, n_step, dtype=tf.float32))
                                                                 * (1 - tf.slice(d_ph, [0, 0], [batch_size, n_step])),
                                                                 tf.slice(r_ph, [0, 0], [batch_size, n_step])), axis=1)
                                       + gamma ** n_step * (1 - tf.reshape(tf.slice(d_ph, [0, n_step], [batch_size, 1]), [-1])) * head_q_pi_targ)
        multihead_backup_list.append(head_backup)
        multihead_q_loss_list.append(tf.reduce_mean((head_q-head_backup)**2))
        multihead_q_pi_loss_list.append(-tf.reduce_mean(head_q_pi))


    # DDPG losses
    # 1. pi loss
    all_q_pi = tf.stack(multihead_q_pi, axis=1)

    # pi_loss = tf.reduce_mean(multihead_q_pi_loss_list) # Works, but not stable

    # Works good, need to test generalization
    # pi_loss = tf.reduce_mean(tf.math.top_k(-all_q_pi, multi_head_size - omit_top_k_Q)[0])

    after_omit_overestimation_for_actor = tf.math.top_k(-all_q_pi, multi_head_size - actor_omit_top_k_Q)[0]
    after_omit_underestimation_for_actor = tf.math.top_k(-after_omit_overestimation_for_actor, multi_head_size - actor_omit_top_k_Q - actor_omit_low_k_Q)[0]
    pi_loss = tf.reduce_mean(tf.reduce_mean(-after_omit_underestimation_for_actor, axis=1))

    # # TODO：test, seems not work
    # pi_loss = tf.reduce_mean(tf.reduce_mean(tf.math.top_k(-all_q_pi, multi_head_size - actor_omit_top_k_Q)[0], axis=1) +
    #                          multihead_q_std_penalty * tf.math.reduce_variance(all_q_pi, axis=1))

    # # import pdb; pdb.set_trace()
    # pi_loss = tf.reduce_sum(tf.reduce_mean(tf.math.top_k(-tf.stack(multihead_q_pi, axis=1), multi_head_size - omit_top_k_Q)[0], axis=0))

    # pi_loss = tf.reduce_mean(-multihead_q_pi[0]) # Too slow

    # # slow
    # pi_loss = tf.reduce_mean(tf.reduce_sum(tf.math.top_k(-all_q_pi,
    #                                                       multi_head_size - actor_omit_top_k_Q)[0], axis=1))

    # 2. q loss
    all_q = tf.stack(multihead_q, axis=1)
    all_q_backup = tf.stack(multihead_backup_list, axis=1)
    # q_loss = tf.reduce_sum((all_q - all_q_backup) ** 2)
    # q_loss = tf.reduce_mean((all_q - all_q_backup) ** 2) # (Currently the best) Works good for Swimmer-s0
    # q_loss = tf.reduce_mean(multihead_q_loss_list)     # works
    q_loss = tf.reduce_sum(multihead_q_loss_list)      # Works good for Swimmer-s3

    # currently the best, and the policy has approximately monotonic improvement
    # TODO: multihead_q_std_penalty should be dynamically changed
    # q_loss = tf.reduce_mean(tf.reduce_mean((all_q - all_q_backup)**2, axis=1) +
    #                         multihead_q_std_penalty * tf.math.reduce_std(all_q, axis=1))

    # # variance penalty is better than standard deviation penalty
    # q_loss = tf.reduce_mean(tf.reduce_mean((all_q - all_q_backup) ** 2, axis=1) +
    #                         multihead_q_std_penalty * tf.math.reduce_variance(all_q, axis=1))

    # # TODO： test reduce_sum and reduce_var
    # q_loss = tf.reduce_mean(tf.reduce_sum((all_q - all_q_backup) ** 2, axis=1) +
    #                         multihead_q_std_penalty * tf.math.reduce_variance(all_q, axis=1))



    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=actor.variables)
    train_q_op = q_optimizer.minimize(q_loss, var_list=multihead_critic.variables)

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(actor.variables+multihead_critic.variables,
                                                        actor_targ.variables+multihead_critic_targ.variables)])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(actor.variables+multihead_critic.variables,
                                                        actor_targ.variables+multihead_critic_targ.variables)])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # # Setup model saving
    # logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q': q})

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
    # # TODO: delete env.render()
    # env.render()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps and not random_action_baseline:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # env.render()
        # Manipulate environment
        change_scale = 1/8
        if nonstationary_env == True:
            if gravity_change_pattern == 'gravity_averagely_equal':
                # gravity = gravity_base * 1 / 2 * (np.cos(2 * np.pi / gravity_cycle * t) + 1) + gravity_base / 2
                gravity = gravity_base + np.abs(gravity_base) * change_scale * np.sin(2 * np.pi / gravity_cycle * t)
            elif gravity_change_pattern == 'gravity_averagely_easier':
                # gravity = gravity_base * 1 / 2 * (np.cos(2 * np.pi / gravity_cycle * t) + 1)
                gravity = gravity_base * change_scale * (np.cos(2 * np.pi / gravity_cycle * t)) + gravity_base * ( 1 - change_scale)
            elif gravity_change_pattern == 'gravity_averagely_harder':
                # gravity = gravity_base * 1 / 2 * (-np.cos(2 * np.pi / gravity_cycle * t) + 1) + gravity_base
                gravity = gravity_base * change_scale * (-np.cos(2 * np.pi / gravity_cycle * t)) + gravity_base * (
                            1 + change_scale)
            else:
                pass

            if 'PyBulletEnv' in env_name:
                env.env._p.setGravity(0, 0, gravity)
            elif 'Roboschool' in env_name:
                pass
            else:
                env.model.opt.gravity[2] = gravity
        # Step the env
        o2, r, d, _ = env.step(a)
        # Add observation noise
        o2 += obs_noise_scale * np.random.randn(obs_dim)
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

        if t > batch_size and without_delay_train:
            if random_n_step:
                n_step = np.random.randint(random_n_step_low, random_n_step_high + 1, 1)[0]

            batch = replay_buffer.sample_batch_multihead_n_step(batch_size, n_step_end=max(multi_head_multi_step_size))
            feed_dict = {x_ph: batch['obs1'],
                         x2_ph: batch['obs2'],
                         a_ph: batch['acts'],
                         r_ph: batch['rews'],
                         d_ph: batch['done']
                         }
            # import pdb; pdb.set_trace()
            # Q-learning update
            outs = sess.run([multihead_q_loss_list, multihead_q, train_q_op], feed_dict)
            logger.store(**{'LossQ{}_{}Step'.format(h_i, multi_head_multi_step_size[h_i]): outs[0][h_i] for h_i in
                            range(multi_head_size)})
            logger.store(**{'QVals{}_{}Step'.format(h_i, multi_head_multi_step_size[h_i]): outs[1][h_i] for h_i in
                            range(multi_head_size)})

            # Policy update
            outs = sess.run([multihead_q_pi_loss_list, train_pi_op, target_update], feed_dict)
            logger.store(**{'LossPi{}_{}Step'.format(h_i, multi_head_multi_step_size[h_i]): outs[0][h_i] for h_i in
                            range(multi_head_size)})


        if d or (ep_len == max_ep_len):
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            if not without_delay_train:
                for _ in range(ep_len):
                    if random_n_step:
                        n_step = np.random.randint(random_n_step_low, random_n_step_high+1, 1)[0]
                    batch = replay_buffer.sample_batch_multihead_n_step(batch_size, n_step_end=max(multi_head_multi_step_size))
                    feed_dict = {x_ph: batch['obs1'],
                                 x2_ph: batch['obs2'],
                                 a_ph: batch['acts'],
                                 r_ph: batch['rews'],
                                 d_ph: batch['done']
                                }

                    # Q-learning update
                    outs = sess.run([multihead_q_loss_list, multihead_q, train_q_op], feed_dict)
                    logger.store(**{'LossQ{}_{}Step'.format(h_i, multi_head_multi_step_size[h_i]): outs[0][h_i] for h_i in
                                    range(multi_head_size)})
                    logger.store(**{'QVals{}_{}Step'.format(h_i, multi_head_multi_step_size[h_i]): outs[1][h_i] for h_i in
                                    range(multi_head_size)})

                    # Policy update
                    outs = sess.run([multihead_q_pi_loss_list, train_pi_op, target_update], feed_dict)
                    logger.store(**{'LossPi{}_{}Step'.format(h_i, multi_head_multi_step_size[h_i]): outs[0][h_i] for h_i in
                                    range(multi_head_size)})

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
            for h_i in range(multi_head_size):
                logger.log_tabular('QVals{}_{}Step'.format(h_i, multi_head_multi_step_size[h_i]), with_min_and_max=True)
            for h_i in range(multi_head_size):
                logger.log_tabular('LossPi{}_{}Step'.format(h_i, multi_head_multi_step_size[h_i]), average_only=True)
            for h_i in range(multi_head_size):
                logger.log_tabular('LossQ{}_{}Step'.format(h_i, multi_head_multi_step_size[h_i]), average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--actor_hidden_layers', nargs='+', type=int, default=[300, 300])
    parser.add_argument('--critic_shared_hidden_layers', nargs='+', type=int, default=[300])
    parser.add_argument('--critic_separated_head_hidden_layers', nargs='+', type=int, default=[300])
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='ddpg_multihead_n_step')
    parser.add_argument('--reward_scale', type=float, default=1)

    parser.add_argument('--multi_head_multi_step_size', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--actor_omit_top_k_Q', type=int, default=0)
    parser.add_argument('--actor_omit_low_k_Q', type=int, default=0)
    parser.add_argument('--multihead_q_std_penalty', type=float, default=0)
    parser.add_argument('--separate_action_and_prediction', action='store_true')
    parser.add_argument('--multi_head_bootstrapping', action='store_true')
    parser.add_argument('--critic_omit_top_k_Q', type=int, default=0)
    parser.add_argument('--critic_omit_low_k_Q', type=int, default=0)

    parser.add_argument('--act_noise', type=float, default=0.1)

    parser.add_argument('--target_policy_smoothing', action='store_true')
    parser.add_argument('--target_noise', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)

    parser.add_argument('--random_n_step', action='store_true')
    parser.add_argument('--random_n_step_low', type=int, default=1)
    parser.add_argument('--random_n_step_high', type=int, default=5)
    parser.add_argument('--without_delay_train', action='store_true')
    parser.add_argument('--random_action_baseline', action='store_true')
    parser.add_argument('--obs_noise_scale', type=float, default=0)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--nonstationary_env', action='store_true')
    parser.add_argument('--gravity_change_pattern', type=str, default='gravity_averagely_equal')
    parser.add_argument('--gravity_cycle', type=int, default=1000)
    parser.add_argument('--gravity_base', type=float, default=-9.81)
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--hardcopy_target_nn', action="store_true", help='Target network update method: hard copy')
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument("--exploration-strategy", type=str, choices=["action_noise", "epsilon_greedy"],
                        default='epsilon_greedy', help='action_noise or epsilon_greedy')
    parser.add_argument("--epsilon-max", type=float, default=1.0, help='maximum of epsilon')
    parser.add_argument("--epsilon-min", type=float, default=.01, help='minimum of epsilon')
    parser.add_argument("--epsilon-decay", type=float, default=.001, help='epsilon decay')

    parser.add_argument("--data_dir", type=str, default='spinup_data')

    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, args.data_dir, datestamp=True)
    # Set log data saving directory
    from spinup.utils.run_utils import setup_logger_kwargs
    data_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))), args.data_dir)

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)

    if args.hardcopy_target_nn:
        polyak = 0

    ddpg_multihead_n_step(env_name=args.env, dropout_rate=args.dropout_rate,
                          actor_hidden_layers=args.actor_hidden_layers,
                          critic_shared_hidden_layers=args.critic_shared_hidden_layers,
                          critic_separated_head_hidden_layers=args.critic_separated_head_hidden_layers,
                          reward_scale=args.reward_scale,
                          multi_head_multi_step_size=args.multi_head_multi_step_size,
                          actor_omit_top_k_Q=args.actor_omit_top_k_Q, actor_omit_low_k_Q=args.actor_omit_low_k_Q,
                          critic_omit_top_k_Q=args.critic_omit_top_k_Q, critic_omit_low_k_Q=args.critic_omit_low_k_Q,
                          multihead_q_std_penalty=args.multihead_q_std_penalty,
                          separate_action_and_prediction=args.separate_action_and_prediction,
                          multi_head_bootstrapping=args.multi_head_bootstrapping,
                          act_noise=args.act_noise,
                          target_policy_smoothing=args.target_policy_smoothing,
                          target_noise=args.target_noise, noise_clip=args.noise_clip,
                          random_n_step=args.random_n_step,
                          random_n_step_low=args.random_n_step_low, random_n_step_high=args.random_n_step_high,
                          replay_size=args.replay_size,
                          batch_size=args.batch_size,
                          without_delay_train=args.without_delay_train, start_steps=args.start_steps,
                          random_action_baseline=args.random_action_baseline,
                          obs_noise_scale=args.obs_noise_scale,
                          nonstationary_env=args.nonstationary_env,
                          gravity_change_pattern=args.gravity_change_pattern,
                          gravity_cycle = args.gravity_cycle, gravity_base = args.gravity_base,
                          gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                          logger_kwargs=logger_kwargs)