"""
@auther: linghengmeng@yahoo.com

Inspired by:
https://github.com/Kyushik/Predictive-Uncertainty-Estimation-using-Deep-Ensemble
"""
import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.dedpg import core
from spinup.algos.dedpg.core import get_vars
from spinup.utils.logx import EpochLogger

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

Deep Ensemble Deterministic Policy Gradient (DEDPG)

"""
def dedpg(env_fn, actor_critic_ensemble=core.mlp_actor_critic_ensemble,
          ac_kwargs=dict(), seed=0,
          steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
          polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
          batch_size=100, ensemble_size=1, feed_ac_ensemble_with_same_batch=True,
          start_steps=10000,
          act_noise=0.1, max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
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

        ensemble_size (int): the size of ensemble i.e. the number of actor-critic

        feed_ac_ensemble_with_same_batch (bool): indicate if feed each actor-critic in ensemble
            with the same batch.

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
        ac_ensemble = actor_critic_ensemble(x_ph, a_ph, **ac_kwargs, ensemble_size=ensemble_size)
    
    # Target networks
    with tf.variable_scope('target'):
        # Note that the action placeholder going to actor_critic here is 
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        ac_ensemble_targ = actor_critic_ensemble(x2_ph, a_ph, **ac_kwargs, ensemble_size=ensemble_size)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    for ac_name in ac_ensemble.keys():
        ac_scope = ['main/{}/pi'.format(ac_name), 'main/{}/q'.format(ac_name), 'main/{}'.format(ac_name)]
        var_counts = tuple(core.count_vars(scope) for scope in ac_scope)
        print('\nNumber of parameters of {}: \t pi: {}, \t q: {}, \t total: {}\n'.format(ac_name, *var_counts))

    # TODO: should backup be a mean of all actor-critics' prediction or treat each actor-critic independently
    for ac_name in ac_ensemble.keys():
        # Bellman backup for Q function
        ac_ensemble[ac_name]['backup'] = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * ac_ensemble_targ[ac_name]['q_pi'])

    for ac_name in ac_ensemble.keys():
        # DEDPG losses
        ac_ensemble[ac_name]['pi_loss'] = -tf.reduce_mean(ac_ensemble[ac_name]['q_pi'])
        ac_ensemble[ac_name]['q_loss'] = tf.reduce_mean((ac_ensemble[ac_name]['q']-ac_ensemble[ac_name]['backup'])**2)

        # Separate train ops for pi, q
        ac_ensemble[ac_name]['pi_optimizer'] = tf.train.AdamOptimizer(learning_rate=pi_lr)
        ac_ensemble[ac_name]['q_optimizer'] = tf.train.AdamOptimizer(learning_rate=q_lr)
        ac_ensemble[ac_name]['train_pi_op'] = ac_ensemble[ac_name]['pi_optimizer'].minimize(ac_ensemble[ac_name]['pi_loss'],
                                                                                            var_list=get_vars('main/{}/pi'.format(ac_name)))
        ac_ensemble[ac_name]['train_q_op'] = ac_ensemble[ac_name]['q_optimizer'].minimize(ac_ensemble[ac_name]['q_loss'],
                                                                                          var_list=get_vars('main/{}/q'.format(ac_name)))

        # Polyak averaging for target variables
        ac_ensemble[ac_name]['target_update'] = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                                          for v_main, v_targ in zip(get_vars('main/{}'.format(ac_name)),
                                                                                    get_vars('target/{}'.format(ac_name)))])

        # Initializing targets to match main variables
        ac_ensemble[ac_name]['target_init'] = tf.group([tf.assign(v_targ, v_main)
                                                        for v_main, v_targ in zip(get_vars('main/{}'.format(ac_name)),
                                                                                  get_vars('target/{}'.format(ac_name)))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for ac_name in ac_ensemble.keys():
        sess.run(ac_ensemble[ac_name]['target_init'])
    # import pdb; pdb.set_trace()
    # Setup model saving
    # TODO:
    # logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q': q})

    def get_action(o, noise_scale):
        # TODO: Take action according to estimated uncertainty
        # take_mean_action = False
        # a = np.zeros([act_dim, len(ac_ensemble)])
        # for ac_i, ac_name in enumerate(ac_ensemble.keys()):
        #     a[:, ac_i] = sess.run(ac_ensemble[ac_name]['pi'], feed_dict={x_ph: o.reshape(1, -1)})[0]
        # a = np.mean(a, axis=1)  # mean of actions of ensemble
        # a += noise_scale * np.random.randn(act_dim)

        a_ensemble = np.zeros((len(ac_ensemble), act_dim))
        for ac_i, ac_name in enumerate(ac_ensemble.keys()):
            a_ensemble[ac_i] = sess.run(ac_ensemble[ac_name]['pi'], feed_dict={x_ph: o.reshape(1, -1)})[0]

        a_cov = np.cov(a_ensemble, rowvar=False)
        a_mean = np.mean(a_ensemble, axis=0)
        a_median = np.median(a_ensemble, axis=0)
        # TODO: store this covariance for investigate
        concentration_factor = 0.1
        minimum_exploration_level = 0
        a_cov_shaped = concentration_factor * a_cov + minimum_exploration_level * np.ones(a_cov.shape)
        a = np.random.multivariate_normal(a_mean, a_cov_shaped, 1)[0]
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
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all DEDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for _ in range(ep_len):
                # TODO: should each actor-critic be trained with different batchs or the same batch
                if feed_ac_ensemble_with_same_batch:
                    batch = replay_buffer.sample_batch(batch_size)
                    for ac_name in ac_ensemble.keys():
                        ac_ensemble[ac_name]['feed_dict'] = {x_ph: batch['obs1'],
                                                             x2_ph: batch['obs2'],
                                                             a_ph: batch['acts'],
                                                             r_ph: batch['rews'],
                                                             d_ph: batch['done']
                                                             }
                else:
                    for ac_name in ac_ensemble.keys():
                        batch = replay_buffer.sample_batch(batch_size)
                        ac_ensemble[ac_name]['feed_dict'] = {x_ph: batch['obs1'],
                                                             x2_ph: batch['obs2'],
                                                             a_ph: batch['acts'],
                                                             r_ph: batch['rews'],
                                                             d_ph: batch['done']
                                                             }

                # train actor-critic ensemble
                for ac_name in ac_ensemble.keys():
                    # Q-learning update
                    outs = sess.run([ac_ensemble[ac_name]['q_loss'],
                                     ac_ensemble[ac_name]['q'],
                                     ac_ensemble[ac_name]['train_q_op']], ac_ensemble[ac_name]['feed_dict'])

                    logger.store(LossQ=outs[0], QVals=outs[1])

                    # Policy update
                    # TODO: delayed policy update?
                    outs = sess.run([ac_ensemble[ac_name]['pi_loss'],
                                     ac_ensemble[ac_name]['train_pi_op'],
                                     ac_ensemble[ac_name]['target_update']], ac_ensemble[ac_name]['feed_dict'])
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
    parser.add_argument('--feed_ac_ensemble_with_same_batch', action='store_true')
    parser.add_argument('--exp_name', type=str, default='dedpg_5ac')
    parser.add_argument('--ensemble_size', type=int, default=5)
    parser.add_argument('--hardcopy_target_nn', action="store_true", help='Target network update method: hard copy')

    parser.add_argument("--exploration-strategy", type=str, choices=["action_noise", "epsilon_greedy"],
                        default='epsilon_greedy', help='action_noise or epsilon_greedy')
    parser.add_argument("--epsilon-max", type=float, default=1.0, help='maximum of epsilon')
    parser.add_argument("--epsilon-min", type=float, default=.01, help='minimum of epsilon')
    parser.add_argument("--epsilon-decay", type=float, default=.001, help='epsilon decay')


    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, datestamp=True)

    # if args.hardcopy_target_nn:
    #     polyak = 0

    dedpg(lambda : gym.make(args.env), actor_critic_ensemble=core.mlp_actor_critic_ensemble,
          ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
          gamma=args.gamma, seed=args.seed, epochs=args.epochs,
          ensemble_size = args.ensemble_size,
          feed_ac_ensemble_with_same_batch=args.feed_ac_ensemble_with_same_batch,
          logger_kwargs=logger_kwargs)