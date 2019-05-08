import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.sddpg_simple2 import core
from spinup.algos.sddpg_simple2.core import get_vars
from spinup.utils.logx import EpochLogger, Logger
import os.path as osp
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size,
                 logger_fname='experiences_log.txt', **logger_kwargs):
        # ExperienceLogger: save experiences for supervised learning
        logger_kwargs['output_fname'] = logger_fname
        self.experience_logger = Logger(**logger_kwargs)

        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.acts_mu_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.acts_sigma_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, act_mu, act_sigma, rew, next_obs, done,
              step_index, steps_per_epoch, start_time):
        # Save experiences in disk
        self.log_experiences(obs, act, act_mu, act_sigma, rew, next_obs, done,
                             step_index, steps_per_epoch, start_time)
        # Save experiences in memory
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.acts_mu_buf[self.ptr] = act_mu
        self.acts_sigma_buf[self.ptr] = act_sigma
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    acts_mu=self.acts_mu_buf[idxs],
                    acts_sigma=self.acts_sigma_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def log_experiences(self, obs, act, act_mu, act_sigma, rew, next_obs, done,
                        step_index, steps_per_epoch, start_time):
        self.experience_logger.log_tabular('Epoch', step_index // steps_per_epoch)
        self.experience_logger.log_tabular('Step', step_index)
        # Log observation
        for i, o_i in enumerate(obs):
            self.experience_logger.log_tabular('o_{}'.format(i), o_i)
        # Log action
        for i, a_i in enumerate(act):
            self.experience_logger.log_tabular('a_{}'.format(i), a_i)
        for i, a_mu_i in enumerate(act_mu):
            self.experience_logger.log_tabular('a_mu_{}'.format(i), a_mu_i)
        # if np.sum(act_sigma > 1) >= 1:
        # if step_index > 10000:
        #     import pdb;
        #     pdb.set_trace()
        for i, a_sigma_i in enumerate(act_sigma):
            if a_sigma_i>1:
                print('step_index={}, a_sigma_{}={}'.format(step_index, i, a_sigma_i))
            self.experience_logger.log_tabular('a_sigma_{}'.format(i), a_sigma_i)
        # Log reward
        self.experience_logger.log_tabular('r', rew)
        # Log next observation
        for i, o2_i in enumerate(next_obs):
            self.experience_logger.log_tabular('o2_{}'.format(i), o2_i)
        # Log done
        self.experience_logger.log_tabular('d', done)
        self.experience_logger.log_tabular('Time', time.time() - start_time)
        self.experience_logger.dump_tabular(print_data=False)

"""

Stochastic Deep Deterministic Policy Gradient (DDPG)

"""
def sddpg_simple2(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
                 steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
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

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    # x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    x_ph, \
    a_ph, a_mu_ph, a_sigma_ph,\
    x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, act_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, pi_mu, pi_sigma, q, q_pi, q_pi_mu = actor_critic(x_ph, a_ph, **ac_kwargs)
        # pi, q, q_mu, q_sigma, q_pi, q_pi_mu, q_pi_sigma = actor_critic(x_ph, a_ph, **ac_kwargs)
    # Target networks
    with tf.variable_scope('target'):
        # Note that the action placeholder going to actor_critic here is 
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        pi_targ, pi_mu_targ, pi_sigma_targ, _, q_pi_targ, q_pi_mu_targ = actor_critic(x2_ph, a_ph, **ac_kwargs)
        # pi_targ, _, _, _, q_pi_targ, q_pi_mu_targ, q_pi_sigma_targ = actor_critic(x2_ph, a_ph, **ac_kwargs)
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
                                 logger_fname='experiences_log.txt', **logger_kwargs)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q function
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_pi_targ)

    # DDPG losses
    # TODO: add term to penalize large variance, give penalize term cofficient
    # pi_loss = tf.reduce_mean(-q_pi + tf.norm(pi_sigma,ord=1,axis=1))
    # adv_explore = q_pi - q_pi_mu
    pi_loss = tf.reduce_mean(-q_pi-q_pi_mu)
    # pi_loss = tf.reduce_mean(-q_pi)
    q_loss = tf.reduce_mean((q-backup)**2)

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

    writer = tf.summary.FileWriter(osp.join(logger_kwargs['output_dir'], 'graph'), sess.graph)
    writer.flush()
    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                          outputs={'pi': pi, 'pi_mu': pi_mu, 'pi_sigma': pi_sigma, 'q': q})

    def get_action(o):
        # import pdb; pdb.set_trace()
        a, a_mu, a_sigma = sess.run([pi, pi_mu, pi_sigma], feed_dict={x_ph: o.reshape(1,-1)})
        return a[0], a_mu[0], a_sigma[0]
        # return np.clip(a[0], -act_limit, act_limit), a_mu[0], a_sigma[0]

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a, a_mu, a_sigma = get_action(o)
                o, r, d, _ = test_env.step(a)
                # o, r, d, _ = test_env.step(a_mu)
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
            a, a_mu, a_sigma = get_action(o)
        else:
            a = env.action_space.sample()
            a_mu = a
            a_sigma = np.zeros((act_dim,))

        # Step the env
        env.render()
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        # TODO: delete scaled r
        replay_buffer.store(o, a, a_mu, a_sigma, 5*r, o2, d,
                            t, steps_per_epoch, start_time)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             a_mu_ph: batch['acts_mu'],
                             a_sigma_ph: batch['acts_sigma'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']
                            }
                #
                # outs = sess.run([pi, pi_mu, pi_sigma], feed_dict)

                # Q-learning update
                outs = sess.run([q_loss, q, train_q_op], feed_dict)
                logger.store(LossQ=outs[0], QVals=outs[1])
                if outs[0]>10000:
                    print('q_loss={}'.format(outs[0]))
                    # import pdb;
                    # pdb.set_trace()
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
            # TODO: change test number
            test_agent(1)

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
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--hardcopy_target_nn', action="store_true", help='Target network update method: hard copy')

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

    sddpg_simple2(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
                 ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                 gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                 logger_kwargs=logger_kwargs)

# python  spinup/algos/ddpg/ddpg.py --env HalfCheetah-v2 --seed 3 --l 2 --exp_name ddpg_two_layers_delay_policy
# python  spinup/algos/ddpg/ddpg.py --env Ant-v2 --seed 3 --l 2 --exp_name ddpg_Ant_v2_two_layers_delay_policy
# python  spinup/algos/ddpg/ddpg.py --env Humanoid-v2 --seed 3 --l 2 --exp_name ddpg_Humanoid_v2_two_layers_delay_policy