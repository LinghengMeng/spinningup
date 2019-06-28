import numpy as np
import tensorflow as tf
import pybulletgym
# import roboschool
import gym
import time
from spinup.algos.dbedpg_distributed import core
from spinup.algos.dbedpg_distributed.core import get_vars, BootstrappedActorCriticEnsemble
from spinup.utils.logx import EpochLogger
import os.path as osp

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""

Deep Bootstrapped Ensemble Deterministic Policy Gradient (DBEDPG)

"""
def dbedpg_distributed(env_name, render_env=False, ac_kwargs=dict(), seed=0,
           steps_per_epoch=5000, epochs=100,
           ensemble_size = 20,
           batch_size=100, raw_batch_size=500, uncertainty_based_minibatch=False,
           replay_size=int(1e6), replay_buf_bootstrap_p=0.75,
           gamma=0.99,
           polyak=0.995, pi_lr=1e-3, q_lr=1e-3, start_steps=10000,
           act_noise=0.1, policy_delay=2, max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """

    Args:
        env_name : Environment name.
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

    hidden_sizes = list(ac_kwargs['hidden_sizes'])
    ac_ensemble = BootstrappedActorCriticEnsemble(ensemble_size,
                                                  obs_dim, act_dim, act_limit, hidden_sizes,
                                                  gamma, pi_lr, q_lr, polyak,
                                                  replay_size, replay_buf_bootstrap_p, logger_kwargs)

    # # Setup model saving
    # logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q': q})

    def get_action(o, noise_scale, ac_i=0):
        ac_ensemble_preds_a = ac_ensemble.predict_a(o)
        ac_ensemble_preds_a = ac_ensemble_preds_a.reshape(ensemble_size, -1)
        # Select action with the highest predicted Q-value
        ac_ensemble_preds_q = ac_ensemble.predict_q(o)
        # Descending sort
        descending_sort_q = np.argsort(-ac_ensemble_preds_q)
        a = ac_ensemble_preds_a[descending_sort_q[0], :]
        a += noise_scale * np.random.randn(act_dim)

        # #1. random actor-critic
        # a = ac_ensemble_preds_a[ac_i, :]
        # a += noise_scale * np.random.randn(act_dim)
        # 2. one actor-critic for each epoch
        # a = ac_ensemble_preds[ac_i, :]
        # a += noise_scale * np.random.randn(act_dim)
        # # 3.
        # concentration_factor = 0.5
        # a_cov = np.cov(ac_ensemble_preds, rowvar=False)
        # a_cov_shaped = concentration_factor * a_cov
        # a = np.random.multivariate_normal(np.mean(ac_ensemble_preds, axis=0), a_cov_shaped, 1)[0]
        return np.clip(a, -act_limit, act_limit)

    def get_action_test(o, ac_i):
        a = ac_ensemble.predict_a(o, ac_i)
        return a

    def test_agent(n=10, ac_i=0):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action_test(o, ac_i))
                ep_ret += r
                ep_len += 1
            log_args = {'TestEpRetAC{}'.format(ac_i): ep_ret, 'TestEpLenAC{}'.format(ac_i): ep_len}
            logger.store(**log_args)
        return ['TestEpRetAC{}'.format(ac_i), 'TestEpLenAC{}'.format(ac_i)]

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
            a = get_action(o, act_noise, ac_i=int((t // steps_per_epoch)%ensemble_size))
        else:
            a = env.action_space.sample()

        # # TODO: delete
        # pred = ac_ensemble.predict_a(o)
        # pred_q = ac_ensemble.predict_q(o)
        # get_action(o, act_noise, ac_i=0)
        # import pdb; pdb.set_trace()

        # Step the env
        if render_env:
            env.render()
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        ac_ensemble.add_to_replay_buffer(o, a, r, o2, d, t, t // steps_per_epoch, time.time()-start_time)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for j in range(ep_len):
                ac_en_q_loss, ac_en_q, ac_en_pi_loss = ac_ensemble.train(batch_size,
                                                                         raw_batch_size,
                                                                         uncertainty_based_minibatch)
                kwargs = {}
                for i, q_loss in enumerate(ac_en_q_loss):
                    kwargs['LossQ{}'.format(i)] = q_loss
                logger.store(**kwargs)

                kwargs = {}
                for i, pi_loss in enumerate(ac_en_pi_loss):
                    kwargs['LossPi{}'.format(i)] = pi_loss
                logger.store(**kwargs)

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of each agent in the ensemble.
            test_epochs = 2
            test_log_args_keys = []
            # TODO: parallel
            for ac_i in range(ensemble_size):
                log_args_keys = test_agent(test_epochs, ac_i)
                test_log_args_keys.append(log_args_keys)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            for log_args_kyes in test_log_args_keys:
                logger.log_tabular(log_args_kyes[0], with_min_and_max=True)
                logger.log_tabular(log_args_kyes[1], average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            for i in range(ensemble_size):
                logger.log_tabular('LossQ{}'.format(i), with_min_and_max=True)
                logger.log_tabular('LossPi{}'.format(i), with_min_and_max=True)
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
    parser.add_argument('--exp_name', type=str, default='dbedpg')
    parser.add_argument('--ensemble_size', type=int, default=20)
    parser.add_argument('--replay_buf_bootstrap_p', type=float, default=1)
    parser.add_argument('--uncertainty_based_minibatch', action='store_true')
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

    dbedpg_distributed(env_name=args.env, render_env=args.render_env,
                       act_noise=args.act_noise,
                       ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                       gamma=args.gamma, seed=args.seed,
                       ensemble_size=args.ensemble_size,
                       replay_buf_bootstrap_p=args.replay_buf_bootstrap_p,
                       uncertainty_based_minibatch=args.uncertainty_based_minibatch,
                       epochs=args.epochs,
                       steps_per_epoch=args.steps_per_epoch, start_steps=args.start_steps,
                       logger_kwargs=logger_kwargs)

