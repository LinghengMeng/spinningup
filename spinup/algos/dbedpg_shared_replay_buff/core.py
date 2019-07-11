import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from spinup.utils.logx import Logger
import time
from multiprocessing import Process
# import ray
# if not ray.is_initialized():
#     ray.init(num_gpus=1)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

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
              step_index, epoch_index, time, **kwargs):
        # Save experiences in disk
        self.log_experiences(obs, act, rew, next_obs, done,
                             step_index, epoch_index, time, **kwargs)
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
                        step_index, epoch_index, time, **kwargs):
        self.experience_logger.log_tabular('Epoch', epoch_index)
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
        self.experience_logger.log_tabular('Time', time)
        self.experience_logger.dump_tabular(print_data=False)

class MLP(tf.keras.Model):
    """
    Multi-Layer Perceptron Network:
        Model class used to create mlp.
    """
    def __init__(self, layer_sizes=[32],
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None,
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(MLP, self).__init__()
        self.hidden_layers = []
        for h in layer_sizes[:-1]:
            self.hidden_layers.append(tf.keras.layers.Dense(h, activation=hidden_activation,
                                                            kernel_initializer=kernel_initializer,
                                                            bias_initializer=bias_initializer,
                                                            kernel_regularizer=kernel_regularizer))
        self.out_layer = tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation,
                                               kernel_initializer=kernel_initializer,
                                               bias_initializer=bias_initializer,
                                               kernel_regularizer=kernel_regularizer)

    def call(self, inputs):
        x = inputs
        for h_layer in self.hidden_layers:
            x = h_layer(x)
        return self.out_layer(x)


class BootstrappedActorCriticEnsemble():
    def __init__(self, ensemble_size,
                 obs_dim, act_dim, act_limit, hidden_sizes,
                 gamma, pi_lr, q_lr, polyak,
                 replay_size, replay_buf_bootstrap_p, logger_kwargs):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit

        self.hidden_sizes = hidden_sizes
        self.replay_size = replay_size
        self.replay_buf_bootstrap_p = replay_buf_bootstrap_p
        self.logger_kwargs = logger_kwargs

        self.ensemble_size = ensemble_size
        self.ensemble = []

        self.actor_hidden_activation = tf.keras.activations.relu
        self.actor_output_activation = tf.keras.activations.tanh
        self.critic_hidden_activation = tf.keras.activations.relu
        self.critic_output_activation = tf.keras.activations.linear
        self.gamma = gamma
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.polyak = polyak

        # Replay_Buffer
        self.replay_buf = ReplayBuffer(self.obs_dim, self.act_dim, self.replay_size,
                                       logger_fname='exp_log_ac_ensemble.txt', **self.logger_kwargs)
        # Create ensemble
        for ac_i in range(self.ensemble_size):
            self.ensemble.append(self._create_actor_critic(ac_i))

        # Gather training ops
        ac_en_q_loss_ops = [self.ensemble[ac_i]['q_loss'] for ac_i in range(self.ensemble_size)]
        ac_en_q_ops = [self.ensemble[ac_i]['q'] for ac_i in range(self.ensemble_size)]
        ac_en_train_q_ops = [self.ensemble[ac_i]['train_q_op'] for ac_i in range(self.ensemble_size)]
        self.train_critic_ops = [ac_en_q_loss_ops, ac_en_q_ops, ac_en_train_q_ops]

        ac_en_pi_loss_ops = [self.ensemble[ac_i]['pi_loss'] for ac_i in range(self.ensemble_size)]
        ac_en_train_pi_ops = [self.ensemble[ac_i]['train_pi_op'] for ac_i in range(self.ensemble_size)]
        ac_en_target_update_ops = [self.ensemble[ac_i]['target_update'] for ac_i in range(self.ensemble_size)]
        self.train_actor_ops = [ac_en_pi_loss_ops, ac_en_train_pi_ops, ac_en_target_update_ops]

        # Create tf.Session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Initialize target
        for ac_i in range(self.ensemble_size):
            self.sess.run(self.ensemble[ac_i]['target_init'])

    def _create_actor_critic(self, ac_i):
        """Create actor-critic"""
        obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
        act_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.act_dim))
        rew_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        new_obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
        done_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

        # Actor and Critic
        with tf.variable_scope('{}_main'.format(ac_i)):
            actor = MLP(self.hidden_sizes + [self.act_dim],
                        hidden_activation=self.actor_hidden_activation,
                        output_activation=self.actor_output_activation)
            critic = MLP(self.hidden_sizes + [1],
                         hidden_activation=self.critic_hidden_activation,
                         output_activation=self.critic_output_activation)
            pi = self.act_limit * actor(obs_ph)
            q = tf.squeeze(critic(tf.concat([obs_ph, act_ph], axis=-1)), axis=1)
            q_pi = tf.squeeze(critic(tf.concat([obs_ph, pi], axis=-1)), axis=1)

        # Target Actor and Target Critic
        with tf.variable_scope('{}_target'.format(ac_i)):
            actor_targ = MLP(self.hidden_sizes + [self.act_dim],
                             hidden_activation=self.actor_hidden_activation,
                             output_activation=self.actor_output_activation)
            critic_targ = MLP(self.hidden_sizes + [1],
                              hidden_activation=self.critic_hidden_activation,
                              output_activation=self.critic_output_activation)
            pi_targ = self.act_limit * actor_targ(new_obs_ph)
            q_pi_targ = tf.squeeze(critic_targ(tf.concat([new_obs_ph, pi_targ], axis=-1)), axis=1)

        # Loss
        backup = tf.stop_gradient(rew_ph + self.gamma * (1 - done_ph) * q_pi_targ)
        pi_loss = -tf.reduce_mean(q_pi)
        q_loss = tf.reduce_mean((q - backup) ** 2)

        # Optimization
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.pi_lr)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=self.q_lr)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=actor.variables)
        train_q_op = q_optimizer.minimize(q_loss, var_list=critic.variables)

        # Update Target
        target_update = tf.group([tf.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
                                  for v_main, v_targ in zip(actor.variables + critic.variables,
                                                            actor_targ.variables + critic_targ.variables)])
        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(actor.variables + critic.variables,
                                                          actor_targ.variables + critic_targ.variables)])
        return dict(obs_ph=obs_ph, act_ph = act_ph, rew_ph = rew_ph, new_obs_ph = new_obs_ph, done_ph = done_ph,
                    actor=actor, critic=critic,
                    pi=pi, q=q, q_pi=q_pi,
                    actor_targ=actor_targ, critic_targ=critic_targ,
                    pi_targ=pi_targ, q_pi_targ=q_pi_targ,
                    backup=backup, pi_loss=pi_loss, q_loss=q_loss,
                    train_pi_op=train_pi_op, train_q_op=train_q_op,
                    target_update=target_update, target_init=target_init)

    def add_to_replay_buffer(self, obs, act, rew, next_obs, done,
                             step_index, epoch_index, time_stamp, **kwargs):
        """Add experience to each Actor-Critic's replay buffer with probability replay_buf_bootstrapp_p"""
        self.replay_buf.store(obs, act, rew, next_obs, done,
                              step_index, epoch_index, time_stamp, **kwargs)

    def predict_a(self, input, ac_i=None):
        """Prediction of actor-critic ensemble"""
        if ac_i is None:
            # prediction of all members
            feed_dict = {self.ensemble[ac_i]['obs_ph']: input.reshape(1, -1) for ac_i in range(self.ensemble_size)}
            preds = self.sess.run([self.ensemble[ac_i]['pi'] for ac_i in range(self.ensemble_size)], feed_dict=feed_dict)
        else:
            # prediction of a specific member
            feed_dict = {self.ensemble[ac_i]['obs_ph']: input.reshape(1, -1)}
            preds = self.sess.run(self.ensemble[ac_i]['pi'], feed_dict=feed_dict)
        return np.squeeze(np.asarray(preds))

    def predict_q(self, input, ac_i=None):
        """Prediction of actor-critic ensemble"""
        if ac_i is None:
            # prediction of all members
            feed_dict = {self.ensemble[ac_i]['obs_ph']: input.reshape(1, -1) for ac_i in range(self.ensemble_size)}
            preds = self.sess.run([self.ensemble[ac_i]['q_pi'] for ac_i in range(self.ensemble_size)], feed_dict=feed_dict)
        else:
            # prediction of a specific member
            feed_dict = {self.ensemble[ac_i]['obs_ph']: input.reshape(1, -1)}
            preds = self.sess.run(self.ensemble[ac_i]['q_pi'], feed_dict=feed_dict)
        return np.squeeze(np.asarray(preds))

    def train(self, batch_size=100, raw_batch_size=500, uncertainty_based_minibatch=False):
        """Train each actor-critic on its corresponding replay_buffer"""
        feed_dict = {}
        for ac_i in range(self.ensemble_size):
            if uncertainty_based_minibatch:
                # Select top n highest uncertainty samples
                batch = self._uncertainty_based_mini_batch(ac_i, batch_size, raw_batch_size)
            else:
                batch = self.replay_buf.sample_batch(batch_size)
            feed_dict[self.ensemble[ac_i]['obs_ph']] = batch['obs1']
            feed_dict[self.ensemble[ac_i]['act_ph']] = batch['acts']
            feed_dict[self.ensemble[ac_i]['rew_ph']] = batch['rews']
            feed_dict[self.ensemble[ac_i]['new_obs_ph']] = batch['obs2']
            feed_dict[self.ensemble[ac_i]['done_ph']] = batch['done']
        train_outs = self.sess.run([self.train_critic_ops, self.train_actor_ops], feed_dict=feed_dict)
        return np.asarray(train_outs[0][0]), np.asarray(train_outs[0][1]), np.asarray(train_outs[1][0])

    def _uncertainty_based_mini_batch(self, ac_i, batch_size, raw_batch_size):
        raw_batch = self.replay_buf.sample_batch(raw_batch_size)
        feed_dict = {self.ensemble[ac_i]['obs_ph']: raw_batch['obs1'] for ac_i in range(self.ensemble_size)}
        # # action-based uncertainty
        # preds_a = self.sess.run([self.ensemble[ac_i]['pi'] for ac_i in range(self.ensemble_size)], feed_dict)
        # preds_a = np.transpose(np.asarray(preds_a), (1, 0, 2)) # Original: ensemble_size x raw_batch_size x act_dim
        # q-value-based uncertainty
        preds_q = self.sess.run([self.ensemble[ac_i]['q_pi'] for ac_i in range(self.ensemble_size)], feed_dict)
        preds_q = np.transpose(np.asarray(preds_q), (1, 0))  # Original: ensemble_size x raw_batch_size
        uncertainty_q = np.var(preds_q, axis=1)
        top_n_highest_unc_index = np.argsort(-uncertainty_q)[:batch_size]
        batch = {}
        batch['obs1'] = raw_batch['obs1'][top_n_highest_unc_index, :]
        batch['acts'] = raw_batch['acts'][top_n_highest_unc_index, :]
        batch['rews'] = raw_batch['rews'][top_n_highest_unc_index]
        batch['obs2'] = raw_batch['obs2'][top_n_highest_unc_index, :]
        batch['done'] = raw_batch['done'][top_n_highest_unc_index]
        return batch

    def save_model(self):
        # TODO: save model
        pass


