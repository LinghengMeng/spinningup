import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from spinup.utils.logx import Logger
import time
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

class ActorCritic():
    """
    Class used for create an Actor-Critic.
    """
    def __init__(self, ac_name, obs_dim, act_dim, act_limit, hidden_sizes, gamma, pi_lr, q_lr, polyak):
        self.act_name = ac_name
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
        self.act_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.act_dim))
        self.rew_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.new_obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
        self.done_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

        self.hidden_sizes = hidden_sizes

        self.actor_hidden_activation = tf.keras.activations.relu
        self.actor_output_activation = tf.keras.activations.tanh
        self.critic_hidden_activation = tf.keras.activations.relu
        self.critic_output_activation = tf.keras.activations.linear
        self.gamma = gamma
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.polyak = polyak

        # Actor and Critic
        with tf.variable_scope('{}_main'.format(ac_name)):
            self.actor = MLP(self.hidden_sizes + [self.act_dim],
                             hidden_activation=self.actor_hidden_activation,
                             output_activation=self.actor_output_activation)
            self.critic = MLP(self.hidden_sizes + [1],
                              hidden_activation=self.critic_hidden_activation,
                              output_activation=self.critic_output_activation)
            self.pi = self.act_limit * self.actor(self.obs_ph)
            self.q = tf.squeeze(self.critic(tf.concat([self.obs_ph, self.act_ph], axis=-1)), axis=1)
            self.q_pi = tf.squeeze(self.critic(tf.concat([self.obs_ph, self.pi], axis=-1)), axis=1)

        # Target Actor and Target Critic
        with tf.variable_scope('{}_target'.format(ac_name)):
            self.actor_targ = MLP(self.hidden_sizes + [self.act_dim],
                                  hidden_activation=self.actor_hidden_activation,
                                  output_activation=self.actor_output_activation)
            self.critic_targ = MLP(self.hidden_sizes + [1],
                                   hidden_activation=self.critic_hidden_activation,
                                   output_activation=self.critic_output_activation)
            self.pi_targ = self.act_limit * self.actor_targ(self.new_obs_ph)
            self.q_pi_targ = tf.squeeze(self.critic_targ(tf.concat([self.new_obs_ph, self.pi_targ], axis=-1)), axis=1)

        # Loss
        self.backup = tf.stop_gradient(self.rew_ph + self.gamma * (1 - self.done_ph) * self.q_pi_targ)
        self.pi_loss = -tf.reduce_mean(self.q_pi)
        self.q_loss = tf.reduce_mean((self.q - self.backup) ** 2)

        # Optimization
        self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.pi_lr)
        self.q_optimizer = tf.train.AdamOptimizer(learning_rate=self.q_lr)
        self.train_pi_op = self.pi_optimizer.minimize(self.pi_loss, var_list=self.actor.variables)
        self.train_q_op = self.q_optimizer.minimize(self.q_loss, var_list=self.critic.variables)

        # Update Target
        self.target_update = tf.group([tf.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
                                       for v_main, v_targ in zip(self.actor.variables + self.critic.variables,
                                                                 self.actor_targ.variables + self.critic_targ.variables)])
        # Initializing targets to match main variables
        self.target_init = tf.group([tf.assign(v_targ, v_main)
                                     for v_main, v_targ in zip(self.actor.variables + self.critic.variables,
                                                               self.actor_targ.variables + self.critic_targ.variables)])

        # Create tf.Session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Initialize target
        self.sess.run(self.target_init)

    def train(self, batch):
        feed_dict = {self.obs_ph: batch['obs1'],
                     self.act_ph: batch['acts'],
                     self.rew_ph: batch['rews'],
                     self.new_obs_ph: batch['obs2'],
                     self.done_ph: batch['done']
                     }

        # Train critic
        train_critic_ops = [self.q_loss, self.q, self.train_q_op]
        critic_outs = self.sess.run(train_critic_ops, feed_dict=feed_dict)
        ac_en_q_loss = critic_outs[0]
        ac_en_q = critic_outs[1]
        # Train actor
        train_actor_ops = [self.pi_loss, self.train_pi_op, self.target_update]
        actor_outs = self.sess.run(train_actor_ops, feed_dict=feed_dict)
        ac_en_pi_loss = actor_outs[0]
        return ac_en_q_loss, ac_en_q, ac_en_pi_loss

    def predict(self, input):
        feed_dict = {self.obs_ph: input.reshape(1, -1)}
        return self.sess.run(self.pi, feed_dict=feed_dict)[0]

    def save_model(self):
        # TODO
        pass

class BootstrappedActorCriticEnsemble():
    def __init__(self,ensemble_size,
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

        self.gamma = gamma
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.polyak = polyak

        # Create actor-critic ensemble
        self.ensemble_size = ensemble_size
        self.ensemble = [ActorCritic('act_{}'.format(i), self.obs_dim, self.act_dim, self.act_limit, self.hidden_sizes,
                                     self.gamma, self.pi_lr, self.q_lr, self.polyak)
                         for i in range(self.ensemble_size)]

        # Create Replay Buffer
        self.ensemble_replay_bufs = [ReplayBuffer(self.obs_dim, self.act_dim, self.replay_size,
                                                  logger_fname='exp_log_ac_{}.txt'.format(ac_i),
                                                  **self.logger_kwargs)
                                     for ac_i in range(self.ensemble_size)]

    def predict(self, input):
        predics = [self.ensemble[i].predict(input) for i in range(self.ensemble_size)]
        return np.asarray(predics)

    def train(self, batch_size=100, raw_batch_size=500, uncertainty_based_minibatch=False):
        # Generate mini-batch
        batches = self._generate_mini_batch(batch_size, raw_batch_size, uncertainty_based_minibatch)

        # Train each member on its mini-batch
        train_outs = [self.ensemble[i].train(batches[i]) for i in range(self.ensemble_size)]

        train_outs = np.asarray(train_outs)
        ac_en_q_loss = train_outs[:, 0]
        ac_en_q = np.stack(train_outs[:,1], axis=0)
        ac_en_pi_loss = train_outs[:, 2]
        return ac_en_q_loss, ac_en_q, ac_en_pi_loss

    def _generate_mini_batch(self, batch_size=100, raw_batch_size=500, uncertainty_based_minibatch=False):
        # TODO：use multiprocessing to parallel mini-batch sampling
        import pdb; pdb.set_trace()
        start_time = time.time()
        random_mini_batches = [reply_buf.sample_batch(batch_size) for reply_buf in self.ensemble_replay_bufs]
        print('ramdom mini-batch sampling costs: {}s'.format(time.time()-start_time))
        return random_mini_batches

    def add_to_replay_buffer(self, obs, act, rew, next_obs, done,
                             step_index, epoch_index, time, **kwargs):
        """Add experience to each Actor-Critic's replay buffer with probability replay_buf_bootstrapp_p"""
        for ac_i in range(self.ensemble_size):
            if np.random.uniform(0, 1, 1) < self.replay_buf_bootstrap_p:
                self.ensemble_replay_bufs[ac_i].store(obs, act, rew, next_obs, done,
                                                      step_index, epoch_index, time, **kwargs)

# @ray.remote
# class ActorCritic():
#     """
#     Class used for create an Actor-Critic.
#     """
#     def __init__(self, ac_name, obs_dim, act_dim, act_limit, hidden_sizes, gamma, pi_lr, q_lr, polyak):
#         self.act_name = ac_name
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self.act_limit = act_limit
#         self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
#         self.act_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.act_dim))
#         self.rew_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
#         self.new_obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
#         self.done_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
#
#         self.hidden_sizes = hidden_sizes
#
#         self.actor_hidden_activation = tf.keras.activations.relu
#         self.actor_output_activation = tf.keras.activations.tanh
#         self.critic_hidden_activation = tf.keras.activations.relu
#         self.critic_output_activation = tf.keras.activations.linear
#         self.gamma = gamma
#         self.pi_lr = pi_lr
#         self.q_lr = q_lr
#         self.polyak = polyak
#
#         # Actor and Critic
#         with tf.variable_scope('{}_main'.format(ac_name)):
#             self.actor = MLP(self.hidden_sizes + [self.act_dim],
#                              hidden_activation=self.actor_hidden_activation,
#                              output_activation=self.actor_output_activation)
#             self.critic = MLP(self.hidden_sizes + [1],
#                               hidden_activation=self.critic_hidden_activation,
#                               output_activation=self.critic_output_activation)
#             self.pi = self.act_limit * self.actor(self.obs_ph)
#             self.q = tf.squeeze(self.critic(tf.concat([self.obs_ph, self.act_ph], axis=-1)), axis=1)
#             self.q_pi = tf.squeeze(self.critic(tf.concat([self.obs_ph, self.pi], axis=-1)), axis=1)
#
#         # Target Actor and Target Critic
#         with tf.variable_scope('{}_target'.format(ac_name)):
#             self.actor_targ = MLP(self.hidden_sizes + [self.act_dim],
#                                   hidden_activation=self.actor_hidden_activation,
#                                   output_activation=self.actor_output_activation)
#             self.critic_targ = MLP(self.hidden_sizes + [1],
#                                    hidden_activation=self.critic_hidden_activation,
#                                    output_activation=self.critic_output_activation)
#             self.pi_targ = self.act_limit * self.actor_targ(self.new_obs_ph)
#             self.q_pi_targ = tf.squeeze(self.critic_targ(tf.concat([self.new_obs_ph, self.pi_targ], axis=-1)), axis=1)
#
#         # Loss
#         self.backup = tf.stop_gradient(self.rew_ph + self.gamma * (1 - self.done_ph) * self.q_pi_targ)
#         self.pi_loss = -tf.reduce_mean(self.q_pi)
#         self.q_loss = tf.reduce_mean((self.q - self.backup) ** 2)
#
#         # Optimization
#         self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.pi_lr)
#         self.q_optimizer = tf.train.AdamOptimizer(learning_rate=self.q_lr)
#         self.train_pi_op = self.pi_optimizer.minimize(self.pi_loss, var_list=self.actor.variables)
#         self.train_q_op = self.q_optimizer.minimize(self.q_loss, var_list=self.critic.variables)
#
#         # Update Target
#         self.target_update = tf.group([tf.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
#                                        for v_main, v_targ in zip(self.actor.variables + self.critic.variables,
#                                                                  self.actor_targ.variables + self.critic_targ.variables)])
#         # Initializing targets to match main variables
#         self.target_init = tf.group([tf.assign(v_targ, v_main)
#                                      for v_main, v_targ in zip(self.actor.variables + self.critic.variables,
#                                                                self.actor_targ.variables + self.critic_targ.variables)])
#
#         # Create tf.Session
#         self.sess = tf.Session()
#         self.sess.run(tf.global_variables_initializer())
#
#         # Initialize target
#         self.sess.run(self.target_init)
#
#     def train(self, batch):
#         feed_dict = {self.obs_ph: batch['obs1'],
#                      self.act_ph: batch['acts'],
#                      self.rew_ph: batch['rews'],
#                      self.new_obs_ph: batch['obs2'],
#                      self.done_ph: batch['done']
#                      }
#
#         # Train critic
#         train_critic_ops = [self.q_loss, self.q, self.train_q_op]
#         critic_outs = self.sess.run(train_critic_ops, feed_dict=feed_dict)
#         ac_en_q_loss = critic_outs[0]
#         ac_en_q = critic_outs[1]
#         # Train actor
#         train_actor_ops = [self.pi_loss, self.train_pi_op, self.target_update]
#         actor_outs = self.sess.run(train_actor_ops, feed_dict=feed_dict)
#         ac_en_pi_loss = actor_outs[0]
#         return ac_en_q_loss, ac_en_q, ac_en_pi_loss
#
#     def predict(self, input):
#         feed_dict = {self.obs_ph: input.reshape(1, -1)}
#         return self.sess.run(self.pi, feed_dict=feed_dict)[0]
#
#     def save_model(self):
#         # TODO
#         pass
#
# class BootstrappedActorCriticEnsemble():
#     def __init__(self,ensemble_size,
#                  obs_dim, act_dim, act_limit, hidden_sizes,
#                  gamma, pi_lr, q_lr, polyak,
#                  replay_size, replay_buf_bootstrap_p, logger_kwargs):
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self.act_limit = act_limit
#
#         self.hidden_sizes = hidden_sizes
#         self.replay_size = replay_size
#         self.replay_buf_bootstrap_p = replay_buf_bootstrap_p
#         self.logger_kwargs = logger_kwargs
#
#         self.gamma = gamma
#         self.pi_lr = pi_lr
#         self.q_lr = q_lr
#         self.polyak = polyak
#
#         # Create actor-critic ensemble
#         self.ensemble_size = ensemble_size
#         self.ensemble = [ActorCritic.remote('act_{}'.format(i),
#                                             self.obs_dim, self.act_dim, self.act_limit, self.hidden_sizes,
#                                             self.gamma, self.pi_lr, self.q_lr, self.polyak)
#                                   for i in range(self.ensemble_size)]
#
#         # Create Replay Buffer
#         self.ensemble_replay_bufs = [ReplayBuffer(self.obs_dim, self.act_dim, self.replay_size,
#                                                   logger_fname='exp_log_ac_{}.txt'.format(ac_i),
#                                                   **self.logger_kwargs)
#                                      for ac_i in range(self.ensemble_size)]
#
#     def predict(self, input):
#         predics_id, _ = ray.wait([self.ensemble[i].predict.remote(input) for i in range(self.ensemble_size)],
#                                  num_returns=self.ensemble_size)
#         predics = ray.get(predics_id)
#         return np.asarray(predics)
#
#     def train(self, batch_size=100, raw_batch_size=500, uncertainty_based_minibatch=False):
#         # Generate mini-batch
#         batches = self._generate_mini_batch(batch_size, raw_batch_size, uncertainty_based_minibatch)
#
#         # Train each member on its mini-batch
#         train_outs_id, _ = ray.wait([self.ensemble[i].train.remote(batches[i]) for i in range(self.ensemble_size)],
#                                     num_returns=self.ensemble_size)
#         ac_en_q_loss, ac_en_q, ac_en_pi_loss = [], [], []
#         for i in range(self.ensemble_size):
#             ac_en_q_loss.append(ray.get(train_outs_id[0])[0])
#             ac_en_q.append(ray.get(train_outs_id[0])[1])
#             ac_en_pi_loss.append(ray.get(train_outs_id[0])[2])
#         return np.asarray(ac_en_q_loss), np.asarray(ac_en_q), np.asarray(ac_en_pi_loss)
#
#     def _generate_mini_batch(self, batch_size=100, raw_batch_size=500, uncertainty_based_minibatch=False):
#         random_mini_batches = [reply_buf.sample_batch(batch_size) for reply_buf in self.ensemble_replay_bufs]
#         return random_mini_batches
#
#     def add_to_replay_buffer(self, obs, act, rew, next_obs, done,
#                              step_index, epoch_index, time, **kwargs):
#         """Add experience to each Actor-Critic's replay buffer with probability replay_buf_bootstrapp_p"""
#         for ac_i in range(self.ensemble_size):
#             if np.random.uniform(0, 1, 1) < self.replay_buf_bootstrap_p:
#                 self.ensemble_replay_bufs[ac_i].store(obs, act, rew, next_obs, done,
#                                                       step_index, epoch_index, time, **kwargs)
# @ray.remote
# class ActorCritic():
#     """
#     Class used for create an Actor-Critic.
#     """
#     def __init__(self, ac_name,
#                  obs_dim, act_dim, act_limit, hidden_sizes, gamma, pi_lr, q_lr, polyak,
#                  replay_size, replay_buf_bootstrap_p, logger_kwargs):
#         self.act_name = ac_name
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self.act_limit = act_limit
#         self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
#         self.act_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.act_dim))
#         self.rew_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
#         self.new_obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
#         self.done_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
#
#         self.hidden_sizes = hidden_sizes
#         self.replay_size = replay_size
#         self.replay_buf_bootstrap_p = replay_buf_bootstrap_p
#         self.logger_kwargs = logger_kwargs
#
#         self.actor_hidden_activation = tf.keras.activations.relu
#         self.actor_output_activation = tf.keras.activations.tanh
#         self.critic_hidden_activation = tf.keras.activations.relu
#         self.critic_output_activation = tf.keras.activations.linear
#         self.gamma = gamma
#         self.pi_lr = pi_lr
#         self.q_lr = q_lr
#         self.polyak = polyak
#
#         # Actor and Critic
#         with tf.variable_scope('{}_main'.format(ac_name)):
#             self.actor = MLP(self.hidden_sizes + [self.act_dim],
#                              hidden_activation=self.actor_hidden_activation,
#                              output_activation=self.actor_output_activation)
#             self.critic = MLP(self.hidden_sizes + [1],
#                               hidden_activation=self.critic_hidden_activation,
#                               output_activation=self.critic_output_activation)
#             self.pi = self.act_limit * self.actor(self.obs_ph)
#             self.q = tf.squeeze(self.critic(tf.concat([self.obs_ph, self.act_ph], axis=-1)), axis=1)
#             self.q_pi = tf.squeeze(self.critic(tf.concat([self.obs_ph, self.pi], axis=-1)), axis=1)
#
#         # Target Actor and Target Critic
#         with tf.variable_scope('{}_target'.format(ac_name)):
#             self.actor_targ = MLP(self.hidden_sizes + [self.act_dim],
#                                   hidden_activation=self.actor_hidden_activation,
#                                   output_activation=self.actor_output_activation)
#             self.critic_targ = MLP(self.hidden_sizes + [1],
#                                    hidden_activation=self.critic_hidden_activation,
#                                    output_activation=self.critic_output_activation)
#             self.pi_targ = self.act_limit * self.actor_targ(self.new_obs_ph)
#             self.q_pi_targ = tf.squeeze(self.critic_targ(tf.concat([self.new_obs_ph, self.pi_targ], axis=-1)), axis=1)
#
#         # Create Replay Buffer
#         self.replay_buf = ReplayBuffer(self.obs_dim, self.act_dim, self.replay_size,
#                                        logger_fname='exp_log_{}.txt'.format(ac_name), **self.logger_kwargs)
#
#         # Loss
#         self.backup = tf.stop_gradient(self.rew_ph + self.gamma * (1 - self.done_ph) * self.q_pi_targ)
#         self.pi_loss = -tf.reduce_mean(self.q_pi)
#         self.q_loss = tf.reduce_mean((self.q - self.backup) ** 2)
#
#         # Optimization
#         self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.pi_lr)
#         self.q_optimizer = tf.train.AdamOptimizer(learning_rate=self.q_lr)
#         self.train_pi_op = self.pi_optimizer.minimize(self.pi_loss, var_list=self.actor.variables)
#         self.train_q_op = self.q_optimizer.minimize(self.q_loss, var_list=self.critic.variables)
#
#         # Update Target
#         self.target_update = tf.group([tf.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
#                                        for v_main, v_targ in zip(self.actor.variables + self.critic.variables,
#                                                                  self.actor_targ.variables + self.critic_targ.variables)])
#         # Initializing targets to match main variables
#         self.target_init = tf.group([tf.assign(v_targ, v_main)
#                                      for v_main, v_targ in zip(self.actor.variables + self.critic.variables,
#                                                                self.actor_targ.variables + self.critic_targ.variables)])
#
#         # Create tf.Session
#         self.sess = tf.Session()
#         self.sess.run(tf.global_variables_initializer())
#
#         # Initialize target
#         self.sess.run(self.target_init)
#
#     def train(self, batch_size=100):
#         # Generate mini-batch
#         batch = self.replay_buf.sample_batch(batch_size)
#         feed_dict = {self.obs_ph: batch['obs1'],
#                      self.act_ph: batch['acts'],
#                      self.rew_ph: batch['rews'],
#                      self.new_obs_ph: batch['obs2'],
#                      self.done_ph: batch['done']
#                      }
#
#         # Train critic
#         train_critic_ops = [self.q_loss, self.q, self.train_q_op]
#         critic_outs = self.sess.run(train_critic_ops, feed_dict=feed_dict)
#         ac_en_q_loss = critic_outs[0]
#         ac_en_q = critic_outs[1]
#         # Train actor
#         train_actor_ops = [self.pi_loss, self.train_pi_op, self.target_update]
#         actor_outs = self.sess.run(train_actor_ops, feed_dict=feed_dict)
#         ac_en_pi_loss = actor_outs[0]
#         return ac_en_q_loss, ac_en_q, ac_en_pi_loss
#
#     def predict(self, input):
#         feed_dict = {self.obs_ph: input.reshape(1, -1)}
#         return self.sess.run(self.pi, feed_dict=feed_dict)[0]
#
#     def add_to_replay_buffer(self, obs, act, rew, next_obs, done, step_index, epoch_index, time, kwargs):
#         self.replay_buf.store(obs, act, rew, next_obs, done, step_index, epoch_index, time, **kwargs)
#
#     def save_model(self):
#         # TODO
#         pass
#
# class BootstrappedActorCriticEnsemble():
#     def __init__(self,ensemble_size,
#                  obs_dim, act_dim, act_limit, hidden_sizes,
#                  gamma, pi_lr, q_lr, polyak,
#                  replay_size, replay_buf_bootstrap_p, logger_kwargs):
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self.act_limit = act_limit
#
#         self.hidden_sizes = hidden_sizes
#         self.replay_size = replay_size
#         self.replay_buf_bootstrap_p = replay_buf_bootstrap_p
#         self.logger_kwargs = logger_kwargs
#
#         self.gamma = gamma
#         self.pi_lr = pi_lr
#         self.q_lr = q_lr
#         self.polyak = polyak
#
#         # Create actor-critic ensemble
#         self.ensemble_size = ensemble_size
#         self.ensemble = [ActorCritic.remote('act_{}'.format(i),
#                                             self.obs_dim, self.act_dim, self.act_limit, self.hidden_sizes,
#                                             self.gamma, self.pi_lr, self.q_lr, self.polyak,
#                                             self.replay_size, self.replay_buf_bootstrap_p, self.logger_kwargs)
#                                   for i in range(self.ensemble_size)]
#
#     def predict(self, input):
#         predics_id, _ = ray.wait([self.ensemble[i].predict.remote(input) for i in range(self.ensemble_size)],
#                                  num_returns=self.ensemble_size)
#         predics = ray.get(predics_id)
#         return np.asarray(predics)
#
#     def train(self, batch_size=100, raw_batch_size=500, uncertainty_based_minibatch=False):
#         # Train each member on its mini-batch
#         train_outs_id, _ = ray.wait([self.ensemble[i].train.remote() for i in range(self.ensemble_size)],
#                                     num_returns=self.ensemble_size)
#         ac_en_q_loss, ac_en_q, ac_en_pi_loss = [], [], []
#         for i in range(self.ensemble_size):
#             ac_en_q_loss.append(ray.get(train_outs_id[0])[0])
#             ac_en_q.append(ray.get(train_outs_id[0])[1])
#             ac_en_pi_loss.append(ray.get(train_outs_id[0])[2])
#         return np.asarray(ac_en_q_loss), np.asarray(ac_en_q), np.asarray(ac_en_pi_loss)
#
#     def add_to_replay_buffer(self, obs, act, rew, next_obs, done,
#                              step_index, epoch_index, time, **kwargs):
#         """Add experience to each Actor-Critic's replay buffer with probability replay_buf_bootstrapp_p"""
#         for ac_i in range(self.ensemble_size):
#             if np.random.uniform(0, 1, 1) < self.replay_buf_bootstrap_p:
#                 self.ensemble[ac_i].add_to_replay_buffer.remote(obs, act, rew, next_obs, done, step_index, epoch_index, time, kwargs)



# class BootstrappedActorCriticEnsemble():
#     def __init__(self, ensemble_size,
#                  obs_dim, act_dim, act_limit, hidden_sizes,
#                  gamma, pi_lr, q_lr, polyak,
#                  replay_size, replay_buf_bootstrap_p, logger_kwargs):
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self.act_limit = act_limit
#         self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
#         self.act_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.act_dim))
#         self.rew_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
#         self.new_obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
#         self.done_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
#
#         self.hidden_sizes = hidden_sizes
#         self.replay_size = replay_size
#         self.replay_buf_bootstrap_p = replay_buf_bootstrap_p
#         self.logger_kwargs = logger_kwargs
#
#         self.ensemble_size = ensemble_size
#         self.ensemble = []
#
#         self.actor_hidden_activation = tf.keras.activations.relu
#         self.actor_output_activation = tf.keras.activations.tanh
#         self.critic_hidden_activation = tf.keras.activations.relu
#         self.critic_output_activation = tf.keras.activations.linear
#         self.gamma = gamma
#         self.pi_lr = pi_lr
#         self.q_lr = q_lr
#         self.polyak = polyak
#
#         # Create ensemble
#         for ac_i in range(self.ensemble_size):
#             self.ensemble.append(self._create_actor_critic(ac_i))
#
#         # Create tf.Session
#         self.sess = tf.Session()
#         self.sess.run(tf.global_variables_initializer())
#
#         # Initialize target
#         for ac_i in range(self.ensemble_size):
#             self.sess.run(self.ensemble[ac_i]['target_init'])
#
#     def _create_actor_critic(self, ac_i):
#         """Create actor-critic"""
#         # Replay_Buffer
#         replay_buf = ReplayBuffer(self.obs_dim, self.act_dim, self.replay_size,
#                                   logger_fname='exp_log_ac_{}.txt'.format(ac_i), **self.logger_kwargs)
#
#         # Actor and Critic
#         with tf.variable_scope('{}_main'.format(ac_i)):
#             actor = MLP(self.hidden_sizes + [self.act_dim],
#                         hidden_activation=self.actor_hidden_activation,
#                         output_activation=self.actor_output_activation)
#             critic = MLP(self.hidden_sizes + [1],
#                          hidden_activation=self.critic_hidden_activation,
#                          output_activation=self.critic_output_activation)
#             pi = self.act_limit * actor(self.obs_ph)
#             q = tf.squeeze(critic(tf.concat([self.obs_ph, self.act_ph], axis=-1)), axis=1)
#             q_pi = tf.squeeze(critic(tf.concat([self.obs_ph, pi], axis=-1)), axis=1)
#
#         # Target Actor and Target Critic
#         with tf.variable_scope('{}_target'.format(ac_i)):
#             actor_targ = MLP(self.hidden_sizes + [self.act_dim],
#                              hidden_activation=self.actor_hidden_activation,
#                              output_activation=self.actor_output_activation)
#             critic_targ = MLP(self.hidden_sizes + [1],
#                               hidden_activation=self.critic_hidden_activation,
#                               output_activation=self.critic_output_activation)
#             pi_targ = self.act_limit * actor_targ(self.new_obs_ph)
#             q_pi_targ = tf.squeeze(critic_targ(tf.concat([self.new_obs_ph, pi_targ], axis=-1)), axis=1)
#
#         # Loss
#         backup = tf.stop_gradient(self.rew_ph + self.gamma * (1 - self.done_ph) * q_pi_targ)
#         pi_loss = -tf.reduce_mean(q_pi)
#         q_loss = tf.reduce_mean((q - backup) ** 2)
#
#         # Optimization
#         pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.pi_lr)
#         q_optimizer = tf.train.AdamOptimizer(learning_rate=self.q_lr)
#         train_pi_op = pi_optimizer.minimize(pi_loss, var_list=actor.variables)
#         train_q_op = q_optimizer.minimize(q_loss, var_list=critic.variables)
#
#         # Update Target
#         target_update = tf.group([tf.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
#                                   for v_main, v_targ in zip(actor.variables + critic.variables,
#                                                             actor_targ.variables + critic_targ.variables)])
#         # Initializing targets to match main variables
#         target_init = tf.group([tf.assign(v_targ, v_main)
#                                 for v_main, v_targ in zip(actor.variables + critic.variables,
#                                                           actor_targ.variables + critic_targ.variables)])
#         return dict(replay_buf=replay_buf,
#                     actor=actor, critic=critic, pi=pi, q=q, q_pi=q_pi,
#                     actor_targ=actor_targ, critic_targ=critic_targ,
#                     pi_targ=pi_targ, q_pi_targ=q_pi_targ,
#                     backup=backup, pi_loss=pi_loss, q_loss=q_loss,
#                     train_pi_op=train_pi_op, train_q_op=train_q_op,
#                     target_update=target_update, target_init=target_init)
#
#     def add_to_replay_buffer(self, obs, act, rew, next_obs, done,
#                              step_index, epoch_index, time, **kwargs):
#         """Add experience to each Actor-Critic's replay buffer with probability replay_buf_bootstrapp_p"""
#         for ac_i in range(self.ensemble_size):
#             if np.random.uniform(0, 1, 1) < self.replay_buf_bootstrap_p:
#                 self.ensemble[ac_i]['replay_buf'].store(obs, act, rew, next_obs, done,
#                                                         step_index, epoch_index, time, **kwargs)
#
#     def prediction(self, input):
#         """Prediction of actor-critic ensemble"""
#         feed_dict = {self.obs_ph: input.reshape(1, -1)}
#         preds = np.zeros((self.ensemble_size, self.act_dim))
#         for ac_i in range(self.ensemble_size):
#             preds[ac_i, :] = self.sess.run(self.ensemble[ac_i]['pi'], feed_dict=feed_dict)
#         return preds
#
#     def train(self, batch_size=100, raw_batch_size=500, uncertainty_based_minibatch=False):
#         """Train each actor-critic on its corresponding replay_buffer"""
#         ac_en_q_loss = np.zeros((self.ensemble_size,))
#         ac_en_q = np.zeros((self.ensemble_size, batch_size))
#         ac_en_pi_loss = np.zeros((self.ensemble_size,))
#         for ac_i in range(self.ensemble_size):
#             if uncertainty_based_minibatch:
#                 # Select top n highest uncertainty samples
#                 # TODO
#                 pass
#             else:
#                 batch = self.ensemble[ac_i]['replay_buf'].sample_batch(batch_size)
#
#             feed_dict = {self.obs_ph: batch['obs1'],
#                          self.act_ph: batch['acts'],
#                          self.rew_ph: batch['rews'],
#                          self.new_obs_ph: batch['obs2'],
#                          self.done_ph: batch['done']
#                          }
#             # Train critic
#             train_critic_ops = [self.ensemble[ac_i]['q_loss'],
#                                 self.ensemble[ac_i]['q'], self.ensemble[ac_i]['train_q_op']]
#             critic_outs = self.sess.run(train_critic_ops, feed_dict=feed_dict)
#             ac_en_q_loss[ac_i] = critic_outs[0]
#             ac_en_q[ac_i,:] = critic_outs[1]
#             # Train actor
#             train_actor_ops = [self.ensemble[ac_i]['pi_loss'],
#                                self.ensemble[ac_i]['train_pi_op'], self.ensemble[ac_i]['target_update']]
#             actor_outs = self.sess.run(train_actor_ops, feed_dict=feed_dict)
#             ac_en_pi_loss[ac_i] = actor_outs[0]
#         return ac_en_q_loss, ac_en_q, ac_en_pi_loss
#
#     def save_model(self):
#         # TODO: save model
#         pass

# class BootstrappedEnsemble():
#     """
#     Bootstrapped Ensemble:
#         a set of MLP are trained with their corresponding bootstrapped training set.
#     """
#     def __init__(self, ensemble_size = 10, x_dim=5, y_dim=2, replay_size=1e6,
#                  x_ph=None, y_ph=None, layer_sizes=[], kernel_regularizer=None,
#                  hidden_activation=tf.keras.activations.relu,
#                  output_activation=tf.keras.activations.sigmoid, learning_rate=1e-3):
#         self.ensemble_size = ensemble_size
#         self.x_dim = x_dim
#         self.y_dim = y_dim
#         self.x_ph = x_ph
#         self.y_ph = y_ph
#
#         self.replay_buffers = []
#         self.ensemble = []
#         self.ensemble_preds = []
#         self.ensemble_losses = []
#         self.ensemble_train_ops = []
#
#         for ens_i in range(self.ensemble_size):
#             self.replay_buffers.append(ReplayBuffer(x_dim=x_dim, y_dim=y_dim, size=replay_size))
#             self.ensemble.append(MLP(layer_sizes, kernel_regularizer=kernel_regularizer,
#                                      hidden_activation=hidden_activation,
#                                      output_activation=output_activation))
#             self.ensemble_preds.append(self.ensemble[ens_i](self.x_ph))
#             # mean-square-error
#             self.ensemble_losses.append(tf.reduce_mean((self.y_ph - self.ensemble_preds[ens_i]) ** 2))
#             self.ensemble_train_ops.append(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
#                 self.ensemble_losses[ens_i],
#                 var_list=self.ensemble[ens_i].variables))
#
#     def prediction(self, sess, input):
#         # predict
#         feed_dict = {self.x_ph: input.reshape(-1, self.x_dim)}
#         preds = np.zeros((self.ensemble_size, self.y_dim))
#         for i in range(self.ensemble_size):
#             preds[i,:] = sess.run(self.ensemble_preds[i], feed_dict=feed_dict)
#         return preds
#
#     def add_to_replay_buffer(self, input, label, bootstrapp_p=0.75):
#         # add to replay buffer
#         for i in range(self.ensemble_size):
#             if np.random.uniform(0, 1, 1) < bootstrapp_p:  # with bootstrapp_p probability to add to replay buffer
#                 self.replay_buffers[i].store(input, label)
#
#     def train(self, sess, raw_batch_size, batch_size, uncertainty_based_minibatch_sample):
#         loss = np.zeros((self.ensemble_size,))
#         for i in range(self.ensemble_size):
#             if uncertainty_based_minibatch_sample:
#                 # Select top n highest uncertainty samples
#                 raw_batch = self.replay_buffers[i].sample_batch(raw_batch_size)
#                 batch = self.resample_based_on_uncertainty(sess, raw_batch, raw_batch_size, batch_size)
#             else:
#                 batch = self.replay_buffers[i].sample_batch(batch_size)
#             # Train on the batch
#             out = sess.run([self.ensemble_losses[i], self.ensemble_train_ops[i]], feed_dict={self.x_ph: batch['x'], self.y_ph: batch['y']})
#             loss[i] = out[0]
#         return loss
#
#     def resample_based_on_uncertainty(self, sess, raw_batch, raw_batch_size, batch_size):
#         ensemble_postSample = sess.run(self.ensemble_preds, feed_dict={self.x_ph: raw_batch['x'], self.y_ph: raw_batch['y']})
#         ensemble_postSample = np.reshape(np.concatenate(ensemble_postSample, axis=1), (-1, self.ensemble_size, self.y_dim))
#         uncertainty_rank = np.zeros((raw_batch_size,))
#         for i_batch in range(ensemble_postSample.shape[0]):
#             uncertainty_rank[i_batch] = np.sum(np.diag(np.cov(ensemble_postSample[i_batch], rowvar=False)))
#         # Find top_n highest uncertainty samples
#         top_n_highest_uncertainty_indices = np.argsort(uncertainty_rank)[-batch_size:]
#         batch = {}
#         batch['x'] = raw_batch['x'][top_n_highest_uncertainty_indices]
#         batch['y'] = raw_batch['y'][top_n_highest_uncertainty_indices]
#         return batch
