import numpy as np
import time
from spinup.utils.logx import EpochLogger, Logger
from collections import deque, namedtuple
import random

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size,
                 logger_fname='experiences_log.txt', **logger_kwargs):
        # # ExperienceLogger: save experiences for supervised learning
        # logger_kwargs['output_fname'] = logger_fname
        # self.experience_logger = Logger(**logger_kwargs)
        self.Experience = namedtuple("Experience", ["obs", "act", "rew", "next_obs", "done"])
        self.experience_buf = []
        self.size, self.max_size = 0, size

    def store(self, obs, act, rew, next_obs, done,
              step_index, steps_per_epoch, start_time, **kwargs):
        # # Save experiences in disk
        # self.log_experiences(obs, act, rew, next_obs, done,
        #                      step_index, steps_per_epoch, start_time, **kwargs)
        if self.size == self.max_size:
            self.experience_buf.pop(0)
        self.experience_buf.append(self.Experience(obs, act, rew, next_obs, done))
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        obs1_batch, acts_batch, rews_batch, obs2_batch, done_batch = [], [], [], [], []
        samples = random.sample(self.experience_buf, batch_size)
        for entry in samples:
            obs1_batch.append(np.asarray(entry[0]))  # obs1
            acts_batch.append(entry[1])              # act
            rews_batch.append(entry[2])              # rew
            obs2_batch.append(np.asarray(entry[3]))  # obs2
            done_batch.append(entry[4])              # done
        return dict(obs1=np.asarray(obs1_batch).squeeze(),
                    acts=np.asarray(acts_batch).squeeze(),
                    rews=np.asarray(rews_batch).squeeze(),
                    obs2=np.asarray(obs2_batch).squeeze(),
                    done=np.asarray(done_batch).squeeze())

    # def log_experiences(self, obs, act, rew, next_obs, done,
    #                     step_index, steps_per_epoch, start_time, **kwargs):
    #     self.experience_logger.log_tabular('Epoch', step_index // steps_per_epoch)
    #     self.experience_logger.log_tabular('Step', step_index)
    #     # Log observation
    #     for i, o_i in enumerate(obs):
    #         self.experience_logger.log_tabular('o_{}'.format(i), o_i)
    #     # Log action
    #     for i, a_i in enumerate(act):
    #         self.experience_logger.log_tabular('a_{}'.format(i), a_i)
    #     # Log reward
    #     self.experience_logger.log_tabular('r', rew)
    #     # Log next observation
    #     for i, o2_i in enumerate(next_obs):
    #         self.experience_logger.log_tabular('o2_{}'.format(i), o2_i)
    #     # Log other data
    #     for key, value in kwargs.items():
    #         for i, v in enumerate(np.array(value).flatten(order='C')):
    #             self.experience_logger.log_tabular('{}_{}'.format(key, i), v)
    #     # Log done
    #     self.experience_logger.log_tabular('d', done)
    #     self.experience_logger.log_tabular('Time', time.time() - start_time)
    #     self.experience_logger.dump_tabular(print_data=False)

class RandomNetReplayBuffer:
    """
    A simple FIFO experience replay buffer.
    """

    def __init__(self, x_dim, y_dim, size):
        self.x_buf = np.zeros([size, x_dim], dtype=np.float32)
        self.y_buf = np.zeros([size, y_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, x, y):
        self.x_buf[self.ptr] = x
        self.y_buf[self.ptr] = y
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        if batch_size > self.size:
            batch_size = self.size
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(x=self.x_buf[idxs],
                    y=self.y_buf[idxs])