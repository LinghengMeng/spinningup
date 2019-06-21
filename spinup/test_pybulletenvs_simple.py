import pybulletgym
import mujoco_py
import gym

env_name = 'HopperMuJoCoEnv-v0'#'AntMuJoCotEnv-v0'#'InvertedPendulumMuJoCoEnv-v0'#'PusherPyBulletEnv-v0' #'HumanoidPyBulletEnv-v0'#'InvertedPendulumPyBulletEnv-v0'#''HumanoidPyBulletEnv-v0'#'HopperPyBulletEnv-v0' #'AntBulletEnv-v0'
env = gym.make(env_name)

env.render()
env.reset()

for i in range(10000):
    obs, rewards, done, _ = env.step(env.action_space.sample())


