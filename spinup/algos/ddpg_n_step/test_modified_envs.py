import gym
import spinup.algos.ddpg_n_step.modified_envs

# env=gym.make('HalfCheetahFriction-v2')
env=gym.make('Walker2dFriction-v2')
# env=gym.make('HalfCheetahFriction-v3')
env=gym.make('HumanoidStandup-v2')
o = env.reset()

for i in range(10000):
    a = env.action_space.sample()
    o, r, d, _ = env.step(a)
    print(r)
    env.render()
    if d:
        env.reset()