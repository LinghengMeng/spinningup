import gym

env = gym.make('HalfCheetah-v2')
# Manipulate environment
# Change gravity
env.model.opt.gravity[0] = 0        # gravity: "0 0 -9.81"
env.model.opt.gravity[1] = 0        # gravity: "0 0 -9.81"
env.model.opt.gravity[2] = -9.81    # gravity: "0 0 -9.81"

# # Change wind
# env.model.opt.density = 1.2         # density of air: 1.2, density of water: 1000 (Setting density to 0 disables lift and drag forces.)
# env.model.opt.wind[0] = 0           # wind: "0 0 0"
# env.model.opt.wind[1] = 0           # wind: "0 0 0"
# env.model.opt.wind[2] = 0           # wind: "0 0 0"
#
# env.model.opt.viscosity = 0         # viscosity of air is around 0.00002 while the viscosity of water is around 0.0009 depending on temperature. (Setting viscosity to 0 disables viscous forces.)
obs = env.reset()
i = 0
while True:
    if i % 100 == 0:
        env.model.opt.gravity[2] = env.model.opt.gravity[2]+0.01
    print('gravity={}'.format(env.model.opt.gravity[2]))
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()


