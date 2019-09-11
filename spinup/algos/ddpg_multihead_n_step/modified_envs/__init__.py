from gym.envs.registration import register

register(
    id='HalfCheetahFriction-v2',
    entry_point='spinup.algos.ddpg_n_step.modified_envs.mujoco.half_cheetah:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='Walker2dFriction-v2',
    max_episode_steps=1000,
    entry_point='spinup.algos.ddpg_n_step.modified_envs.mujoco.walker2d:Walker2dEnv',
)

register(
    id='HalfCheetahFriction-v3',
    max_episode_steps=1000,
    entry_point='spinup.algos.ddpg_n_step.modified_envs.mujoco.half_cheetah_v3:HalfCheetahEnv'
)

register(
    id='HalfCheetahGravity-v3',
    max_episode_steps=1000,
    entry_point='spinup.algos.ddpg_n_step.modified_envs.mujoco.half_cheetah_v3:HalfCheetahEnv'
)