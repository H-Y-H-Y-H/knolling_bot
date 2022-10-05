from knolling_env import Arm_env
from stable_baselines3.common.env_checker import check_env

env=Arm_env(is_render=True, num_objects=1)
# assert check_env(env)==None
assert env.observation_space.shape==(19,)
# assert env.action_space.shape==(3,)

obs=env.reset()
for _ in range(1000):
    action=env.action_space.sample()
    print(action)
    state_, reward, done, _ =env.step(action)
    print(reward)
    if done:
        obs=env.reset()