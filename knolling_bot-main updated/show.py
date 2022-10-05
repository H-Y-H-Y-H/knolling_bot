from knolling_env import Arm_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import pathlib

env=Arm_env(is_render = True, is_good_view= True, num_objects=1)

model = PPO.load("best_model")

obs = env.reset()

print(obs)
# env.render()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break