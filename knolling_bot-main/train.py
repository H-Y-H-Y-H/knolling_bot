from knolling_env import Arm_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import pathlib

current_file_name = pathlib.Path(__file__).stem
log_dir = "models/" + current_file_name
os.makedirs(log_dir, exist_ok=True)

# Create environment
env=Arm_env(is_render=False, num_objects=1)

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
# Train the agent
model.learn(total_timesteps=50000)
# Save the agent
model.save("PPO_knolling")

