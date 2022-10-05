from knolling_env import Arm_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,VecEnv
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True

def make_env(seed):
    def _init():
        env = Arm_env(is_render=False)
        env.seed(seed)
        return env

    return _init

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    
    run()

    current_file_name = pathlib.Path(__file__).stem
    log_dir = "models/" + current_file_name
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env=SubprocVecEnv([make_env(None) for _ in range(2)])
    env=VecMonitor(env,log_dir)
    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    # Train the agent
    callback = SaveOnBestTrainingRewardCallback(check_freq=50, log_dir=log_dir)
    model.learn(total_timesteps=50000, callback=callback)
    # Save the agent
    # model.save("PPO_knolling")
    print('finished')


