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

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    MAX_STEP = 6

    mode = 0
    if mode == 0:

        current_file_name = pathlib.Path(__file__).stem
        log_dir = "models/" + current_file_name
        os.makedirs(log_dir, exist_ok=True)

        env = Arm_env(max_step = MAX_STEP, is_render=False)
        env.slep_t = 0

        # model = PPO.load(log_path + "best_model", env)
        model = PPO("MlpPolicy", env,n_steps=MAX_STEP, verbose = 0)

        num_epoch = 10000
        num_steps = MAX_STEP * 100
        best_r = -np.inf

        mean_reward_list = []
        std_reward_list = []
        N_ID = 2
        model_save_path = log_dir+"run%d"%N_ID
        try:
            os.mkdir(model_save_path)
        except OSError:
            pass

        for epoch in range(num_epoch):
            print(f'epoch:{epoch}')
            model.learn(total_timesteps=num_steps)
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3)
            mean_reward_list.append(mean_reward)
            std_reward_list.append(std_reward)
            print("start evaluation",mean_reward)
            if best_r < mean_reward:

                best_r = mean_reward
                model.save(model_save_path + "/best_model")

                np.savetxt(model_save_path+'mean_r.csv',np.asarray(mean_reward_list))
                np.savetxt(model_save_path+'std_r.csv',np.asarray(std_reward_list))

        print('finished')

    if mode == 1:
        N_ID = 0
        log_path= "models/trainrun%d/"%N_ID
        env = Arm_env(max_step = MAX_STEP, is_render=True)

        model = PPO.load(log_path + "best_model", env)
        obs = env.reset()
        for i in range(10000):

            a, _ = model.predict(obs)
            obs, reward, done, _ = env.step(a)

            if done == True:
                print("fail")
                obs = env.reset()
                # break




