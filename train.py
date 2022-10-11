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
        N_ID = 1
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
        env = Arm_env(max_step = MAX_STEP * 5, is_render=False)

        model = PPO.load(log_path + "best_model", env)
        obs = env.reset()
        for i in range(10000):

            a, _ = model.predict(obs)
            print(f'action {a}')
            obs, reward, done, _ = env.step(a)
            # print(f'ee_pos {obs[:3]}')
            # print(f'box_pos {obs[6:9]}')
            print(reward)

            if done == True:
                print("fail")
                obs = env.reset()
                # break




