import time
import pybullet as p
import pybullet_data as pd
import os
import gym
import cv2
import numpy as np
import random
import math
from PIL import Image
from knolling_env import *


def sort_cube(obj_list, x_sorted):
    s = np.array(x_sorted)
    sort_index = np.argsort(s)
    # print(sort_index)
    sorted_sas_list = [obj_list[j] for j in sort_index]
    return sorted_sas_list


if __name__ == '__main__':

    p.connect(1)

    startnum = 0
    endnum = 1
    lebal_list = []
    reset_flag = True
    num_reset = True
    count_item = 0
    num_item = 0
    for epoch in range(startnum, endnum):

        # path = "urdf/box/"

        if epoch % 1 == 0:
            # count_item = 0
            reset_flag = True

        if reset_flag == True:
            # num_item = random.randint(1, 5)
            num_item = 3
            env = Arm_env(max_step=1, is_render=False, num_objects=num_item)
            state, rdm_pos_x, rdm_pos_y, rdm_pos_z, rdm_ori_yaw, lucky_list = env.reset()

        # time.sleep(10)
        # if reset_flag != True:
        #
        #     while True:
        #         rdm_pos_x = np.random.uniform(env.x_low_obs * 2.2, env.x_high_obs, size=3)
        #         rdm_pos_y = np.random.uniform(env.y_low_obs, env.y_high_obs, size=3)
        #
        #         dis1_2 = math.dist([rdm_pos_x[0], rdm_pos_y[0]], [rdm_pos_x[1], rdm_pos_y[1]])
        #         dis1_3 = math.dist([rdm_pos_x[0], rdm_pos_y[0]], [rdm_pos_x[2], rdm_pos_y[2]])
        #         dis2_3 = math.dist([rdm_pos_x[1], rdm_pos_y[1]], [rdm_pos_x[2], rdm_pos_y[2]])
        #
        #         if dis1_2 > 0.03 and dis1_3 > 0.03 and dis2_3 > 0.03:
        #             break
        #
        #     rdm_pos_z = 0.01
        #     rdm_ori_yaw = np.random.uniform(0, math.pi / 2, size=3)
        #
        #
        #
        #     for i in range(num_item):
        #         # print("num",num_item)
        #         # print(len(env.obj_idx))
        #         # p.removeBody(env.obj_idx[i])
        #
        #         p.resetBasePositionAndOrientation(env.obj_idx[i], posObj=[rdm_pos_x[i],rdm_pos_y[i],0.01],
        #                                    ornObj=p.getQuaternionFromEuler([0,0,rdm_ori_yaw[i]]))
        #
        #     state = env.get_obs()

        # self.obs = np.concatenate([self.ee_pos, self.ee_ori, self.box_pos, self.box_ori, self.joints_angle])
        #
        #                                  3             3          3*N           3*N            7
        # reset_flag = False
        label = []
        # print(state)
        all_pos = state[6:6 + 3 * num_item]
        all_ori = state[6 + 3 * num_item: 6 + 6 * num_item]
        print(all_pos)
        print(all_ori)

        for j in range(3):
            if j >= num_item:
                element = np.zeros(7)
                # element = np.append(element, 0)
                label.append(element)
            else:
                xpos = all_pos[0 + 3 * j]
                ypos = all_pos[1 + 3 * j]
                yawori = all_ori[2 + 3 * j]

                corn1, corn2, corn3, corn4 = find_corner(
                    xpos, ypos, lucky_list[j], yawori)

                element = []
                # element.append(rdm_pos_x[j])
                # element.append(rdm_pos_y[j])
                # element.append(rdm_pos_z)
                # element.append(0)
                # element.append(0)
                # element.append(rdm_ori_yaw[j])
                # element.append(1)
                element.append(corn1[0])
                element.append(corn1[1])
                element.append(corn2[0])
                element.append(corn2[1])
                element.append(corn3[0])
                element.append(corn3[1])
                element.append(corn4[0])
                element.append(corn4[1])
                element.append(1)
                element = np.asarray(element)
                label.append(element)

        x_compare = [label[0][0], label[1][0], label[2][0]]
        idx_cmp = np.argsort(x_compare)
        # print(idx_cmp)
        # print(x_compare)
        # print(x_compare)
        label2 = sort_cube(label, x_compare)
        # print(label2)

        obj_list = np.concatenate([label2[0], label2[1], label2[2]])
        # print(len(obj_list))

        lebal_list.append(obj_list)
        img = env.get_image()
        # time.sleep(1)
        for _ in range(20000):
            p.stepSimulation()
            time.sleep(1 / 240)
        # num = startnum // 10000
        img.save("Dataset/lego_data_test5/" + "IMG%s.png" % epoch)
        # p.disconnect()
        # p.resetSimulation()
        # time.sleep(0.5)
    np.savetxt("Dataset/label/label_lego_test5.csv", lebal_list)
