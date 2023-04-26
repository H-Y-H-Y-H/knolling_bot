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
from yolo_pose_data_collection_env import *
# from create_urdf import *
# from create_urdf import *

def sort_cube(obj_list, x_sorted):
    s = np.array(x_sorted)
    sort_index = np.argsort(s)
    # print(sort_index)
    sorted_sas_list = [obj_list[j] for j in sort_index]
    return sorted_sas_list

def yolo_box(img, label):
    # label = [0,x,y,l,w],[0,x,y,l,w],...
    # label = label[:,1:]
    for i in range(len(label)):
        # label = label[i]
        # print('1',label)
        x_lt = int(label[i][1] * 640 - label[i][3] * 640/2)
        y_lt = int(label[i][2] * 640 - label[i][4] * 640/2)

        x_rb = int(label[i][1] * 640 + label[i][3] * 640/2)
        y_rb = int(label[i][2] * 640 + label[i][4] * 640/2)

        # img = img/255
        img = cv2.rectangle(img,(x_lt,y_lt),(x_rb,y_rb), color = (0,0,0), thickness = 1)
    # cv2.imshow('x',img)
    # cv2.waitKey(0)

    return img

if __name__ == '__main__':

    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_small/'

    p.connect(p.GUI)

    startnum = 0
    endnum = 100
    lebal_list = []

    num_reset = True
    CLOSE_FLAG = False
    max_lego_num = 15
    mm2px = 530 / 0.34

    for epoch in range(startnum,endnum):
        # num_item = random.randint(1, 5)

        num_item = int(np.random.uniform(2, max_lego_num + 1))
        # num_item = 3

        env = Arm_env(max_step=1, is_render=False, num_objects=num_item)
        state, rdm_pos_x, rdm_pos_y, rdm_pos_z, rdm_ori_yaw, lucky_list = env.reset_table(close_flag=CLOSE_FLAG)

        label = np.zeros((num_item, 6))

        all_pos = state[6:6+3*num_item]
        all_ori = state[6+3*num_item: 6+6*num_item]


        corner_list = []
        for j in range(num_item):
            if j >= num_item:
                element = np.zeros(7)
                # element = np.append(element, 0)
                label.append(element)
            else:
                xpos1 = all_pos[0+3*j]
                ypos1 = all_pos[1+3*j]
                yawori = all_ori[2+3*j]

                if lucky_list[j] == 0:
                    l = 16/1000
                    w = 16/1000
                if lucky_list[j] == 1:
                    l = 24/1000
                    w = 16/1000
                if lucky_list[j] == 2:
                    l = 32/1000
                    w = 16/1000

                # if xpos1 > 0.29 or ypos1 < -0.2 or ypos1 > 0.2:
                #     print('???')
                element = np.array([1, xpos1, ypos1, l, w, yawori])

            label[j] = element
        my_im2 = env.get_image()
        # add = int((640 - 480) / 2)
        # img = cv2.copyMakeBorder(my_im2, add, add, 0, 0, cv2.BORDER_CONSTANT, None, value=(0, 0, 0, 255))

        cv2.imwrite(os.path.join(data_root, 'images/%012d.png') % epoch, my_im2)

        np.savetxt(os.path.join(data_root, "labels/%012d.txt" % epoch), label)
