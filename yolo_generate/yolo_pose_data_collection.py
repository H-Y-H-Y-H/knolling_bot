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
from tqdm import tqdm

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

    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_3/'
    os.makedirs(data_root, exist_ok=True)

    p.connect(p.GUI)

    startnum = 0
    endnum = 10
    lebal_list = []

    num_reset = True
    CLOSE_FLAG = False
    texture_flag = True
    max_lego_num = 12
    mm2px = 530 / 0.34

    for epoch in tqdm(range(startnum,endnum)):
        # num_item = random.randint(1, 5)

        num_item = int(np.random.uniform(4, max_lego_num + 1))
        boxes_index = np.random.choice(50, num_item)

        env = Arm_env(max_step=1, is_render=False, boxes_index=boxes_index)
        state, lw_list = env.reset_table(close_flag=CLOSE_FLAG, texture_flag=texture_flag)

        label = np.zeros((num_item, 6))

        all_pos = state[6:6+3*num_item]
        all_ori = state[6+3*num_item: 6+6*num_item]


        corner_list = []
        for j in range(num_item):
            if j >= num_item:
                element = np.zeros(6)
                # element = np.append(element, 0)
                label.append(element)
            else:
                xpos1 = all_pos[0+3*j]
                ypos1 = all_pos[1+3*j]
                yawori = all_ori[2+3*j]
                l = lw_list[j, 0]
                w = lw_list[j, 1]
                element = np.array([1, xpos1, ypos1, l, w, yawori])

            label[j] = element
        my_im2 = env.get_image()[:, :, :3]
        temp = np.copy(my_im2[:, :, 0])  # change rgb image to bgr for opencv to save
        my_im2[:, :, 0] = my_im2[:, :, 2]
        my_im2[:, :, 2] = temp
        # add = int((640 - 480) / 2)
        # img = cv2.copyMakeBorder(my_im2, add, add, 0, 0, cv2.BORDER_CONSTANT, None, value=(0, 0, 0, 255))

        cv2.imwrite(os.path.join(data_root, 'images/%012d.png') % epoch, my_im2)

        np.savetxt(os.path.join(data_root, "labels/%012d.txt" % epoch), label)
