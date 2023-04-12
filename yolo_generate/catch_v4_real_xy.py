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
from knolling_env3_real_xy import *

# np.random.seed(100)
# random.seed(100)

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

    return img

if __name__ == '__main__':


    # p.connect(p.DIRECT)
    p.connect(p.GUI)

    startnum = 2000
    endnum = 4000
    lebal_list = []
    reset_flag = True
    num_reset = True

    mm2px = 530 / 0.34 # (1558)
    for epoch in range(startnum,endnum):
        # num_item = random.randint(1, 5)
        num_item = 15
        # path = "urdf/box/"
        # num_item = 4

        env = Arm_env(max_step=1, is_render=False, num_objects=num_item)
        if random.random() < 0.5:
            state, rdm_pos_x, rdm_pos_y, rdm_pos_z, rdm_ori_yaw, lucky_list = env.reset_table(close_flag=True)
        else:
            state, rdm_pos_x, rdm_pos_y, rdm_pos_z, rdm_ori_yaw, lucky_list = env.reset_table(close_flag=False)

        # self.obs = np.concatenate([self.ee_pos, self.ee_ori, self.box_pos, self.box_ori, self.joints_angle])
        #
        #                                  3             3          3*N           3*N            7
        # reset_flag = False
        label = []
        # print(state)
        all_pos = state[6:6+3*num_item]
        all_ori = state[6+3*num_item: 6+6*num_item]
        # print(all_pos)
        # print(all_ori)


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

                # xpos, ypos = xyz_resolve(xpos1,ypos1)

                corn1, corn2, corn3, corn4 = find_corner(xpos1, ypos1, lucky_list[j], yawori)
                # print('this is corn after find corner', corn1, corn2, corn3, corn4)

                # corn1, corn2, corn3, corn4 = resolve_img(corn1, corn2, corn3, corn4)
                # print('this is corn after resolve img', corn1, corn2, corn3, corn4)

                corner_list.append([corn1, corn2, corn3, corn4])

                corns = corner_list[j]

                col_offset = 320
                # row_offset = (0.15 - (0.3112 - 0.15)) * mm2px + 5
                row_offset = 0

                col_list = [int(mm2px * corns[0][1] + col_offset), int(mm2px * corns[3][1] + col_offset),
                            int(mm2px * corns[1][1] + col_offset), int(mm2px * corns[2][1] + col_offset)]
                row_list = [int(mm2px * corns[0][0] - row_offset), int(mm2px * corns[3][0] - row_offset),
                            int(mm2px * corns[1][0] - row_offset), int(mm2px * corns[2][0] - row_offset)]
                # print(col_list)
                # print(row_list)

                col_list = np.sort(col_list)
                row_list = np.sort(row_list)
                col_list[3] = col_list[3] + 7
                col_list[0] = col_list[0] - 7

                row_list[3] = row_list[3] + 7
                row_list[0] = row_list[0] - 7

                label_x = ((col_list[0] + col_list[3]) / 2)/640
                label_y = (((row_list[0] + row_list[3]) / 2)+86)/640

                length = (col_list[3] - col_list[0])/640
                width = (row_list[3] - row_list[0])/640

                # if lucky_list[j] == 2 and rdm_ori_yaw[j] < 0:
                #     rdm_ori_yaw[j] = rdm_ori_yaw[j] + np.pi/2


                element = []
                element.append(xpos1)
                element.append(ypos1)
                element.append(lucky_list[j])
                element.append(yawori)
                element = np.asarray(element)
                label.append(element)

                # element = []
                # element.append(0)
                # element.append(label_x)
                # element.append(label_y)
                # element.append(length)
                # element.append(width)
                # element = np.asarray(element)
                # label.append(element)

        # print(label)
        np.savetxt("../YOLO_data/Label/real_world_label_409/img%s.txt" %epoch, label,fmt='%.8s')

        # for nnn in range(1):
        lebal_list.append(element)
        my_im2 = env.get_image()

        # ratio = 34 / 30
        # x_ratio = 0.975
        # y_ratio = 480 * x_ratio * ratio / 640
        # print(int((640 - 640 * y_ratio) / 2), int((480 - 480 * x_ratio) / 2))
        # print(int((640 - 640 * y_ratio) / 2 + int(640 * y_ratio)), int((480 - 480 * x_ratio) / 2) + int(480 * x_ratio))
        # my_im2 = cv2.rectangle(my_im2, (int((640 - 640 * y_ratio) / 2), int((480 - 480 * x_ratio) / 2)),
        #                        (int((640 - 640 * y_ratio) / 2 + int(640 * y_ratio)), int((480 - 480 * x_ratio) / 2) + int(480 * x_ratio)),
        #                        (255, 0, 0), 1)

        # my_im2 = cv2.rectangle(my_im2, (int((480 - 480 * x_ratio) / 2), int((640 - 640 * y_ratio) / 2)),
        #                        (int((480 - 480 * x_ratio) / 2) + int(480 * x_ratio), int((640 - 640 * y_ratio) / 2 + int(640 * y_ratio))),
        #                        (255, 0, 0), 1)

        # print(my_im2.shape)
        add = int((640 - 480) / 2)
        img = cv2.copyMakeBorder(my_im2, add, add, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 255))
        # print(img.shape)
        # img = yolo_box(img,label) # draw the box of each lego
        # cv2.imshow('zzz', img)
        # cv2.waitKey(0)
        # cv2.destroyWindow()
        cv2.imwrite("../YOLO_data/Dataset/image_yolo_409/" + "IMG_test%s.png" % epoch, img)

        # img = cv2.imread("Dataset/lego_yolo_403_test/" + "IMG_test%s.png" % epoch)
        # # cv2.namedWindow('zzz', 0)
        # cv2.imshow('zzz', img)
        # cv2.waitKey(0)
        # # cv2.destroyWindow()