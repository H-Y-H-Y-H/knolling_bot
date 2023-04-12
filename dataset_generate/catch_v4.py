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
from knolling_env3 import *
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


    p.connect(p.GUI)

    startnum = 0
    endnum = 1000
    lebal_list = []
    num_item = 15

    num_reset = True
    CLOSE_FLAG = False

    count_item = 135000
    mm2px = 530 / 0.34

    for epoch in range(startnum,endnum):
        # num_item = random.randint(1, 5)


        env = Arm_env(max_step=1, is_render=False, num_objects=num_item)
        state, rdm_pos_x, rdm_pos_y, rdm_pos_z, rdm_ori_yaw, lucky_list = env.reset_table(close_flag=CLOSE_FLAG)

        label = []

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

                # corn1, corn2, corn3, corn4 = resolve_img(corn1, corn2, corn3, corn4)

                corner_list.append([corn1, corn2, corn3, corn4])

                if lucky_list[j] == 0:
                    l = 16/1000
                    w = 16/1000
                if lucky_list[j] == 1:
                    l = 24/1000
                    w = 16/1000
                if lucky_list[j] == 2:
                    l = 32/1000
                    w = 16/1000

                corns = corner_list[j]

                col_offset = 320
                # row_offset = (0.16 - (0.3112 - 0.16)) * mm2px - 12
                row_offset = 0
                # print('this is row_offset', row_offset)

                col_list = [int(mm2px * corns[0][1] + col_offset), int(mm2px * corns[3][1] + col_offset),
                            int(mm2px * corns[1][1] + col_offset), int(mm2px * corns[2][1] + col_offset)]
                row_list = [int(mm2px * corns[0][0] - row_offset), int(mm2px * corns[3][0] - row_offset),
                            int(mm2px * corns[1][0] - row_offset), int(mm2px * corns[2][0] - row_offset)]

                col_list = np.sort(col_list)
                row_list = np.sort(row_list)
                col_list[3] = col_list[3] + 0
                col_list[0] = col_list[0] - 0

                row_list[3] = row_list[3] + 0
                row_list[0] = row_list[0] - 0

                label_x = ((col_list[0] + col_list[3]) / 2)/640
                label_y = (((row_list[0] + row_list[3]) / 2)+86)/640

                length = (col_list[3] - col_list[0])/640
                width = (row_list[3] - row_list[0])/640

                # if lucky_list[j] == 2 and rdm_ori_yaw[j] < 0:
                #     rdm_ori_yaw[j] = rdm_ori_yaw[j] + np.pi/2

                matrix = np.array([[np.cos(yawori), -np.sin(yawori)],
                                   [np.sin(yawori), np.cos(yawori)]])
                grasp_point = np.array([[0, 0.016 / 2],
                                        [0, -0.016 / 2]])
                grasp_point_rotate = (matrix.dot(grasp_point.T)).T
                # print('this is grasp point', grasp_point_rotate)

                # element_yolo = []
                # element_yolo.append(0)
                # element_yolo.append(label_x)
                # element_yolo.append(label_y)
                # element_yolo.append(length)
                # element_yolo.append(width)
                # element_yolo = np.asarray(element_yolo)
                # label.append(element_yolo)

                element = []

                if grasp_point_rotate[0][0] > grasp_point_rotate[1][0]:
                    element.append(grasp_point_rotate[0][0])
                    element.append(grasp_point_rotate[0][1])
                    element.append(grasp_point_rotate[1][0])
                    element.append(grasp_point_rotate[1][1])
                else:
                    element.append(grasp_point_rotate[1][0])
                    element.append(grasp_point_rotate[1][1])
                    element.append(grasp_point_rotate[0][0])
                    element.append(grasp_point_rotate[0][1])
                element.append(l)
                element.append(w)
                element.append(yawori)

            lebal_list.append(element)
            my_im2 = env.get_image()
            add = int((640 - 480) / 2)
            img = cv2.copyMakeBorder(my_im2, add, add, 0, 0, cv2.BORDER_CONSTANT, None, value=0)

            px_resy1 = row_list[0]
            px_resy2 = row_list[3]
            px_resx1 = col_list[0]
            px_resx2 = col_list[3]

            px_padtop = px_resy1 - 7 + 6 # 6 is the edge between the image and zero point in coordinate system
            px_padbot = px_resy2 + 7 + 6
            px_padleft = px_resx1 - 7
            px_padright = px_resx2 + 7

            if px_padtop < 0:
                px_padtop = 0

            if px_padbot > 640:
                px_padbot = 640

            if px_padleft < 0:
                px_padleft = 0

            if px_padright > 640:
                px_padright = 640

            my_im2 = my_im2[px_padtop:px_padbot, px_padleft:px_padright, :]
            im2_y = my_im2.shape[0]
            im2_x = my_im2.shape[1]
            # print('this is im2', im2_x, im2_y)
            # print(my_im2.shape)

            pad_top = int((96 - im2_y) / 2)
            pad_bot = (96 - im2_y) - pad_top

            pad_left = int((96 - im2_x) / 2)
            pad_right = (96 - im2_x) - pad_left

            # img_yolo = yolo_box(img, label)

            # cv2.imwrite("Dataset/yolo_407_normal/" + "IMG_test%s.png" % epoch, img_yolo)

            img = cv2.copyMakeBorder(my_im2, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                     value=(0, 0, 0))

            cv2.imwrite('../ResNet_data/Dataset/yolo_409_normal/img%s.png'%count_item, img)
            count_item += 1

    np.savetxt("../ResNet_data/Label/Label_segmented/label_409_normal_135000150000.csv", lebal_list)
