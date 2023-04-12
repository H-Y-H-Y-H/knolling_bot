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

# np.random.seed(100)
# random.seed(100)

def sort_cube(obj_list, x_sorted):
    s = np.array(x_sorted)
    sort_index = np.argsort(s)
    # print(sort_index)
    sorted_sas_list = [obj_list[j] for j in sort_index]
    return sorted_sas_list


def my_plot_one_box(x, img, my_yaw, my_length, my_width, color=None, label1='1', line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label1:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label1, 0, fontScale=tl / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # my boundary
        gamma = my_yaw

        rot_z = [[np.cos(gamma), -np.sin(gamma)],
                 [np.sin(gamma), np.cos(gamma)]]

        yy = (x[2]+x[0])/2
        xx = (x[3]+x[1])/2

        rot_z = np.asarray(rot_z)

        c11 = [my_length / 2, my_width / 2]
        c22 = [my_length / 2, -my_width / 2]
        c33 = [-my_length / 2, my_width / 2]
        c44 = [-my_length / 2, -my_width / 2]

        mm2px = 1 / 0.000625
        c11, c22, c33, c44 = np.asarray(c11), np.asarray(c22), np.asarray(c33), np.asarray(c44)
        c11 = c11 * mm2px
        c22 = c22 * mm2px
        c33 = c33 * mm2px
        c44 = c44 * mm2px

        corn1 = np.dot(rot_z, c11)
        corn2 = np.dot(rot_z, c22)
        corn3 = np.dot(rot_z, c33)
        corn4 = np.dot(rot_z, c44)

        corn1 = [corn1[0] + xx, corn1[1] + yy]
        corn2 = [corn2[0] + xx, corn2[1] + yy]
        corn3 = [corn3[0] + xx, corn3[1] + yy]
        corn4 = [corn4[0] + xx, corn4[1] + yy]

        cv2.line(img, (int(corn1[1]), int(corn1[0])), (int(corn2[1]), int(corn2[0])), (0, 0, 0), 2)
        cv2.line(img, (int(corn2[1]), int(corn2[0])), (int(corn4[1]), int(corn4[0])), (0, 0, 0), 2)
        cv2.line(img, (int(corn4[1]), int(corn4[0])), (int(corn3[1]), int(corn3[0])), (0, 0, 0), 2)
        cv2.line(img, (int(corn3[1]), int(corn3[0])), (int(corn1[1]), int(corn1[0])), (0, 0, 0), 2)

        # cv2.putText(img, label1, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # cv2.putText(img, label2, (c1[0], c1[1] - 20), 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)
        # cv2.putText(img, label3, (c1[0], c1[1] - 40), 0, tl / 3, color, thickness=tf,
        #             lineType=cv2.LINE_AA)


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

    startnum = 0
    endnum = 400
    lebal_list = []
    reset_flag = True
    num_reset = True

    # count_item = 0
    # mm2px = 1 / 0.000625 # (1600)
    mm2px = 530 / 0.34 # (1558)
    # mm2px = 1600
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
        print(all_pos)
        print(all_ori)


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

                xpos, ypos = xyz_resolve(xpos1,ypos1)

                corn1, corn2, corn3, corn4 = find_corner(xpos, ypos, lucky_list[j], yawori)
                # print('this is corn after find corner', corn1, corn2, corn3, corn4)

                corn1, corn2, corn3, corn4 = resolve_img(corn1, corn2, corn3, corn4)
                # print('this is corn after resolve img', corn1, corn2, corn3, corn4)

                corner_list.append([corn1, corn2, corn3, corn4])

                if lucky_list[j] == 2:
                    l = 16/1000
                    w = 16/1000
                if lucky_list[j] == 3:
                    l = 24/1000
                    w = 16/1000
                if lucky_list[j] == 4:
                    l = 32/1000
                    w = 16/1000

                corns = corner_list[j]

                col_offset = 320
                # row_offset = (0.15 - (0.3112 - 0.15)) * mm2px + 5
                row_offset = 0
                print(col_offset)
                print(row_offset)

                col_list = [int(mm2px * corns[0][1] + col_offset), int(mm2px * corns[3][1] + col_offset),
                            int(mm2px * corns[1][1] + col_offset), int(mm2px * corns[2][1] + col_offset)]
                row_list = [int(mm2px * corns[0][0] - row_offset), int(mm2px * corns[3][0] - row_offset),
                            int(mm2px * corns[1][0] - row_offset), int(mm2px * corns[2][0] - row_offset)]
                print(col_list)
                print(row_list)

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


                element = []
                # element.append(0)
                # element.append(xpos1)
                # element.append(ypos1)
                # element.append(lucky_list[j])
                # element.append(yawori)
                # element.append(l)
                # element.append(w)

                ########################################
                element.append(0)
                element.append(label_x)
                element.append(label_y)
                element.append(length)
                element.append(width)
                element = np.asarray(element)
                label.append(element)
                ###############################
                # # element.append(0)
                # element.append(rdm_pos_x[j])
                # element.append(rdm_pos_y[j])
                # # element.append(length)
                # # element.append(width)
                # element = np.asarray(element)
                # label.append(element)

        # x_compare = [label[0][0], label[1][0], label[2][0]]
        # idx_cmp = np.argsort(x_compare)
        # print(idx_cmp)
        # print(x_compare)
        # print(x_compare)
        # label2 = sort_cube(label, x_compare)
        # print(label2)

        # obj_list = np.concatenate([label2[0], label2[1], label2[2]])

        # print(label)
        np.savetxt("Dataset/real_world_label_405/img%s.txt" %epoch, label,fmt='%.8s')

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
        img = yolo_box(img,label) # draw the box of each lego
        # cv2.imshow('zzz', img)
        # cv2.waitKey(0)
        # cv2.destroyWindow()
        cv2.imwrite("Dataset/image_yolo_405/" + "IMG_test%s.png" % epoch, img)

        # img = cv2.imread("Dataset/lego_yolo_403_test/" + "IMG_test%s.png" % epoch)
        # # cv2.namedWindow('zzz', 0)
        # cv2.imshow('zzz', img)
        # cv2.waitKey(0)
        # # cv2.destroyWindow()