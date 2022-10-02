from turtle import pos
import numpy as np
import random
import math

x_low_obs = 0.15
x_high_obs = 0.55
y_low_obs = -0.2
y_high_obs = 0.2
xlen = x_high_obs - x_low_obs
ylen = y_high_obs - y_low_obs

def configuration(pos, ori):

    n = len(pos)
    result_pos = np.empty([n, 3])
    result_ori = np.empty([n, 3])
    print(result_pos[0, ...])

    box1_pos = pos[0]
    box2_pos = pos[1]
    box3_pos = pos[2]
    box4_pos = pos[3]
    box1_ori = ori[0]
    box2_ori = ori[1]
    box3_ori = ori[2]
    box4_ori = ori[3]
    print(box1_pos)
    print(box1_ori)
    box1_xlen = box1_pos[3] - box1_pos[0]
    box1_ylen = box1_pos[4] - box1_pos[1]
    box1_zlen = box1_pos[5] - box1_pos[2]
    box2_xlen = box2_pos[3] - box2_pos[0]
    box2_ylen = box2_pos[4] - box2_pos[1]
    box2_zlen = box2_pos[5] - box2_pos[2]
    box3_xlen = box3_pos[3] - box3_pos[0]
    box3_ylen = box3_pos[4] - box3_pos[1]
    box3_zlen = box3_pos[5] - box3_pos[2]
    box4_xlen = box4_pos[3] - box4_pos[0]
    box4_ylen = box4_pos[4] - box4_pos[1]
    box4_zlen = box4_pos[5] - box4_pos[2]

    # position
    if box1_xlen + box2_xlen + box3_xlen + box4_xlen > xlen:
        print('Out of length in X axis')
    elif box1_ylen + box2_ylen + box3_ylen + box4_ylen > ylen:
        print('Out of length in Y axis')
    else:
        print('the distance is ok')
        spacing = (xlen - box1_xlen - box2_xlen - box3_xlen - box4_xlen) / (n+1)
        # print(spacing)
        box1_xnew = x_low_obs + spacing + box1_xlen/2
        box2_xnew = box1_xnew + box1_xlen/2 + spacing + box2_xlen/2
        box3_xnew = box2_xnew + box2_xlen/2 + spacing + box3_xlen/2
        box4_xnew = box3_xnew + box3_xlen/2 + spacing + box4_xlen/2
        box1_ynew = (y_low_obs + y_high_obs) / 2
        box2_ynew = (y_low_obs + y_high_obs) / 2
        box3_ynew = (y_low_obs + y_high_obs) / 2
        box4_ynew = (y_low_obs + y_high_obs) / 2
        result_pos[0, ...] = box1_xnew, box1_ynew, 0.01
        result_pos[1, ...] = box2_xnew, box2_ynew, 0.01
        result_pos[2, ...] = box3_xnew, box3_ynew, 0.01
        result_pos[3, ...] = box4_xnew, box4_ynew, 0.01

    # orientation
    box1_ori = [0, 0, 0]
    box2_ori = [0, 0, 0]
    box3_ori = [0, 0, 0]
    box4_ori = [0, 0, 0]
    result_ori[0, ...] = box1_ori
    result_ori[1, ...] = box2_ori
    result_ori[2, ...] = box3_ori
    result_ori[3, ...] = box4_ori

    return result_pos, result_ori

