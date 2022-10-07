import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from VD_model import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import cv2
import torchvision

# a = label[:,3]
# print(a*180/np.pi)
# # num = label[:,3]//np.pi
# # print(num)
# # a_i = a - num*np.pi
# a_i = a%np.pi
# for i in range(len(a_i)):
#     if a_i[i] < 0:
#         a_i[i] = a_i[i] + np.ppi/2
#
# print(a_i*180/np.pi)


#
# for i in range(len(label)):
#     if label[i,3]< (-np.pi/2):
#         new_label[i, 3] = label[i, 3] + (np.pi)
#     elif  label[i,3]> (np.pi/2):
#         new_label[i, 3] = label[i, 3] - (np.pi / 2)

# for i in range(len(label)):
#     if label[i,3]< (0):
#         new_label[i, 3] = (label[i, 3] + (np.pi*2))%(np.pi/2)
#     elif  label[i,3]> (0):
#         new_label[i, 3] = label[i, 3]%(np.pi/2)
#
# print(new_label[:,3]*180/np.pi)
# print(label[:,3]*180/np.pi)


def transform_img(self, img):

        if random.randint(0, 1) == 1:
            # cv2.imshow('IMG_origin',np.hstack((img[0],img[1],img[2],img[3],img[4])))
            sig_r_xy = random.uniform(0.1, 5)
            win_r = 2 * random.randint(1, 20) + 1
            img = cv2.GaussianBlur(img, (win_r, win_r), sigmaX=sig_r_xy, sigmaY=sig_r_xy,
                                   borderType=cv2.BORDER_DEFAULT)
            cv2.imshow('IMG_Blurring', np.hstack((img[0], img[1], img[2], img[3], img[4])))
            cv2.waitKey(0)
        IMG = torch.from_numpy(img).to(device, dtype=torch.float)

        if random.randint(0, 1) == 1:
            T = torchvision.transforms.ColorJitter(brightness=[0.1, 10])
            IMG = T(IMG)



img_pth = "../Dataset/Data_50k/"


