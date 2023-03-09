import sys

import argparse
import time
# import pyrealsense2 as rs
from pathlib import Path
from math import isclose
import numpy as np
import torch.nn as nn
import tqdm
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ubuntu/Desktop/knolling_bot/yolov7')
sys.path.append('/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_bot/yolov7')
from utils.plots import my_plot_one_box_lwcossin
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadImages2
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import my_plot_one_box
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import os
from sklearn.preprocessing import MinMaxScaler
# from train_img.visual_domain_random import eval_img4Batch


torch.manual_seed(42)
criterion = 'lwcossin'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=(1, 1)):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()

        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):

        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, output_size):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)
        self.sigmoid = nn.Sigmoid()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # # 225_combine structure
        # self.fc0 = nn.Linear(512 * 4, 256 * 4)
        # self.fc1 = nn.Linear(256 * 4, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 64)
        # self.fc5 = nn.Linear(64, 32)
        # self.fc6 = nn.Linear(32, output_size)

        # 226_combine_2 structure
        self.fc0 = nn.Linear(512 * 4, 512 * 6)
        self.fc1 = nn.Linear(512 * 6, 256 * 6)
        self.fc2 = nn.Linear(256 * 6, 256 * 4)
        self.fc3 = nn.Linear(256 * 4, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6_1 = nn.Linear(128, 80)
        self.fc6_2 = nn.Linear(128, 64)
        self.fc7_1 = nn.Linear(80, 32)
        self.fc7_2 = nn.Linear(64, 16)
        self.fc8_1 = nn.Linear(32, output_size - 2)
        self.fc8_2 = nn.Linear(16, 2)

        # # 226_combine structure
        # self.fc0 = nn.Linear(512 * 4, 512 * 6)
        # self.fc1 = nn.Linear(512 * 6, 256 * 6)
        # self.fc2 = nn.Linear(256 * 6, 256 * 4)
        # self.fc3 = nn.Linear(256 * 4, 256 * 2)
        # self.fc4 = nn.Linear(256 * 2, 256)
        # self.fc5 = nn.Linear(256, 128)
        # self.fc6 = nn.Linear(128, 64)
        # self.fc7 = nn.Linear(64, 16)
        # self.fc8 = nn.Linear(16, output_size)

    def forward(self, IMG):

        x = self.conv1(IMG)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # # 226_combine structure
        # x = self.avgpool(x)
        # x = x.reshape(x.shape[0], -1)
        # x = self.relu(self.fc0(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc3(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc4(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc5(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc6(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc7(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc8(x))

        # 226_combine_2 structure
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc0(x))
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x1 = self.relu(self.fc6_1(x))
        x1 = self.dropout(x1)
        x1 = self.relu(self.fc7_1(x1))
        x1 = self.dropout(x1)
        x1 = self.fc8_1(x1)
        x2 = self.relu(self.fc6_2(x))
        x2 = self.dropout(x2)
        x2 = self.relu(self.fc7_2(x2))
        x2 = self.dropout(x2)
        x2 = self.fc8_2(x2)

        # x = self.fc7(x)
        # print(x1)
        # print(x2)
        # print(torch.cat((x1, x2), dim=-1))

        return torch.cat((x1, x2), dim=-1)
        # return x

    def loss(self, pred, target):

            # index_yaw = 3, 11, 19, 27, 35
            # index_LWH = [4,5,6] [12,13,14] [20,21,22] [28,29,30] [36,37,38]
            # weights = np.ones(40)
            #
            # for k in range(5):
            #     l = target[k][4 + 8*k]
            #     w = target[k][5 + 8*k]
            #     if abs(l - w) >= 0.003:
            #         weights[3 + 8*k] = 0.01
            #
            # weights_tensor = torch.from_numpy(weights).to('cuda')
            # value = torch.mul((pred - target) ** 2, weights_tensor)

            value = (pred - target) ** 2

            return torch.mean(value)

    # def loss(self, pred, target):

    #     # index_yaw = 3, 11, 19, 27, 35
    #     # index_LWH = [4,5,6] [12,13,14] [20,21,22] [28,29,30] [36,37,38]
    #     weights = np.ones(40)
    #
    #     for k in range(5):
    #         l = target[4 + 8*k]
    #         w = target[5 + 8*k]
    #         h = target[6 + 8*k]
    #         if l == w == h:
    #             weights[3 + 8*k] = 0.01
    #
    #     class_weights = torch.from_numpy(weights)
    #     loss = nn.CrossEntropyLoss(weight=class_weights)
    #
    #     return loss(pred, target)

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

def ResNet18(img_channel, output_size):
    return ResNet(block, [2, 2, 2, 2], img_channel, output_size)

def ResNet50(img_channel, output_size):
    return ResNet(block, [3, 4, 6, 3], img_channel, output_size)

class VD_Data(Dataset):
    def __init__(self, img_data, label_data, transform=None):
        self.img_data = img_data
        self.label_data = label_data
        self.transform = transform

    def __getitem__(self, idx):
        img_sample = self.img_data[idx]
        label_sample = self.label_data[idx]

        sample = {'image': img_sample, 'xyzyaw': label_sample}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_data)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_sample, label_sample= sample['image'], sample['xyzyaw']
        img_sample = np.asarray([img_sample])

        # print(type(img_sample))
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print(img_sample.shape)
        img_sample = np.squeeze(img_sample)
        img_sample = img_sample[:,:,:3]
        # print(img_sample.shape)
        # image = img_sample
        image = img_sample.transpose((2, 0, 1))
        # print(image.shape)
        img_sample = torch.from_numpy(image)


        return {'image': img_sample,
                'xyzyaw': label_sample}

def eval_img4Batch(img_array, num_obj):

    # print('this is img_array', img_array)
    # print('this is num_obj', num_obj)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)

    if criterion == 'lwcossin':
        model = ResNet50(img_channel=3, output_size=4).to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.load_state_dict(torch.load('Model/best_model_306_combine.pt', map_location='cuda:0'))
        # add map_location='cuda:0' to run this model trained in multi-gpu environment on single-gpu environment
        model.eval()

        data_num = 60000
        data_4_train = int(data_num * 0.8)
        ratio = 0.5  # close3, normal7
        close_num_test = int((data_num - data_4_train) * ratio)
        normal_num_test = int((data_num - data_4_train) - close_num_test)
        close_index = int(data_4_train * ratio)
        normal_index = int(data_4_train * (1 - ratio))

        close_label = np.loadtxt('./Dataset/label/label_304_close_2.csv')[:, :4]
        normal_label = np.loadtxt('./Dataset/label/label_304_normal_2.csv')[:, :4]
        test_label = []

        for i in range(close_num_test):
            test_label.append(close_label[close_index])
            close_index += 1
        for i in range(normal_num_test):
            test_label.append(normal_label[normal_index])
            normal_index += 1

        scaler = MinMaxScaler()
        scaler.fit(test_label)
        print(scaler.data_max_)
        print(scaler.data_min_)

        test_data = []
        # mm_sc = [[0, 14 / 1000, 14 / 1000], [np.pi, 34 / 1000, 16 / 1000]]
        for i in range(num_obj):
            img_array1 = img_array[i].astype(np.float32) / 255
            img_array1[0], img_array1[2] = img_array1[2], img_array1[0]
            image2 = img_array1
            # image = plt.imread('real_test/img%s.png' %i)
            # print(image)
            # print(image2)

            # if (image == image2).all():
            #     print('!')

            # print('shape',image.shape)
            # print('shape', image2.shape)
            test_data.append(image2)

        test_dataset = VD_Data(
            img_data=test_data, label_data=test_label, transform=ToTensor())

        BATCH_SIZE = 32

        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0)

        with torch.no_grad():

            for batch in test_loader:
                img = batch["image"]

                # ############################## test the shape of img ##############################
                # img_show = img.cpu().detach().numpy()
                # print(img_show[0].shape)
                # temp = img_show[0]
                # temp_shape = temp.shape
                # temp = temp.reshape(temp_shape[1], temp_shape[2], temp_shape[0])
                # print(temp.shape)
                # cv2.namedWindow("well",0)
                # cv2.imshow('well', temp)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # ############################## test the shape of img ##############################

                img = img.to(device)
                pred_lwcossin = model.forward(img)
                pred_lwcossin = pred_lwcossin.cpu().detach().numpy()
                # print('this is', pred_x1y1x2y2l)
                pred_lwcossin = scaler.inverse_transform(pred_lwcossin)
                # print('this is', pred_x1y1x2y2l)
                # pred_xyzyaw_ori[:, 0] = pred_xyzyaw_ori[:, 0] * np.pi / 180

                return pred_lwcossin

    if criterion == 'x1y1x2y2':
        model = ResNet50(img_channel=3, output_size=4).to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.load_state_dict(torch.load('Model/best_model_302_combine_**2.pt', map_location='cuda:0'))
        # add map_location='cuda:0' to run this model trained in multi-gpu environment on single-gpu environment
        model.eval()

        data_num = 60000
        data_4_train = int(data_num * 0.8)
        ratio = 0.5  # close3, normal7
        close_num_test = int((data_num - data_4_train) * ratio)
        normal_num_test = int((data_num - data_4_train) - close_num_test)
        close_index = int(data_4_train * ratio)
        normal_index = int(data_4_train * (1 - ratio))

        close_label = np.loadtxt('./Dataset/label/label_301_close_2.csv')[:, :5]
        normal_label = np.loadtxt('./Dataset/label/label_301_normal_2.csv')[:, :5]
        test_label = []

        for i in range(close_num_test):
            test_label.append(close_label[close_index])
            close_index += 1
        for i in range(normal_num_test):
            test_label.append(normal_label[normal_index])
            normal_index += 1

        scaler = MinMaxScaler()
        scaler.fit(test_label)
        print(scaler.data_max_)
        print(scaler.data_min_)

        test_data = []
        # mm_sc = [[0, 14 / 1000, 14 / 1000], [np.pi, 34 / 1000, 16 / 1000]]
        for i in range(num_obj):
            img_array1 = img_array[i].astype(np.float32)/255
            img_array1[0], img_array1[2] = img_array1[2], img_array1[0]
            image2 = img_array1
            # image = plt.imread('real_test/img%s.png' %i)
            # print(image)
            # print(image2)

            # if (image == image2).all():
            #     print('!')

            # print('shape',image.shape)
            # print('shape', image2.shape)
            test_data.append(image2)

        test_dataset = VD_Data(
            img_data=test_data, label_data = test_label, transform=ToTensor())

        BATCH_SIZE = 32

        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0)

        with torch.no_grad():

            for batch in test_loader:
                img = batch["image"]

                # ############################## test the shape of img ##############################
                # img_show = img.cpu().detach().numpy()
                # print(img_show[0].shape)
                # temp = img_show[0]
                # temp_shape = temp.shape
                # temp = temp.reshape(temp_shape[1], temp_shape[2], temp_shape[0])
                # print(temp.shape)
                # cv2.namedWindow("well",0)
                # cv2.imshow('well', temp)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # ############################## test the shape of img ##############################

                img = img.to(device)
                pred_x1y1x2y2l = model.forward(img)
                pred_x1y1x2y2l = pred_x1y1x2y2l.cpu().detach().numpy()
                # print('this is', pred_x1y1x2y2l)
                pred_x1y1x2y2l = scaler.inverse_transform(pred_x1y1x2y2l)
                # print('this is', pred_x1y1x2y2l)
                # pred_xyzyaw_ori[:, 0] = pred_xyzyaw_ori[:, 0] * np.pi / 180

                return pred_x1y1x2y2l

def color_define(obj_hsv):
    # obj_HSV = [H,S,V]
    if obj_hsv[0] >= 170:
        obj_hsv[0] -= 170  # red color

    if obj_hsv[2] < 25:
        return 'Black'

    if (obj_hsv[1] <= 30) and (obj_hsv[2] <= 170) and (obj_hsv[2] >= 70):
        return 'Gray'

    hsv_green_low = [50, 50, 50]
    hsv_green_high = [85, 255, 255]

    hsv_red_low = [0, 50, 50]
    hsv_red_high = [10, 255, 255]

    hsv_blue_low = [100, 50, 50]
    hsv_blue_high = [130, 255, 255]

    hsv_yellow_low = [15, 25, 25]
    hsv_yellow_high = [38, 255, 255]
    # print(obj_hsv)
    if (obj_hsv >= hsv_blue_low).all() and (obj_hsv <= hsv_blue_high).all():
        return 'Blue'

    elif (obj_hsv >= hsv_red_low).all() and (obj_hsv <= hsv_red_high).all():
        return 'Red'

    elif (obj_hsv >= hsv_yellow_low).all() and (obj_hsv <= hsv_yellow_high).all():
        return 'Yellow'

    elif (obj_hsv >= hsv_green_low).all() and (obj_hsv <= hsv_green_high).all():
        return 'Green'

    else:
        return 'undefined'


def Plot4Batch(img, xyxy_list, xy_list, img_label, color_label, obj_num, all_truth):

    # pos_truth is the xy pairs of lego cubes in world coordinate system, not z
    # ori_truth is the yaw, not the angle of two grasp points!!!!!!!!!!

    all_pred = eval_img4Batch(img_label, obj_num)

    ############### order yolo output depend on x, y in the world coordinate system ###############
    xy_list = np.asarray(xy_list)
    order_yolo = np.lexsort((xy_list[:, 1], xy_list[:, 0]))

    xy_list_test = np.copy(xy_list[order_yolo, :])
    for i in range(len(order_yolo) - 1):
        if np.abs(xy_list_test[i, 0] - xy_list_test[i + 1, 0]) < 0.005:
            if xy_list_test[i, 1] < xy_list_test[i + 1, 1]:
                # xy_list_test[order_yolo[i]], xy_list_test[order_yolo[i + 1]] = xy_list_test[order_yolo[i + 1]], xy_list_test[order_yolo[i]]
                order_yolo[i], order_yolo[i + 1] = order_yolo[i + 1], order_yolo[i]
                print('pred change the order!')
                print(xy_list[order_yolo[i]])
                print(xy_list[order_yolo[i+1]])
            else:
                pass

    all_pred = all_pred[order_yolo, :]
    new_xyxy_list = []
    new_color_label = []
    for i in order_yolo:
        new_xyxy_list.append(xyxy_list[i])
        new_color_label.append(color_label[i])
    xyxy_list = new_xyxy_list
    color_label = new_color_label
    xy_list = xy_list[order_yolo, :]
    ############### order yolo output depend on x, y in the world coordinate system ###############

    if criterion == 'lwcossin':
        to_arm = []
        print('this is number of obj', obj_num)
        for i in range(obj_num):
            xy = xy_list[i]
            # my_yaw, my_length, my_width = all_pred[i][0], all_pred[i][1], all_pred[i][2]
            my_length, my_width, my_cos, my_sin = all_pred[i][0], all_pred[i][1], all_pred[i][2], all_pred[i][3]
            my_ori = np.arctan2(my_sin, my_cos) / 2
            # pred_label = [xy[0], xy[1], my_yaw, my_length, my_width]

            info1 = f'cos: {my_cos:.3f} sin: {my_sin:.3f}, ori: {my_ori}'
            info2 = f'length: {my_length * 1000:.3f} width: {my_width * 1000:.3f}'
            # plot_one_box(xyxy, im0, label=label,
            #              color=colors[int(cls)], line_thickness=1)
            color_pos = f'color: {color_label[i]} x_pos: {xy[0]:.4f} y_pos: {xy[1]:.4f}'
            # my_plot_one_box(xyxy_list[i], img, my_yaw, my_length, my_width, label1=info1, label2=color_pos, label3=info2,
            #                 color=[0, 0, 0], line_thickness=1)
            check_flag = False
            my_plot_one_box_lwcossin(xyxy_list[i], img, my_length, my_width, my_ori, label1=info1, label2=color_pos,
                            label3=info2,
                            color=[0, 0, 0], line_thickness=1, check_flag=check_flag)
            my_to_arm = [xy[0], xy[1], my_length, my_width, my_ori, color_label[i]]
            to_arm.append(my_to_arm)
            ############################### plot the ground truth ################################
            length_truth, width_truth, cos_truth, sin_truth, x_truth, y_truth = all_truth[i][:6]
            ori_truth = np.arctan2(sin_truth, cos_truth) / 2
            # message = 'the ground truth is shown below'
            info1 = f'cos: {cos_truth:.3f} sin: {sin_truth:.3f}, ori: {ori_truth:.3f}'
            info2 = f'length: {length_truth * 1000:.3f} width: {width_truth * 1000:.3f}'
            info3 = f'x: {x_truth:.3f} y: {y_truth:.3f}'
            check_flag = True
            my_plot_one_box_lwcossin(xyxy_list[i], img, length_truth, width_truth, ori_truth,
                            label1=info1, label2=info2, label3=info3, color=[0, 0, 0], line_thickness=1,
                            check_flag=check_flag)
            ############################### plot the ground truth ################################

    if criterion == 'x1y1x2y2lw':
        to_arm = []
        print('this is number of obj', obj_num)
        for i in range(obj_num):
            xy = xy_list[i]
            # my_yaw, my_length, my_width = all_pred[i][0], all_pred[i][1], all_pred[i][2]
            x1, y1, x2, y2, my_length = all_pred[i][0], all_pred[i][1], all_pred[i][2], all_pred[i][3], all_pred[i][4]
            my_width = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
            # pred_label = [xy[0], xy[1], my_yaw, my_length, my_width]

            info1 = f'x1: {x1:.3f} y1: {y1:.3f} x2: {x2:.3f} y2: {y2:.3f}'
            info2 = f'length: {my_length * 1000:.3f} width: {my_width * 1000:.3f}'
            # plot_one_box(xyxy, im0, label=label,
            #              color=colors[int(cls)], line_thickness=1)
            color_pos = f'color: {color_label[i]} x_pos: {xy[0]:.4f} y_pos: {xy[1]:.4f}'
            # my_plot_one_box(xyxy_list[i], img, my_yaw, my_length, my_width, label1=info1, label2=color_pos, label3=info2,
            #                 color=[0, 0, 0], line_thickness=1)
            check_flag = False
            my_plot_one_box(xyxy_list[i], img, x1, y1, x2, y2, my_length, my_width, label1=info1, label2=color_pos, label3=info2,
                            color=[0, 0, 0], line_thickness=1, check_flag=check_flag)
            my_to_arm = [xy[0], xy[1], x1, y1, x2, y2, my_length, my_width, color_label[i]]
            to_arm.append(my_to_arm)
            ############################### plot the ground truth ################################
            x1_truth, y1_truth, x2_truth, y2_truth, length_truth = all_truth[i][0], all_truth[i][1], all_truth[i][2], all_truth[i][3], all_truth[i][4]
            width_truth = np.linalg.norm(np.array([x1_truth, y1_truth]) - np.array([x2_truth, y2_truth]))
            message = 'the ground truth is shown below'
            info2 = f'x1: {x1_truth:.3f} y1: {y1_truth:.3f} x2: {x2_truth:.3f} y2: {y2_truth:.3f}'
            check_flag = True
            my_plot_one_box(xyxy_list[i], img, x1_truth, y1_truth, x2_truth, y2_truth, length_truth, width_truth,
                            label1=message, label2=info2, label3=None, color=[0, 0, 0], line_thickness=1, check_flag=check_flag)
            ############################### plot the ground truth ################################


    return img, to_arm


def img_modify(my_im2, xyxy, img_label, color_label, xy_label, num_obj):

    # left-top to right-down
    px_resx1 = int(xyxy[0].cpu().detach().numpy())  # row
    px_resy1 = int(xyxy[1].cpu().detach().numpy())  # column
    px_resx2 = int(xyxy[2].cpu().detach().numpy())  # row
    px_resy2 = int(xyxy[3].cpu().detach().numpy())  # column

    # find obj position:
    obj_x = int((px_resx1 + px_resx2) / 2)
    obj_y = int((px_resy1 + px_resy2) / 2)

    mm2px = 1 / 0.000625  # unit convert = 1600

    obj_x = obj_x - 320  # move it to the world coordinate 从左上角移动到pybullet中的（0，0）
    obj_y = obj_y - 80

    obj_x = obj_x / mm2px
    obj_y = obj_y / mm2px

    xy_label.append([obj_y, obj_x])
    # print('box_number: ',box_number, obj_x, obj_y)
    # obj_label.append(obj_y)
    # obj_label.append(obj_x)
    # obj2_label.append(obj_label)

    ######
    # my_im2 = my_im.copy()

    obj_color = my_im2[int((px_resy1 + px_resy2) / 2), int((px_resx1 + px_resx2) / 2), :]
    # print('bgr',obj_color)
    obj_hsv = np.uint8([[obj_color]])
    obj_hsv = cv2.cvtColor(obj_hsv, cv2.COLOR_BGR2HSV)
    # print('hsv',obj_hsv)

    det_color = color_define(obj_hsv[0][0])
    # make all picture to 96 * 96
    px_padtop = px_resy1 - 7
    px_padbot = px_resy2 + 7
    px_padleft = px_resx1 - 7
    px_padright = px_resx2 + 7
    # print(px_padtop, px_padbot, px_padleft, px_padright)

    if px_padtop < 0:
        px_padtop = 0

    if px_padbot > 640:
        px_padbot = 640

    if px_padleft < 0:
        px_padleft = 0

    if px_padright > 640:
        px_padright = 640

    # print(px_padtop, px_padbot, px_padleft, px_padright)

    my_im2 = my_im2[px_padtop:px_padbot, px_padleft:px_padright, :] # 整张图分割成小块
    im2_y = my_im2.shape[0]
    im2_x = my_im2.shape[1]
    # print(my_im2.shape[0])

    pad_top = int((96 - im2_y) / 2)
    pad_bot = (96 - im2_y) - pad_top

    pad_left = int((96 - im2_x) / 2)
    pad_right = (96 - im2_x) - pad_left

    # print(pad_top,pad_bot,pad_left,pad_right)
    try:
        img = cv2.copyMakeBorder(my_im2, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                 value=(0, 0, 0))
    except:
        img = cv2.resize(my_im2, (96, 96))
    h, w, ch = img.shape
    blank = np.zeros([h, w, ch], img.dtype)
    img = cv2.addWeighted(img, 1.1, blank, 0.1, 60)

    # cv2.imwrite('img_yolo%s.png' % num_obj, img)

    if det_color == 'undefined':
        obj_color = img[48, 48, :]
        # print('bgr',obj_color)
        obj_hsv = np.uint8([[obj_color]])
        obj_hsv = cv2.cvtColor(obj_hsv, cv2.COLOR_BGR2HSV)
        # print('hsv',obj_hsv)

        det_color = color_define(obj_hsv[0][0])

    img_label.append(img)
    color_label.append(det_color)

def check_dataset():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)

    model = ResNet50(img_channel=3, output_size=5).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.load_state_dict(torch.load('Model/best_model_223_combine.pt', map_location='cuda:0'))
    # add map_location='cuda:0' to run this model trained in multi-gpu environment on single-gpu environment
    model.eval()

    data_num = 60000
    data_4_train = int(data_num * 0.8)
    ratio = 0.5  # close3, normal7
    close_num_test = (data_num - data_4_train) * ratio
    normal_num_test = (data_num - data_4_train) - close_num_test

    close_path = "./Dataset/yolo_221_2/"
    normal_path = "./Dataset/yolo_225/"
    close_index = data_4_train * ratio
    normal_index = data_4_train * (1 - ratio)
    test_data = []

    close_label = np.loadtxt('./Dataset/label/label_221_2.csv')[:, :5]
    normal_label = np.loadtxt('./Dataset/label/label_225.csv')[:, :5]
    test_label = []

    for i in range(int(close_num_test)):
        img = plt.imread(close_path + "img%d.png" % close_index)
        test_label.append(close_label[close_index])
        test_data.append(img)
        close_index += 1
    for i in range(int(normal_num_test)):
        img = plt.imread(normal_path + "img%d.png" % normal_index)
        test_label.append(normal_label[normal_index])
        test_data.append(img)
        normal_index += 1
    test_label = np.asarray(test_label)
    scaler = MinMaxScaler()
    scaler.fit(test_label)

    test_dataset = VD_Data(
        img_data=test_data, label_data=test_label, transform=ToTensor())

    BATCH_SIZE = 32

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # label = np.loadtxt('Dataset/label/label_221_combine.csv')[:, :5]
    # # print('this is label', label)
    #
    # xyzyaw = np.copy(label)
    #
    # scaler = MinMaxScaler()
    # scaler.fit(label)
    # print(scaler.data_max_)
    # print(scaler.data_min_)
    #
    # model = ResNet50(img_channel=3, output_size=5).to(device)
    # # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # model.load_state_dict(torch.load('Model/best_model_223_combine.pt', map_location='cuda:0'))
    # # add map_location='cuda:0' to run this model trained in multi-gpu environment on single-gpu environment
    # model.eval()
    #
    # img_pth = "Dataset/yolo_221_combine/"
    # data_num = 60000
    # data_4_train = int(data_num * 0.8)

    # test_data = []
    # for i in range(data_4_train,data_num):
    #     # print(i)
    #     img = plt.imread(img_pth + "img%d.png" % i)
    #     # print(np.shape(img))
    #     test_data.append(img)
    #
    # test_dataset = VD_Data(
    #     img_data=test_data, label_data=xyzyaw[data_4_train:data_num], transform=ToTensor())
    #
    # BATCH_SIZE = 32
    #
    # test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
    #                          shuffle=False, num_workers=0)

    with torch.no_grad():
        total_loss = []
        for batch in test_loader:
            img, x1y1x2y2l = batch["image"], batch["xyzyaw"]

            # ############################## test the shape of img ##############################
            # img_show = img.cpu().detach().numpy()
            # print(img_show[0].shape)
            # temp = img_show[3]
            # temp_shape = temp.shape
            # temp = temp.reshape(temp_shape[1], temp_shape[2], temp_shape[0])
            # print(temp.shape)
            # cv2.namedWindow("affasdf",0)
            # cv2.imshow('affasdf', temp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # ############################## test the shape of img ##############################

            img = img.to(device)
            pred_x1y1x2y2l = model.forward(img)

            target_x1y1x2y2l = scaler.transform(x1y1x2y2l)
            target_x1y1x2y2l = torch.from_numpy(target_x1y1x2y2l)
            target_x1y1x2y2l = target_x1y1x2y2l.to(device)
            # print('this is pred\n', pred_x1y1x2y2l)
            # print('this is target\n', target_x1y1x2y2l)
            loss = model.loss(pred_x1y1x2y2l, target_x1y1x2y2l)

            # if loss.item() < 0.1:
            #     print(loss)
            #     pred_x1y1x2y2l = pred_x1y1x2y2l.cpu().detach().numpy()
            #     # print('this is', pred_x1y1x2y2l)
            #     pred_x1y1x2y2l = scaler.inverse_transform(pred_x1y1x2y2l)
            #     print('this is pred after scaler\n', pred_x1y1x2y2l)
            #     print('this is target after scaler\n', x1y1x2y2l)

            # pred_xyzyaw_ori[:, 0] = pred_xyzyaw_ori[:, 0] * np.pi / 180

            total_loss.append(loss.item())

        total_loss = np.asarray(total_loss)
        print(np.mean(total_loss))
        return total_loss

def detect(cam_img,save_img=False, check_dataset_error=None, evaluation=None, real_operate=None, all_truth=None, order_truth=None):
    cam_obs = True
    path = ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,default='yolov7/runs/train/zzz_yolo/weights/best.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    if cam_obs:
        parser.add_argument('--source', type=str, default=path, help='source')
    else:
        parser.add_argument('--source', type=str, default='2', help='source')
    parser.add_argument('--img-size', type=int, default=640,help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp',help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    save_img = not opt.nosave and not source.endswith(
        '.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name,
    #                                exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
    #                                                       exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
    dataset = LoadImages2(source, cam_img, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # t0 = time.time()
    for path, img, im0s in dataset:

        # change the contrast and light of img and im0s
        # print('this is img', img)
        length_width_channel = img.shape

        ##################### change the lightness of the image ###################
        if real_operate == True:
            print(length_width_channel)
            img = img.reshape(length_width_channel[1], length_width_channel[2], length_width_channel[0])
            # cv2.imshow('aaa', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imshow('bbb', im0s)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img = np.clip((1.03 * img + 50), 0, 255)
            im0s = np.clip((1.03 * im0s + 50), 0, 255)
            # cv2.imshow('aaa', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imshow('bbb', im0s)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img = img.reshape(length_width_channel[0], length_width_channel[1], length_width_channel[2])
        else:
            print(length_width_channel)
            img = img.reshape(length_width_channel[1], length_width_channel[2], length_width_channel[0])
            # img = np.clip((1.03 * img), 0, 255)

            # im0s_split = cv2.split(im0s)
            #
            # result_planes = []
            # result_norm_planes = []
            # for plane in im0s_split:
            #     dilated_img = cv2.dilate(plane, np.ones((3, 3), np.uint8))
            #     bg_img = cv2.medianBlur(dilated_img, 11)
            #     diff_img = 255 - cv2.absdiff(plane, bg_img)
            #     norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
            #                              dtype=cv2.CV_8UC1)
            #     result_planes.append(diff_img)
            #     result_norm_planes.append(norm_img)
            #
            # im0s = cv2.merge(result_planes)
            # im0s_result_norm = cv2.merge(result_norm_planes)

            # im0s = np.uint8(np.clip((1.2 * im0s + 10), 0, 255))
            # cv2.imshow('aaa', im0s)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imshow('bbb', im0s_result_norm)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img = img.reshape(length_width_channel[0], length_width_channel[1], length_width_channel[2])
            pass
        ##################### change the lightness of the image ###################

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            img_label = [] # manual
            color_label = [] # manual
            box_number = 0 # manual
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + \
            #            ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                my_im = im0.copy() # manual
                xy_list = [] # manual
                xyxy_list = [] # manual
                for *xyxy, conf, cls in reversed(det):
                    xyxy_list.append(xyxy) # xyxy是yolo框中左上角和右下角的像素位置
                    # obj_label = []
                    # one_label = []
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                    #                       ) / gn).view(-1).tolist()  # normalized xywh
                    #     # label format
                    #     line = (
                    #         cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'

                        img_modify(my_im, xyxy, img_label, color_label, xy_list, box_number)

                        box_number += 1

                # print(box_number)
                im0, to_arm = Plot4Batch(im0, xyxy_list, xy_list, img_label, color_label, box_number, all_truth)
                # cv2.namedWindow('123', 0)
                # cv2.imshow('123', im0)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(f'./Test_images/movie_yolo_resnet/{evaluation}.png',im0)
                # if real_operate == True:
                #     cv2.imwrite(f'./Test_images/test_306_combine_real', im0)
                # else:
                #     cv2.imwrite(f'./Test_images/test_306_combine_sim.png', im0)

                # cv2.waitKey(1000)
                if cam_obs:
                    return to_arm
            # Print time (inference + NMS)
            print(
                f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            '''
            # Save results (image with detections)
            # print(save_img)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(
                        f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 5, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

            '''

    # if save_txt or save_img:
        # s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')



if __name__ == '__main__':



    # print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))
    cam_img = cv2.imread('img.png')
    print(cam_img)
    # t0 = time.time()
    print(detect(cam_img))
    # print(time.time() - t0)


