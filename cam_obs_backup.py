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

from network import *

torch.manual_seed(42)
data_root = '/home/ubuntu/Desktop/knolling_dataset/resnet/'
log = 'log_412_norm_2/'
norm_flag = False

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Device:", device)

resnet_model = ResNet50(img_channel=3, output_size=4).to(device, dtype=torch.float32)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
resnet_model.load_state_dict(torch.load(os.path.join(data_root, log, 'model/best_model.pt'), map_location='cuda:0'))
# add map_location='cuda:0' to run this model trained in multi-gpu environment on single-gpu environment
resnet_model.eval()

close_path = os.path.join(data_root, "input/yolo_407_close/")
normal_path = os.path.join(data_root, "input/yolo_407_normal/")
close_label = np.loadtxt(os.path.join(data_root, 'label/label_407_close_train.csv'))[:, :4]
normal_label = np.loadtxt(os.path.join(data_root, 'label/label_407_normal_train.csv'))[:, :4]

data_num = 1000
data_4_train = int(data_num * 0.8)
ratio = 0.5  # close3, normal7
close_num_train = int(data_4_train * ratio)
normal_num_train = int(data_4_train - close_num_train)
close_num_test = int((data_num - data_4_train) * ratio)
normal_num_test = int((data_num - data_4_train) - close_num_test)
print('this is num of close', int(close_num_train + close_num_test))
print('this is num of normal', int(normal_num_train + normal_num_test))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def ResNet18(img_channel, output_size):
    return ResNet(block, [2, 2, 2, 2], img_channel, output_size)

def ResNet50(img_channel, output_size):
    return ResNet(block, [3, 4, 6, 3], img_channel, output_size)

class VD_Data(Dataset):
    def __init__(self, img_data, label_data):
        self.img_data = img_data
        self.label_data = label_data

    def __getitem__(self, idx):
        img_sample = self.img_data[idx]
        label_sample = self.label_data[idx]

        img_sample = img_sample[:,:,:3]
        img_sample = img_sample.transpose((2, 0, 1))

        # label_sample = scaler.transform(label_sample)

        img_sample = torch.from_numpy(img_sample)
        label_sample = torch.from_numpy(label_sample)

        sample = {'image': img_sample, 'lwcossin': label_sample}

        return sample

    def __len__(self):
        return len(self.img_data)

def eval_img4Batch(img_array, num_obj, sample_num=100):

    ############################# manual scaler ##############################
    norm_parameters = np.concatenate((np.min(normal_label, axis=0), np.max(normal_label, axis=0)))
    norm_parameters[5] = norm_parameters[1] + 1
    print(norm_parameters.reshape(2, -1))
    ############################# manual scaler ##############################

    test_label = np.concatenate((close_label[close_num_train:(close_num_train + close_num_test)],
                                 normal_label[normal_num_train:(normal_num_train + close_num_test)]))

    # ############################# auto scaler ##############################
    # scaler = MinMaxScaler()
    # scaler.fit(test_label)
    # print(scaler.data_max_)
    # print(scaler.data_min_)
    # ############################# auto scaler ##############################

    test_data = []
    # mm_sc = [[0, 14 / 1000, 14 / 1000], [np.pi, 34 / 1000, 16 / 1000]]
    for i in range(num_obj):
        img_array1 = img_array[i].astype(np.float32) / 255
        # img_array1[0], img_array1[2] = img_array1[2], img_array1[0]
        # img_array1[:, :, 0], img_array1[:, :, 2] = img_array1[:, :, 2], img_array1[:, :, 0]
        # print('this is img array 0', img_array1[0])
        # print('this is img array 2', img_array1[2])
        test_data.append(img_array1)

    test_dataset = VD_Data(
        img_data=test_data, label_data=test_label)

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
            # cv2.namedWindow("well", 0)
            # cv2.imshow('well', temp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # ############################## test the shape of img ##############################

            img = img.to(device, dtype=torch.float32)
            pred_lwcossin = resnet_model.forward(img)
            pred_lwcossin = pred_lwcossin.cpu().detach().numpy()
            # pred_lwcossin = scaler.inverse_transform(pred_lwcossin)
            pred_lwcossin = pred_lwcossin * (norm_parameters[4:] - norm_parameters[:4]) + norm_parameters[:4]

            print('this is pred_lwcossin', pred_lwcossin)
            # pred_xyzyaw_ori[:, 0] = pred_xyzyaw_ori[:, 0] * np.pi / 180

            return pred_lwcossin

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
    print('Plot4Batch!!!')

    all_pred = eval_img4Batch(img_label, obj_num)

    ############### order yolo output depend on x, y in the world coordinate system ###############
    xy_list = np.asarray(xy_list)
    order_yolo = np.lexsort((xy_list[:, 1], xy_list[:, 0]))

    xy_list_test = np.copy(xy_list[order_yolo, :])
    for i in range(len(order_yolo) - 1):
        if np.abs(xy_list_test[i, 0] - xy_list_test[i + 1, 0]) < 0.003:
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

    to_arm = []
    print('this is number of obj', obj_num)
    for i in range(obj_num):
        xy = xy_list[i]
        # my_yaw, my_length, my_width = all_pred[i][0], all_pred[i][1], all_pred[i][2]
        my_length, my_width, my_cos, my_sin = all_pred[i][0], all_pred[i][1], all_pred[i][2], all_pred[i][3]
        my_ori = np.arctan2(my_sin, my_cos) / 2
        if my_length < 0.018:
            my_ori = np.arctan2(my_sin, my_cos) / 4
        # pred_label = [xy[0], xy[1], my_yaw, my_length, my_width]

        info1 = f'cos: {my_cos:.3f} sin: {my_sin:.3f}, ori: {my_ori}'
        info2 = f'length: {my_length * 1000:.3f} width: {my_width * 1000:.3f}'
        # plot_one_box(xyxy, im0, label=label,
        #              color=colors[int(cls)], line_thickness=1)
        color_pos = f'color: {color_label[i]} x_pos: {xy[0]:.4f} y_pos: {xy[1]:.4f}'
        # my_plot_one_box(xyxy_list[i], img, my_yaw, my_length, my_width, label1=info1, label2=color_pos, label3=info2,
        #                 color=[0, 0, 0], line_thickness=1)
        check_flag = False
        center_pred = [int((xyxy_list[i][1] + xyxy_list[i][3]) / 2), int((xyxy_list[i][0] + xyxy_list[i][2]) / 2)]
        my_plot_one_box_lwcossin(center_pred, img, my_length, my_width, my_ori, label1=info1, label2=color_pos,
                        label3=info2,
                        color=[0, 0, 0], line_thickness=1, check_flag=check_flag)
        my_to_arm = [xy[0], xy[1], my_length, my_width, my_ori, color_label[i]]
        to_arm.append(my_to_arm)
        ############################### plot the ground truth ################################
        if not all_truth is None:
            length_truth, width_truth, cos_truth, sin_truth, x_truth, y_truth, ori_truth = all_truth[i][:7]
            # ori_truth = np.arctan2(sin_truth, cos_truth)
            # message = 'the ground truth is shown below'
            info1 = f'cos: {cos_truth:.3f} sin: {sin_truth:.3f}, ori: {ori_truth:.3f}'
            info2 = f'length: {length_truth * 1000:.3f} width: {width_truth * 1000:.3f}'
            info3 = f'x: {x_truth:.3f} y: {y_truth:.3f}'
            check_flag = True
            mm2px = 530 / 0.34
            # mm2px = 1500

            center_ground_truth = [int(x_truth * mm2px + 86), int(y_truth * mm2px + 320)]
            my_plot_one_box_lwcossin(center_ground_truth, img, length_truth, width_truth, ori_truth,
                            label1=info1, label2=info2, label3=info3, color=[0, 0, 0], line_thickness=1,
                            check_flag=check_flag)
        else:
            pass
        # my_to_arm = [center_ground_truth[1], center_ground_truth[0], my_length, my_width, my_ori, color_label[i]]
        # to_arm.append(my_to_arm)
        ############################### plot the ground truth ################################


    return img, to_arm

def img_modify(my_im2, xyxy, img_label, color_label, xy_label, num_obj, real_operate, use_yolo_pos, all_truth):

    # left-top to right-down
    px_resx1 = int(xyxy[0].cpu().detach().numpy())  # row
    px_resy1 = int(xyxy[1].cpu().detach().numpy())  # column
    px_resx2 = int(xyxy[2].cpu().detach().numpy())  # row
    px_resy2 = int(xyxy[3].cpu().detach().numpy())  # column

    # find obj position:
    obj_x = int((px_resx1 + px_resx2) / 2)
    obj_y = int((px_resy1 + px_resy2) / 2)

    # obj_x = obj_x * (530 / 0.34) / (1500)
    # obj_y = obj_y * (530 / 0.34) / (1500)

    if real_operate == True:
        mm2px = 530 / 0.34
    else:
        mm2px = 530 / 0.34

    if real_operate == True:
        obj_x = obj_x - 320  # move it to the world coordinate 从左上角移动到pybullet中的（0，0）
        # obj_y = obj_y - (0.15 - (0.3112 - 0.15)) * mm2px + 5 # add the rm_distortion!!!!!!!!!!
        obj_y = obj_y - 86
    else:
        obj_x = obj_x - 320  # move it to the world coordinate 从左上角移动到pybullet中的（0，0)
        # obj_y = obj_y - (0.15 - (0.3112 - 0.15)) * mm2px + 5 # add the rm_distortion!!!!!!!!!!
        obj_y = obj_y - 86

    # convert px to mm!!!!!!!!!!!!!
    obj_x = obj_x / mm2px
    obj_y = obj_y / mm2px
    # obj_x, obj_y = xyz_resolve_inverse(obj_x, obj_y)

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
    px_padtop = px_resy1
    px_padbot = px_resy2
    px_padleft = px_resx1
    px_padright = px_resx2
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
    # img = cv2.addWeighted(img, 1.1, blank, 0.1, 60)

    cv2.imwrite('img_yolo%s.png' % num_obj, img)

    if det_color == 'undefined':
        obj_color = img[48, 48, :]
        # print('bgr',obj_color)
        obj_hsv = np.uint8([[obj_color]])
        obj_hsv = cv2.cvtColor(obj_hsv, cv2.COLOR_BGR2HSV)
        # print('hsv',obj_hsv)

        det_color = color_define(obj_hsv[0][0])

    img_label.append(img)
    color_label.append(det_color)

def check_resnet_yolo(obs_img_from=None, results=None, target=None):

    test_data = []
    test_label = np.concatenate((close_label[close_num_train:(close_num_train + close_num_test)],
                                 normal_label[normal_num_train:(normal_num_train + close_num_test)]))

    norm_parameters = np.concatenate((np.min(test_label, axis=0), np.max(test_label, axis=0)))
    norm_parameters[5] = norm_parameters[1] + 1
    print(norm_parameters.reshape(2, -1))
    print(norm_parameters[4:] - norm_parameters[:4])

    if obs_img_from == 'dataset':
        for i in range(close_num_train, close_num_train + close_num_test):
            img = plt.imread(close_path + "img%d.png" % i)
            test_data.append(img)

        for i in range(normal_num_train, normal_num_train + normal_num_test):
            img = plt.imread(normal_path + "img%d.png" % i)
            test_data.append(img)

        test_label -= norm_parameters[:4]
        test_label /= (norm_parameters[4:] - norm_parameters[:4])

        test_dataset = VD_Data(img_data=test_data, label_data=test_label)
        ################# choose the ratio of close and normal img #################

        num_epochs = 100
        BATCH_SIZE = 12
        learning_rate = 1e-4

        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=4)
        all_valid_L = []

        print('begin!')
        for epoch in range(num_epochs):
            t0 = time.time()
            valid_L = []

            with torch.no_grad():
                for batch in test_loader:
                    img, lwcossin = batch["image"], batch["lwcossin"]

                    ############################# test the shape of img ##############################
                    img_show = img.cpu().detach().numpy()
                    for i in range(len(img_show)):
                        print(img_show[i].shape)
                        temp = img_show[i]
                        temp_shape = temp.shape
                        temp = temp.reshape(temp_shape[1], temp_shape[2], temp_shape[0])
                        print(temp.shape)
                        cv2.namedWindow("well", 0)
                        cv2.imshow('well', temp)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    # quit()
                    ############################## test the shape of img ##############################

                    img = img.to(device, dtype=torch.float32)
                    lwcossin = lwcossin.to(device, dtype=torch.float32)
                    # print('this is lwcossin\n', lwcossin)

                    lwcossin_origin = lwcossin.cpu() * (norm_parameters[4:] - norm_parameters[:4]) + norm_parameters[:4]
                    print('this is lwcossin origin\n', lwcossin_origin)

                    pred_lwcossin = resnet_model.forward(img)
                    # print('this is pred lwcossin\n', pred_lwcossin)

                    pred_lwcossin_origin = pred_lwcossin.cpu() * (
                            norm_parameters[4:] - norm_parameters[:4]) + norm_parameters[:4]
                    print('this is pred lwcossin origin\n', pred_lwcossin_origin)
                    loss = resnet_model.loss(pred_lwcossin, lwcossin)
                    print('this is loss per batch', loss.item())
                    valid_L.append(loss.item())
            avg_valid_L = np.mean(valid_L)
            all_valid_L.append(avg_valid_L)
            # print('this is avg_valid_L', avg_valid_L)
        all_valid_L = np.asarray(all_valid_L)
        return np.mean(all_valid_L)

    elif obs_img_from == 'env':
        # for i in range(12):
        #     img = plt.imread("../img_yolo%d.png" % i)
        #     test_data.append(img)

        for i in range(len(results)):
            if results[i][2] < 0.018:
                results[i][4] = results[i][4] * 2
        pred_cos = np.cos(2 * results[:, 4].reshape((-1, 1)))
        pred_sin = np.sin(2 * results[:, 4].reshape((-1, 1)))
        pred_compare = np.concatenate((results[:, 2:4], pred_cos, pred_sin), axis=1)
        print('this is the target_compare\n', target)
        print('this is the pred_compare\n', pred_compare)
        # pred_compare_scaled = scaler.transform(pred_compare)
        pred_compare_scaled = pred_compare - norm_parameters[:4]
        pred_compare_scaled /= (norm_parameters[4:] - norm_parameters[:4])
        target_compare_scaled = target - norm_parameters[:4]
        target_compare_scaled /= (norm_parameters[4:] - norm_parameters[:4])

        zzz_error_norm = np.mean((pred_compare_scaled - target_compare_scaled) ** 2)
        # zzz_error = np.mean((pred_compare - target) ** 2)
        print('this is the scaled error between the target and the pred', zzz_error_norm)
        # print('this is the error between the target and the pred', zzz_error)
        return zzz_error_norm

    else:
        pass

def detect(cam_img,save_img=False, evaluation=None, real_operate=None, all_truth=None, use_yolo_pos=True):
    cam_obs = True
    path = ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,default='yolov7/runs/train/zzz_yolo/weights/best_408.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    if cam_obs:
        parser.add_argument('--source', type=str, default=path, help='source')
    else:
        parser.add_argument('--source', type=str, default='2', help='source')
    parser.add_argument('--img-size', type=int, default=640,help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.85, help='object confidence threshold')
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

            # im0s = cv2.rectangle(im0s, (0, 80), (640, 95), (0, 0, 0), thickness=-1)
            # im0s = cv2.rectangle(im0s, (0, 546), (640, 560), (0, 0, 0), thickness=-1)
            #
            # # cv2.namedWindow("zzz_origin", 0)
            # # cv2.imshow('zzz_origin', im0s)
            # # cv2.waitKey(0)
            # # cv2.destroyAllWindows()
            #
            # blank_mask = np.zeros(im0s.shape, dtype=np.uint8)
            # original = im0s.copy()
            # hsv = cv2.cvtColor(im0s, cv2.COLOR_BGR2HSV)
            # lower = np.array([0, 95, 20])
            # upper = np.array([255, 255, 255])
            # mask = cv2.inRange(hsv, lower, upper)
            #
            # cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            # cv2.drawContours(blank_mask, cnts, -1, (255, 255, 255), -1)
            # # for c in cnts:
            # #     cv2.drawContours(blank_mask,[c], -1, (255,255,255), -1)
            # #     break
            #
            # result = cv2.bitwise_and(original, blank_mask)
            # pixel = 200
            # result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            # result[result_gray < 1] = pixel
            # im0s = np.copy(result)
            #
            # # cv2.imshow('result', result)
            # # cv2.waitKey(0)
            # # cv2.destroyAllWindows()
            pass
        else:
            print(length_width_channel)
            # img = img.reshape(length_width_channel[1], length_width_channel[2], length_width_channel[0])
            # img = np.clip((1.03 * img), 0, 255)
            #
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
            #
            # im0s = np.uint8(np.clip((1.2 * im0s + 10), 0, 255))
            # cv2.imshow('aaa', im0s)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imshow('bbb', im0s_result_norm)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # img = img.reshape(length_width_channel[0], length_width_channel[1], length_width_channel[2])
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

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'

                        img_modify(my_im, xyxy, img_label, color_label, xy_list, box_number, real_operate, use_yolo_pos, all_truth)

                        box_number += 1

                # print(box_number)
                im0, to_arm = Plot4Batch(im0, xyxy_list, xy_list, img_label, color_label, box_number, all_truth)
                cv2.namedWindow('123', 0)
                cv2.imshow('123', im0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # cv2.imwrite(f'./Test_images/movie_yolo_resnet/{evaluation}.png',im0)
                if real_operate == True:
                    cv2.imwrite(f'./Test_images/test_412_combine_real_2.png', im0)
                else:
                    cv2.imwrite(f'./Test_images/test_412_combine_sim_2.png', im0)

                # cv2.waitKey(1000)
                if cam_obs:
                    return to_arm
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    print(f'Done. ({time.time() - t0:.3f}s)')



if __name__ == '__main__':



    # print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))
    cam_img = cv2.imread('img.png')
    print(cam_img)
    # t0 = time.time()
    print(detect(cam_img))
    # print(time.time() - t0)


