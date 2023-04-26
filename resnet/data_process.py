import numpy as np
import os
import shutil
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt


def label_combine(path):

    # ll = []
    pj = np.loadtxt(os.path.join(data_root, "segmented_label/label_409_normal_015000.csv"))
    pp = np.loadtxt(os.path.join(data_root, "segmented_label/label_409_normal_1500030000.csv"))
    p0 = np.loadtxt(os.path.join(data_root, "segmented_label/label_409_normal_3000045000.csv"))
    p1 = np.loadtxt(os.path.join(data_root, "segmented_label/label_409_normal_4500060000.csv"))
    p2 = np.loadtxt(os.path.join(data_root, "segmented_label/label_409_normal_6000075000.csv"))
    p3 = np.loadtxt(os.path.join(data_root, "segmented_label/label_409_normal_7500090000.csv"))
    p4 = np.loadtxt(os.path.join(data_root, "segmented_label/label_409_normal_90000105000.csv"))
    p5 = np.loadtxt(os.path.join(data_root, "segmented_label/label_409_normal_105000120000.csv"))
    p6 = np.loadtxt(os.path.join(data_root, "segmented_label/label_409_normal_120000135000.csv"))
    p7 = np.loadtxt(os.path.join(data_root, "segmented_label/label_409_normal_135000150000.csv"))
    # p8 = np.loadtxt("label_326_normal_6000066000.csv")
    # p9 = np.loadtxt("label_326_normal_6600072000.csv")
    # p9 = np.loadtxt("label_217_330000360000_0180_close100.csv")
    # p10 = np.loadtxt("label_217_360000390000_0180_close100.csv")
        # ll.append(p)

    # ll = np.asarray(ll)
    ll2 = np.concatenate((pj,pp,p0,p1,p2,p3,p4,p5,p6,p7),axis = 0)

    negative_num = 0
    for i in range(len(ll2)):
        if ll2[i, 2] < 0:
            negative_num += 1
    print(negative_num)

    return ll2

def change_yaw(data_root, input_label_path, output_label_path):
    mark = 15
    data = np.loadtxt(os.path.join(data_root, input_label_path))

    for i in range(len(data)):
        if data[i][0] < 0.001:
            mark = i
            break
        elif np.abs(data[i][3] - data[i][4]) < 0.001:
            if data[i][5] > np.pi / 2:
                # print('square change!')
                # print(i, data[i][5])
                new_angle = data[i][5] - int(data[i][5] // (np.pi / 2)) * np.pi / 2
                # print(i, data[i][5])
            elif data[i][5] < 0:
                # print('square change!')
                # print(i, data[i][5])
                new_angle = data[i][5] + (int(data[i][5] // (-np.pi / 2)) + 1) * np.pi / 2
                # print(i, data[i][5])
            else:
                new_angle = np.copy(data[i][5])
            data[i][5] = new_angle * 2
        elif data[i][5] > np.pi:
            # print(i, 'rectangle change!')
            # print(data[i])
            data[i][5] = data[i][5] - np.pi
        elif data[i][5] < 0:
            # print(i, 'rectangle change!')
            # print(data[i])
            data[i][5] = data[i][5] + np.pi
    for i in range(len(data)):
        if np.abs(data[i][3] - data[i][4]) < 0.001:
            if data[i][5] > np.pi or data[i][5] < 0:
                print(j, i, '1 error')
        elif data[i][5] > np.pi or data[i][5] < 0:
            print(j, i, '2 error', data[i][5])

    cos = np.cos(2 * data[:, 5]).reshape(-1, 1)
    sin = np.sin(2 * data[:, 5]).reshape(-1, 1)
    cos[mark:, :] = 0
    sin[mark:, :] = 0

    data = np.concatenate((data, cos, sin), axis=1)

    data = np.delete(data, [5], axis=1)

    np.savetxt(os.path.join(data_root, output_label_path), data)

def rm_black_border(data_root, input_img_path, output_img_path):

    upper_border = 80
    lower_border = 560
    raw_img = cv2.imread(os.path.join(data_root, input_img_path))
    # print(raw_img.shape)
    img = raw_img[:, :, :3]
    new_img = np.copy(img[upper_border:lower_border, :, :])
    # print(new_img.shape)
    cv2.imwrite(os.path.join(data_root, output_img_path), new_img)

if __name__ == '__main__':

    # dataset format: bool, x, y, length, width, ori
    # data_root = '/home/ubuntu/Desktop/knolling_dataset/resnet_super/label/'
    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo-pose-small/labels/'

    # data = label_combine(data_root)
    num_label = num_img = 10

    # for j in range(num_label):
    #
    #     input_label_path = 'close_409_label/close_409_%d.csv' % j
    #     output_label_path = 'close_409_label/close_409_%d_train.csv' % j
    #     change_yaw(data_root, input_label_path, output_label_path)

    # data_root = '/home/ubuntu/Desktop/knolling_dataset/resnet_super/input/'
    for j in range(num_label):
        input_label_path = '%012d.txt' % j
        output_label_path = 'new/%012d.txt' % j
        change_yaw(data_root, input_label_path, output_label_path)