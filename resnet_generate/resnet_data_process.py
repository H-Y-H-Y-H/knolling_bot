import numpy as np
import os
import shutil
import numpy as np
import random
import cv2


def to_zero(path, filename):

    file = np.loadtxt(path + filename)

    for i in range(len(file)):

        for j in range(5):

            if file[i][7 + 8*j] == 0:
                file[i][8*j:7+8*j] = 0

    np.savetxt("Dataset/label/label_zero_30k.csv", file)


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

def clean_z(filename):
    file = np.loadtxt(filename)
    file2 = np.delete(file, 2, 1)
    np.savetxt("test.csv", file2)


if __name__ == '__main__':
    data_root = '/home/ubuntu/Desktop/knolling_dataset/resnet/'

    # data = label_combine(data_root)

    data = np.loadtxt(os.path.join(data_root, 'label/label_407_normal.csv'))

    # print(-86 // -90)

    for i in range(len(data)):
        if np.abs(data[i][4] - data[i][5]) < 0.001:
            if data[i][6] > np.pi / 2:
                print('square change!')
                print(i, data[i][6])
                new_angle = data[i][6] - int(data[i][6] // (np.pi / 2)) * np.pi / 2
                print(i, data[i][6])
            elif data[i][6] < 0:
                print('square change!')
                print(i, data[i][6])
                new_angle = data[i][6] + (int(data[i][6] // (-np.pi / 2)) + 1) * np.pi / 2
                print(i, data[i][6])
            else:
                new_angle = np.copy(data[i][6])
            data[i][6] = new_angle * 2
        elif data[i][6] > np.pi:
            print('rectangle change!')
            # print(data[i])
            data[i][6] = data[i][6] - np.pi
        elif data[i][6] < 0:
            print('rectangle change!')
            # print(data[i])
            data[i][6] = data[i][6] + np.pi

    # for i in range(len(data)):
    #     if np.abs(data[i][4] - data[i][5]) < 0.001:
    #         if data[i][6] > np.pi / 2 or data[i][6] < 0:
    #             print('1 error')
    #     elif data[i][6] > np.pi or data[i][6] < 0:
    #         print('2 error')

    cos = np.cos(2 * data[:, 6]).reshape(-1, 1)
    sin = np.sin(2 * data[:, 6]).reshape(-1, 1)

    data = np.concatenate((data, cos, sin), axis=1)

    data = np.delete(data, [0, 1, 2, 3, 6], axis=1)
    # for i in range(len(data)):
    #     if data[i][0] < 0.00001:
    #         print(data[i])
    #         print('ssssssssssssssssss')


    np.savetxt(os.path.join(data_root, 'label/label_407_normal_train.csv'), data)