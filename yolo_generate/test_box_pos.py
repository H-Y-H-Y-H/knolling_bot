import cv2
import os
import numpy as np

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
        cv2.namedWindow("zzz", 0)
        cv2.imshow('zzz', img)
        cv2.WaitKey(0)
        cv2.destroyAllWindows()

    return img

for i in range(20):
    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_datasets/yolo_pose4keypoints/'
    img = cv2.imread(os.path.join(data_root, "images/%012d.png") % i)
    label = np.loadtxt(os.path.join(data_root, "labels/%012d.txt") % i)