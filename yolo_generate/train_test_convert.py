import numpy as np

import shutil

total_num = 4000
ratio = 0.8

train_num = int(total_num * ratio)
test_num = int(total_num - train_num)
print(test_num)

for i in range(train_num):

    cur_path = '../YOLO_data/Dataset/image_yolo_409/IMG_test%s.png' % (i)
    tar_path = '../YOLO_data/Dataset/train_409/images/img%s.png' % i
    shutil.copy(cur_path, tar_path)

    cur_path = '../YOLO_data/Label/yolo_label_409/img%s.txt' % (i)
    tar_path = '../YOLO_data/train_409/labels/img%s.txt' % i
    shutil.copy(cur_path, tar_path)

for i in range(train_num, total_num):
    cur_path = '../YOLO_data/Dataset/image_yolo_409/IMG_test%s.png' % (i)
    tar_path = '../YOLO_data/Dataset/test_409/images/img%s.png' % i
    shutil.copy(cur_path, tar_path)

    cur_path = '../YOLO_data/Label/yolo_label_409/img%s.txt' % (i)
    tar_path = '../YOLO_data/test_409/labels/img%s.txt' % i
    shutil.copy(cur_path, tar_path)