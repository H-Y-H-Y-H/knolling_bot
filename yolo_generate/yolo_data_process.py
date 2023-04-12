import numpy as np
from yolo_data_collection_env import *

if __name__ == '__main__':

    data_root = '/home/ubuntu/Desktop/knolling_dataset/yolo/'

    mm2px = 530 / 0.34  # (1558)
    total_num = 20
    start_index = 2000
    num_item = 15

    for i in range(start_index, total_num + start_index):
        real_world_data = np.loadtxt(os.path.join(data_root, "label/real_world_label_409/img%s.txt") % i)
        corner_list = []
        label = []
        for j in range(num_item):
            # print(real_world_data[j])
            xpos1, ypos1 = real_world_data[j][0], real_world_data[j][1]
            lucky_list = real_world_data[j][2]
            # print(lucky_list)
            yawori = real_world_data[j][3]

            corn1, corn2, corn3, corn4 = find_corner(xpos1, ypos1, int(lucky_list), yawori)

            corner_list.append([corn1, corn2, corn3, corn4])

            corns = corner_list[j]

            col_offset = 320
            # row_offset = (0.154 - (0.3112 - 0.154)) * mm2px + 5
            row_offset = 0

            col_list = [int(mm2px * corns[0][1] + col_offset), int(mm2px * corns[3][1] + col_offset),
                        int(mm2px * corns[1][1] + col_offset), int(mm2px * corns[2][1] + col_offset)]
            row_list = [int(mm2px * corns[0][0] - row_offset), int(mm2px * corns[3][0] - row_offset),
                        int(mm2px * corns[1][0] - row_offset), int(mm2px * corns[2][0] - row_offset)]
            # print(col_list)
            # print(row_list)

            col_list = np.sort(col_list)
            row_list = np.sort(row_list)
            col_list[3] = col_list[3] + 7
            col_list[0] = col_list[0] - 7

            row_list[3] = row_list[3] + 7
            row_list[0] = row_list[0] - 7

            label_x = ((col_list[0] + col_list[3]) / 2)/640
            label_y = (((row_list[0] + row_list[3]) / 2)+86)/640

            length = (col_list[3] - col_list[0])/640
            width = (row_list[3] - row_list[0])/640

            # if lucky_list[j] == 2 and rdm_ori_yaw[j] < 0:
            #     rdm_ori_yaw[j] = rdm_ori_yaw[j] + np.pi/2

            element = []
            element.append(0)
            element.append(label_x)
            element.append(label_y)
            element.append(length)
            element.append(width)
            element = np.asarray(element)
            label.append(element)
            # print(label)

        np.savetxt(os.path.join(data_root, "label/yolo_label_409/img%s.txt") % i, label, fmt='%.8s')

    import shutil

    ratio = 0.8

    train_num = int(total_num * ratio)
    test_num = int(total_num - train_num)
    print(test_num)

    for i in range(start_index, start_index + train_num):
        cur_path = os.path.join(data_root, 'input/image_yolo_409/IMG_test%s.png') % (i)
        tar_path = os.path.join(data_root, 'train_409/images/img%s.png') % i
        shutil.copy(cur_path, tar_path)

        cur_path = os.path.join(data_root, 'label/yolo_label_409/img%s.txt') % (i)
        tar_path = os.path.join(data_root, 'train_409/labels/img%s.txt') % i
        shutil.copy(cur_path, tar_path)

    for i in range(start_index, start_index + test_num):
        cur_path = os.path.join(data_root, 'input/image_yolo_409/IMG_test%s.png') % (i)
        tar_path = os.path.join(data_root, 'test_409/images/img%s.png') % i
        shutil.copy(cur_path, tar_path)

        cur_path = os.path.join(data_root, 'label/yolo_label_409/img%s.txt') % (i)
        tar_path = os.path.join(data_root, 'test_409/labels/img%s.txt') % i
        shutil.copy(cur_path, tar_path)