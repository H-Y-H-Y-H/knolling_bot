import numpy as np
from yolo_data_collection_env_useless import *

def yolo_box(img, label):
    # label = [0,x,y,l,w],[0,x,y,l,w],...
    # label = label[:,1:]
    for i in range(len(label)):
        # label = label[i]
        # print('1',label)
        x_lt = int(label[i][1] * 640 - label[i][3] * 640/2)
        y_lt = int(label[i][2] * 480 - label[i][4] * 480/2)

        x_rb = int(label[i][1] * 640 + label[i][3] * 640/2)
        y_rb = int(label[i][2] * 480 + label[i][4] * 480/2)

        # img = img/255
        img = cv2.rectangle(img,(x_lt,y_lt),(x_rb,y_rb), color = (0,0,0), thickness = 1)
    cv2.namedWindow("zzz", 0)
    cv2.imshow('zzz', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img

def find_corner(x,y,type,yaw):

    gamma = yaw

    rot_z = [[np.cos(gamma), -np.sin(gamma)],
             [np.sin(gamma), np.cos(gamma)]]

    pos = [x, y]

    rot_z = np.asarray(rot_z)

    if type == 0:
        c1 = [16/2,16/2]
        c2 = [16/2,-16/2]
        c3 = [-16/2,16/2]
        c4 = [-16/2,-16/2]

    elif type == 1:
        c1 = [24/2,16/2]
        c2 = [24/2,-16/2]
        c3 = [-24/2,16/2]
        c4 = [-24/2,-16/2]

    elif type == 2:
        c1 = [32/2,16/2]
        c2 = [32/2,-16/2]
        c3 = [-32/2,16/2]
        c4 = [-32/2,-16/2]

    c1,c2,c3,c4 = np.asarray(c1),np.asarray(c2),np.asarray(c3),np.asarray(c4)
    c1 = c1/1000
    c2 = c2/1000
    c3 = c3/1000
    c4 = c4/1000

    corn1 = np.dot(rot_z,c1)
    corn2 = np.dot(rot_z,c2)
    corn3 = np.dot(rot_z,c3)
    corn4 = np.dot(rot_z,c4)

    corn11 = [corn1[0] + x, corn1[1] + y]
    corn22 = [corn2[0] + x, corn2[1] + y]
    corn33 = [corn3[0] + x, corn3[1] + y]
    corn44 = [corn4[0] + x, corn4[1] + y]

    return corn11, corn22, corn33, corn44

def object_detection():

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

def pose_estimation():

    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo-pose/'
    mm2px = 530 / 0.34  # (1558)
    total_num = 2000
    start_index = 2000
    num_item = 15

    for i in range(total_num):
        real_world_data = np.loadtxt(os.path.join(data_root, "label/real_world_label_418/%012d.txt") % i)
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

            col_offset = 0
            # row_offset = (0.154 - (0.3112 - 0.154)) * mm2px + 5
            row_offset = 0

            col_list = np.array([int(mm2px * corns[0][1] + col_offset), int(mm2px * corns[3][1] + col_offset),
                        int(mm2px * corns[1][1] + col_offset), int(mm2px * corns[2][1] + col_offset)])
            row_list = np.array([int(mm2px * corns[0][0] - row_offset), int(mm2px * corns[3][0] - row_offset),
                        int(mm2px * corns[1][0] - row_offset), int(mm2px * corns[2][0] - row_offset)])

            col_list = np.sort(col_list)
            row_list = np.sort(row_list)
            col_list[3] = col_list[3] + 7
            col_list[0] = col_list[0] - 7
            row_list[3] = row_list[3] + 7
            row_list[0] = row_list[0] - 7

            label_x = ((col_list[0] + col_list[3]) / 2)/640
            label_y = (((row_list[0] + row_list[3]) / 2) + 6)/480

            length = (col_list[3] - col_list[0])/640
            width = (row_list[3] - row_list[0])/480



            element = []
            element.append(0)
            element.append(label_x)
            element.append(label_y)
            element.append(length)
            element.append(width)
            element.append(corns[0])
            element.append(corns[1])
            element.append(corns[2])
            element.append(corns[3])
            element = np.asarray(element)
            label.append(element)
            # print(label)

        np.savetxt(os.path.join(data_root, "label/yolo_label_418/%012d.txt") % i, label, fmt='%.8s')

def find_keypoints(xpos, ypos, l, w, ori, mm2px):

    gamma = ori
    rot_z = [[np.cos(gamma), -np.sin(gamma)],
             [np.sin(gamma), np.cos(gamma)]]
    # rot_z = [[1, 0],
    #          [0, 1]]
    rot_z = np.asarray(rot_z)

    kp1 = np.asarray([l / 2, 0])
    kp2 = np.asarray([0, w / 2])
    kp3 = np.asarray([-l / 2, 0])
    kp4 = np.asarray([0, -w / 2])

    keypoint1 = np.dot(rot_z, kp1)
    keypoint2 = np.dot(rot_z, kp2)
    keypoint3 = np.dot(rot_z, kp3)
    keypoint4 = np.dot(rot_z, kp4)

    keypoint1 = np.array([((keypoint1[1] + ypos) * mm2px + 320) / 640, ((keypoint1[0] + xpos) * mm2px + 6) / 480, 1])
    keypoint2 = np.array([((keypoint2[1] + ypos) * mm2px + 320) / 640, ((keypoint2[0] + xpos) * mm2px + 6) / 480, 1])
    keypoint3 = np.array([((keypoint3[1] + ypos) * mm2px + 320) / 640, ((keypoint3[0] + xpos) * mm2px + 6) / 480, 1])
    keypoint4 = np.array([((keypoint4[1] + ypos) * mm2px + 320) / 640, ((keypoint4[0] + xpos) * mm2px + 6) / 480, 1])
    keypoints = np.concatenate((keypoint1, keypoint2, keypoint3, keypoint4), axis=0).reshape(-1, 3)

    return keypoints

def pose4keypoints(data_root, target_path):
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(target_path + 'images/', exist_ok=True)
    os.makedirs(target_path + 'labels/', exist_ok=True)
    mm2px = 530 / 0.34  # (1558)
    total_num = 100
    start_index = 2000
    num_item = 15

    import shutil
    for i in range(total_num):
        cur_path = os.path.join(data_root, "images/%012d.png") % i
        tar_path = os.path.join(target_path, "images/%012d.png") % i
        shutil.copy(cur_path, tar_path)


    for i in range(total_num):
        real_world_data = np.loadtxt(os.path.join(data_root, "labels/%012d.txt") % i)
        corner_list = []
        label_plot = []
        # if real_world_data[14, 0] == 1:
        #     pass
        # else:
        #     cut_id = np.where(real_world_data[:, 0] == 0)[0][0]
        #     real_world_data = np.delete(real_world_data, np.arange(cut_id, num_item), 0)
        real_world_data[:, 0] = 0

        label = []
        for j in range(len(real_world_data)):
            # print(real_world_data[j])
            xpos1, ypos1 = real_world_data[j][1], real_world_data[j][2]
            l, w = real_world_data[j][3], real_world_data[j][4]
            yawori = real_world_data[j][5]

            # ensure the yolo sequence!
            label_y = (xpos1 * mm2px + 6) / 480
            label_x = (ypos1 * mm2px + 320) / 640
            length = l * 3
            width = w * 3
            # ensure the yolo sequence!
            keypoints = find_keypoints(xpos1, ypos1, l, w, yawori, mm2px)
            keypoints_order = np.lexsort((keypoints[:, 0], keypoints[:, 1]))
            keypoints = keypoints[keypoints_order]

            element = np.concatenate(([0], [label_x, label_y], [length, width], keypoints.reshape(-1)))
            label.append(element)
            # print(label)
            if 0.8 < l / w < 1.2:
                lucky_list = 0
            elif 1.3 < (l / w) < 1.7:
                lucky_list = 1
            else: lucky_list = 2

            corn1, corn2, corn3, corn4 = find_corner(xpos1, ypos1, int(lucky_list), yawori)
            corner_list.append([corn1, corn2, corn3, corn4])
            corns = corner_list[j]

            col_offset = 320
            # row_offset = (0.154 - (0.3112 - 0.154)) * mm2px + 5
            row_offset = 0

            col_list = np.array([mm2px * corns[0][1] + col_offset, mm2px * corns[3][1] + col_offset,
                                 mm2px * corns[1][1] + col_offset, mm2px * corns[2][1] + col_offset])
            row_list = np.array([mm2px * corns[0][0] - row_offset, mm2px * corns[3][0] - row_offset,
                                 mm2px * corns[1][0] - row_offset, mm2px * corns[2][0] - row_offset])

            col_list = np.sort(col_list)
            row_list = np.sort(row_list)
            col_list[3] = col_list[3]
            col_list[0] = col_list[0]
            row_list[3] = row_list[3]
            row_list[0] = row_list[0]

            label_x_plot = ((col_list[0] + col_list[3]) / 2) / 640
            label_y_plot = (((row_list[0] + row_list[3]) / 2) + 6) / 480
            label_y = (xpos1 * mm2px + 6) / 480
            label_x = (ypos1 * mm2px + 320) / 640

            length_plot = (col_list[3] - col_list[0]) / 640
            width_plot = (row_list[3] - row_list[0]) / 480
            element_plot = []
            element_plot.append(0)
            element_plot.append(label_x_plot)
            element_plot.append(label_y_plot)
            element_plot.append(length_plot)
            element_plot.append(width_plot)
            element_plot = np.asarray(element_plot)
            label_plot.append(element_plot)
        # print('this is element\n', label)
        # print('this is plot element\n', label_plot)


        np.savetxt(os.path.join(target_path, "labels/%012d.txt") % i, label, fmt='%.8s')
        img = cv2.imread(os.path.join(data_root, "images/%012d.png") % i)
        # img = yolo_box(img, label_plot)


def train_test_split(data_root, target_path):

    import shutil
    ratio = 0.8
    total_num = 100
    train_num = int(total_num * ratio)
    test_num = int(total_num - train_num)
    print(train_num)
    print(test_num)

    os.makedirs(target_path + '/labels/train', exist_ok=True)
    os.makedirs(target_path + '/labels/val', exist_ok=True)
    os.makedirs(target_path + '/images/train', exist_ok=True)
    os.makedirs(target_path + '/images/val', exist_ok=True)

    for i in range(0, train_num):
        cur_path = os.path.join(data_root, 'images/%012d.png') % (i)
        tar_path = os.path.join(target_path, 'images/train/%012d.png') % i
        shutil.copy(cur_path, tar_path)

        cur_path = os.path.join(data_root, 'labels/%012d.txt') % (i)
        tar_path = os.path.join(data_root, 'labels/train/%012d.txt') % i
        shutil.copy(cur_path, tar_path)

    for i in range(train_num, total_num):
        cur_path = os.path.join(data_root, 'images/%012d.png') % (i)
        tar_path = os.path.join(target_path, 'images/val/%012d.png') % i
        shutil.copy(cur_path, tar_path)

        cur_path = os.path.join(data_root, 'labels/%012d.txt') % (i)
        tar_path = os.path.join(data_root, 'labels/val/%012d.txt') % i
        shutil.copy(cur_path, tar_path)

if __name__ == '__main__':

    # object_detection()

    # pose_estimation()

    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_small/'
    target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/yolo_pose4keypoints_small/'
    pose4keypoints(data_root, target_path)

    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/yolo_pose4keypoints_small/'
    target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/yolo_pose4keypoints_small/'
    train_test_split(data_root, target_path)