import numpy as np
# from yolo_data_collection_env import *
import os
import cv2
import shutil
from itertools import combinations, permutations

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

    img = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_AREA)
    # cv2.namedWindow("zzz", 0)
    cv2.imshow('zzz', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img

def find_corner(x, y, l, w, yaw):

    gamma = yaw
    rot_z = [[np.cos(gamma), -np.sin(gamma)],
             [np.sin(gamma), np.cos(gamma)]]
    rot_z = np.asarray(rot_z)

    # if type == 0:
    #     c1 = [16/2,16/2]
    #     c2 = [16/2,-16/2]
    #     c3 = [-16/2,16/2]
    #     c4 = [-16/2,-16/2]
    #
    # elif type == 1:
    #     c1 = [24/2,16/2]
    #     c2 = [24/2,-16/2]
    #     c3 = [-24/2,16/2]
    #     c4 = [-24/2,-16/2]
    #
    # elif type == 2:
    #     c1 = [32/2,16/2]
    #     c2 = [32/2,-16/2]
    #     c3 = [-32/2,16/2]
    #     c4 = [-32/2,-16/2]

    c1 = [l / 2, w / 2]
    c2 = [l / 2, -w / 2]
    c3 = [-l / 2, w / 2]
    c4 = [-l / 2, -w / 2]

    c1,c2,c3,c4 = np.asarray(c1),np.asarray(c2),np.asarray(c3),np.asarray(c4)

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

def color_segmentation(image, num_clusters, label):

    # Reshape the image to a 2D array of pixels

    image_part = []

    for i in range(len(label)):
        # label = label[i]
        # print('1',label)
        x_lt = int(label[i][1] * 640 - label[i][3] * 640/2)
        y_lt = int(label[i][2] * 480 - label[i][4] * 480/2)

        x_rb = int(label[i][1] * 640 + label[i][3] * 640/2)
        y_rb = int(label[i][2] * 480 + label[i][4] * 480/2)

        image_part = image[y_lt:y_rb, x_lt:x_rb, :]
        shape = image_part.shape[:2]

        pixels = image_part.reshape((-1, 3))

        # Convert the pixel values to floating point
        pixels = np.float32(pixels)

        # Define the criteria and apply k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Reshape the labels to the original image shape
        labels = labels.reshape(image_part.shape[:2])
        center_label = labels[int(shape[0] / 2), int(shape[1] / 2)]
        center_mask = np.array(labels == center_label)

        result = cv2.bitwise_and(image_part, image_part, mask=center_mask)
        result[np.where(result != 0)] = 30

        cv2.namedWindow("zzz", 0)
        cv2.imshow('zzz', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        bg_mask = cv2.bitwise_not(center_mask)
        image_part_bg = cv2.bitwise_and(image_part, image_part, mask=bg_mask)

        cv2.namedWindow("zzz", 0)
        cv2.imshow('zzz', image_part_bg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        new_image_part = cv2.add(image_part_bg, result)

        cv2.namedWindow("zzz", 0)
        cv2.imshow('zzz', new_image_part)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Create masks for each cluster label
        masks = []
        for j in range(num_clusters):
            masks.append(np.uint8(labels == j))

        # Show the segmented regions
        for j, mask in enumerate(masks):
            result = cv2.bitwise_and(image_part, image_part, mask=mask)
            cv2.namedWindow("Segmented Region " + str(j + 1), 0)
            cv2.imshow("Segmented Region " + str(j + 1), result)

    # cv2.namedWindow("zzz", 0)
    # cv2.imshow('zzz', image_part)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # image = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_AREA)
    # # cv2.namedWindow("zzz", 0)
    # cv2.imshow('zzz', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # for j in range(len(image_part)):
    #
    #     pixels = image_part[j].reshape((-1, 3))
    #
    #     # Convert the pixel values to floating point
    #     pixels = np.float32(pixels)
    #
    #     # Define the criteria and apply k-means clustering
    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #     _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #
    #     # Reshape the labels to the original image shape
    #     labels = labels.reshape(image_part[j].shape[:2])
    #
    #     # Create masks for each cluster label
    #     masks = []
    #     for j in range(num_clusters):
    #         masks.append(np.uint8(labels == j))
    #
    #     # Show the segmented regions
    #     for j, mask in enumerate(masks):
    #         result = cv2.bitwise_and(image_part[j], image_part[j], mask=mask)
    #         cv2.namedWindow("Segmented Region " + str(j + 1), 0)
    #         cv2.imshow("Segmented Region " + str(j + 1), result)

        # Wait for key press and exit gracefully

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # cv2.imwrite(data_root + 'images/%012d.png' % i, raw_img)

total_1 = 0
total_2 = 0
def find_keypoints(xpos, ypos, l, w, ori, mm2px, total_1, total_2):

    gamma = ori
    rot_z = [[np.cos(gamma), -np.sin(gamma)],
             [np.sin(gamma), np.cos(gamma)]]
    rot_z = np.asarray(rot_z)

    kp1 = np.asarray([l / 2, 0])
    kp2 = np.asarray([0, w / 2])
    kp3 = np.asarray([-l / 2, 0])
    kp4 = np.asarray([0, -w / 2])

    # here is simulation xy sequence, not yolo lw sequence!
    keypoint1 = np.dot(rot_z, kp1)
    keypoint2 = np.dot(rot_z, kp2)
    keypoint3 = np.dot(rot_z, kp3)
    keypoint4 = np.dot(rot_z, kp4)
    keypoints = np.concatenate((keypoint1, keypoint2, keypoint3, keypoint4), axis=0).reshape(-1, 2)

    # top_left_corner =

    # change the sequence of keypoints based on xy
    keypoints_order = np.lexsort((keypoints[:, 1], keypoints[:, 0]))[::-1]
    keypoints = keypoints[keypoints_order]

    # # fine-tuning the pos of keypoints to avoid the ambiguity!
    # if np.abs(keypoints[0, 0] - keypoints[1, 0]) < 0.001 and keypoints[0, 1] < keypoints[1, 1]:
    #     new_order = np.array([1, 0, 3, 2])
    #     keypoints = keypoints[new_order]
    #     total_1 += 1
    # elif np.abs(keypoints[1, 0] - keypoints[2, 0]) < 0.001 and keypoints[1, 1] < keypoints[2, 1]:
    #     new_order = np.array([0, 2, 1, 3])
    #     keypoints = keypoints[new_order]
    #     total_2 += 1
    # # keypoint1 = np.array([((keypoint1[1] + ypos) * mm2px + 320) / 640, ((keypoint1[0] + xpos) * mm2px + 6) / 480, 1])
    # # keypoint2 = np.array([((keypoint2[1] + ypos) * mm2px + 320) / 640, ((keypoint2[0] + xpos) * mm2px + 6) / 480, 1])
    # # keypoint3 = np.array([((keypoint3[1] + ypos) * mm2px + 320) / 640, ((keypoint3[0] + xpos) * mm2px + 6) / 480, 1])
    # # keypoint4 = np.array([((keypoint4[1] + ypos) * mm2px + 320) / 640, ((keypoint4[0] + xpos) * mm2px + 6) / 480, 1])
    # # keypoints = np.concatenate((keypoint1, keypoint2, keypoint3, keypoint4), axis=0).reshape(-1, 3)

    # # top-left, top-right, bottom-left, bottom-right
    # # change sequence of kpts based on the minimum value of the sum of the distance from the kpts to the corner
    # real_world_corner = np.array([[0, -0.17],
    #                               [0, 0.17],
    #                               [0.3, -0.17],
    #                               [0.3, 0.17]])
    # real_world_keypoints = np.copy(keypoints)
    # real_world_keypoints[:, 1] += ypos
    # real_world_keypoints[:, 0] += xpos
    #
    # keypoints_order = np.asarray(list(permutations([i for i in range(0, 4)], 4)))
    # min_dist = np.inf
    # best_order = np.arange(4)
    # for i in range(len(keypoints_order)):
    #     real_world_keypoints = real_world_keypoints[keypoints_order[i]]
    #     test_dist = np.sum(np.linalg.norm(real_world_keypoints - real_world_corner, axis=1))
    #     if test_dist < min_dist:
    #         min_dist = test_dist
    #         best_order = keypoints_order[i]
    #     else:
    #         pass
    # new_keypoints = keypoints[best_order]



    keypoints = np.concatenate(((((keypoints[:, 1] + ypos) * mm2px + 320) / 640).reshape(-1, 1),
                                (((keypoints[:, 0] + xpos) * mm2px + 6) / 480).reshape(-1, 1),
                                np.ones((4, 1))), axis=1).reshape(-1, 3)

    return keypoints, total_1, total_2

def pose4keypoints(data_root, target_path):
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(data_root + 'labels/', exist_ok=True)
    os.makedirs(data_root + 'images/', exist_ok=True)
    os.makedirs(target_path + 'images/', exist_ok=True)
    os.makedirs(target_path + 'labels/', exist_ok=True)
    mm2px = 530 / 0.34  # (1558)
    total_num = 60
    start_index = 2000
    num_item = 15

    for i in range(total_num):

        raw_img = cv2.imread(data_root + "origin_images/%012d.png" % i)

        # cv2.namedWindow("zzz", 0)
        # cv2.imshow('zzz', raw_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=int)
        # kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=int)
        # img = cv2.filter2D(raw_img,-1,kernel)

        # kernel_size = (3, 3)
        # img = cv2.blur(raw_img, kernel_size)

        # cv2.namedWindow("zzz", 0)
        # cv2.imshow('zzz', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print(i)
        cv2.imwrite(data_root + 'images/%012d.png' % i, raw_img)

    import warnings
    with warnings.catch_warnings(record=True) as w:

        # for i in range(15000):
        #     real_world_data = np.loadtxt(os.path.join(data_root, "labels/%012d.txt") % i)
        #     if real_world_data[14, 0] == 1:
        #         pass
        #     else:
        #         cut_id = np.where(real_world_data[:, 0] == 0)[0][0]
        #         real_world_data = np.delete(real_world_data, np.arange(cut_id, num_item), 0)
        #     real_world_data[:, 0] = 0
        #     np.savetxt(os.path.join(data_root, "labels_new/%012d.txt") % i, real_world_data, fmt='%.8s')
        # quit()

        total_1 = 0
        total_2 = 0
        for i in range(total_num):
            real_world_data = np.loadtxt(os.path.join(data_root, "origin_labels/%012d.txt") % i)
            if real_world_data[0, 5] == 0:
                real_world_data = np.delete(real_world_data, 5, axis=1)
            real_world_img = cv2.imread(data_root + "origin_images/%012d.png" % i)
            corner_list = []
            label_plot = []
            label = []

            print('this is index of images', i)
            for j in range(len(real_world_data)):
                # print(real_world_data[j])
                # print('this is index if legos', j)
                xpos1, ypos1 = real_world_data[j][1], real_world_data[j][2]
                l, w = real_world_data[j][3], real_world_data[j][4]
                yawori = real_world_data[j][5]
                if l < w:
                    l = real_world_data[j][4]
                    w = real_world_data[j][3]
                    if yawori > np.pi / 2:
                        yawori = yawori - np.pi / 2
                    else:
                        yawori = yawori + np.pi / 2

                # ensure the yolo sequence!
                label_y = (xpos1 * mm2px + 6) / 480
                label_x = (ypos1 * mm2px + 320) / 640
                length = l * 3
                width = w * 3
                # ensure the yolo sequence!
                keypoints, total_1, total_2 = find_keypoints(xpos1, ypos1, l, w, yawori, mm2px, total_1, total_2)

                element = np.concatenate(([0], [label_x, label_y], [length, width], keypoints.reshape(-1)))
                # print(label)

                corn1, corn2, corn3, corn4 = find_corner(xpos1, ypos1, l, w, yawori)
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
                col_list[3] = col_list[3] + 10
                col_list[0] = col_list[0] - 10
                row_list[3] = row_list[3] + 10
                row_list[0] = row_list[0] - 10

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

                # change the lw to yolo_lw in label!!!!!!
                element[3] = length_plot
                element[4] = width_plot
                label.append(element)

            label = np.asarray(label)
            # print('this is element\n', label)
            # print('this is plot element\n', label_plot)


            np.savetxt(os.path.join(data_root, "labels/%012d.txt") % i, label, fmt='%.8s')
            # img = cv2.imread(os.path.join(data_root, "images/%012d.png") % i)
            img = yolo_box(real_world_img, label_plot)
            # color_segmentation(real_world_img, 5, label_plot)
        print('this is total_1', total_1)
        print('this is total_2', total_2)

def manual_pose4keypoints(data_root, target_path):
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(target_path + 'images/', exist_ok=True)
    os.makedirs(target_path + 'labels/', exist_ok=True)
    mm2px = 530 / 0.34  # (1558)
    img_index = input('Enter the index of img:')
    img_index = int(img_index)

    cur_path = os.path.join(data_root, "images/%012d.png") % img_index
    tar_path = os.path.join(target_path, "images/%012d.png") % img_index
    shutil.copy(cur_path, tar_path)

    # num_item = input('Enter the num of boxes in one img:')
    #
    # data_total = []
    # for i in range(int(num_item)):
    #     data = input('Enter the xylwori of one img (five datas in total):').split(" ")
    #     data_total.append(np.array([float(j) for j in data]))
    # data_total = np.concatenate((np.zeros((len(data_total), 1)), np.asarray(data_total)), axis=1)
    # data_total[:, 5] = data_total[:, 5] / 180 * np.pi
    # print(data_total)
    # np.savetxt(os.path.join(data_root, "labels/%012d.txt") % img_index, data_total, fmt='%.8s')

    data_total = np.loadtxt(os.path.join(data_root, "labels/%012d.txt") % img_index)
    data_total[:, 5] = data_total[:, 5] / 180 * np.pi

    corner_list = []
    label_plot = []
    label = []
    total_1 = 0
    total_2 = 0
    for j in range(len(data_total)):
        print(data_total[j])
        # print('this is index if legos', j)
        xpos1, ypos1 = data_total[j][1], data_total[j][2]
        l, w = data_total[j][3], data_total[j][4]
        yawori = data_total[j][5]
        if l < w:
            l = data_total[j][4]
            w = data_total[j][3]
            if yawori > np.pi / 2:
                yawori = yawori - np.pi / 2
            else:
                yawori = yawori + np.pi / 2

        # ensure the yolo sequence!
        label_y = (xpos1 * mm2px + 6) / 480
        label_x = (ypos1 * mm2px + 320) / 640
        length = l * 3
        width = w * 3
        # ensure the yolo sequence!
        keypoints, total_1, total_2 = find_keypoints(xpos1, ypos1, l, w, yawori, mm2px, total_1, total_2)
        # keypoints_order = np.lexsort((keypoints[:, 0], keypoints[:, 1]))[::-1]
        # keypoints = keypoints[keypoints_order]

        element = np.concatenate(([0], [label_x, label_y], [length, width], keypoints.reshape(-1)))
        # print(label)

        corn1, corn2, corn3, corn4 = find_corner(xpos1, ypos1, l, w, yawori)
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

        # change the lw to yolo_lw in label!!!!!!
        element[3] = length_plot
        element[4] = width_plot
        label.append(element)

    label = np.asarray(label)
    print('this is element\n', label)
    # print('this is plot element\n', label_plot)
    img = cv2.imread(os.path.join(data_root, "images/%012d.png") % img_index)
    img = yolo_box(img, label_plot)

    np.savetxt(os.path.join(target_path, "labels/%012d.txt") % img_index, label, fmt='%.8s')

def train_test_split(data_root, target_path):

    import shutil
    ratio = 0.8
    total_num = 60
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
        tar_path = os.path.join(target_path, 'labels/train/%012d.txt') % i
        shutil.copy(cur_path, tar_path)

    for i in range(train_num, total_num):
        cur_path = os.path.join(data_root, 'images/%012d.png') % (i)
        tar_path = os.path.join(target_path, 'images/val/%012d.png') % i
        shutil.copy(cur_path, tar_path)

        cur_path = os.path.join(data_root, 'labels/%012d.txt') % (i)
        tar_path = os.path.join(target_path, 'labels/val/%012d.txt') % i
        shutil.copy(cur_path, tar_path)

if __name__ == '__main__':

    # data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_510_tuning/'
    # target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/yolo_pose4keypoints_510_tuning/'
    # manual_pose4keypoints(data_root, target_path)

    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_tuning/'
    target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/yolo_pose4keypoints_521_tuning/'
    pose4keypoints(data_root, target_path)

    data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints_tuning/'
    target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/datasets/yolo_pose4keypoints_521_tuning/'
    train_test_split(data_root, target_path)

    # data_root = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/yolo_pose4keypoints/labels/'
    # files = os.listdir(data_root)
    # print(files)
    #
    # l = [i for i in files if 'normal' in i]
    #
    # for m in l:
    #     os.remove(data_root + m)
