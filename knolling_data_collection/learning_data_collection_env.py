import numpy as np
import pyrealsense2 as rs
from sort_data_collection import Sort_objects
import pybullet_data as pd
import math
import random
# from turdf import *
import socket
import pybullet as p
import os
import cv2
# from cam_obs_yolov8 import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Polygon
from tqdm import tqdm
from urdfpy import URDF

torch.manual_seed(42)
# np.random.seed(202)
random.seed(202)


class Arm:

    def __init__(self, is_render=True):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = '../urdf/'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render
        if self.is_render:
            # p.connect(p.GUI, options="--width=1280 --height=720")
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.num_motor = 5

        self.low_scale = np.array([0.03, -0.14, 0.0, - np.pi / 2, 0])
        self.high_scale = np.array([0.27, 0.14, 0.05, np.pi / 2, 0.4])
        self.low_act = -np.ones(5)
        self.high_act = np.ones(5)
        self.x_low_obs = self.low_scale[0]
        self.x_high_obs = self.high_scale[0]
        self.y_low_obs = self.low_scale[1]
        self.y_high_obs = self.high_scale[1]
        self.z_low_obs = self.low_scale[2]
        self.z_high_obs = self.high_scale[2]
        self.table_boundary = 0.03

        self.lateral_friction = 1
        self.spinning_friction = 1
        self.rolling_friction = 0

        self.camera_parameters = {
            'width': 640.,
            'height': 480,
            'fov': 42,
            'near': 0.1,
            'far': 100.,
            'eye_position': [0.59, 0, 0.8],
            'target_position': [0.55, 0, 0.05],
            'camera_up_vector':
                [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.
        }
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.15, 0, 0],
            distance=0.4,
            yaw=90,
            pitch=-90,
            roll=0,
            upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] / self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

        if random.uniform(0, 1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(1, 2), 5])
        else:
            p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(-2, -1), 5])
        p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(1, 2), 5],
                                   shadowMapResolution=8192, shadowMapIntensity=np.random.randint(5, 8) / 10)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())

    def get_parameters(self, lego_num=None, area_num=None, ratio_num=None, boxes_index=None,
                       total_offset=None,
                       gap_item=0.03, gap_block=0.02,
                       real_operate=False, obs_order='1',
                       random_offset=False, check_detection_loss=None, obs_img_from=None, use_yolo_pos=True):

        # self.lego_num = lego_num
        self.total_offset = total_offset
        self.area_num = area_num
        self.ratio_num = ratio_num
        self.gap_item = gap_item
        self.gap_block = gap_block
        self.real_operate = real_operate
        self.obs_order = obs_order
        self.random_offset = random_offset
        self.num_list = lego_num
        self.check_detection_loss = check_detection_loss
        self.obs_img_from = obs_img_from
        self.use_yolo_pos = use_yolo_pos
        self.boxes_index = boxes_index

    def get_obs(self, order, evaluation):

        def get_images():
            (width, length, image, _, _) = p.getCameraImage(width=640,
                                                            height=480,
                                                            viewMatrix=self.view_matrix,
                                                            projectionMatrix=self.projection_matrix,
                                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
            return image

        if order == 'images':
            image = get_images()
            return image

    def label2image(self, labels_data, index_flag):
        print(index_flag)
        index_flag = index_flag.reshape(2, -1)
        labels_data = labels_data.reshape(-1, 5)
        pos_data = labels_data[:, :2]
        pos_data = np.concatenate((pos_data, np.zeros(len(pos_data)).reshape(-1, 1)), axis=1)
        lw_data = labels_data[:, 2:4]
        ori_data = labels_data[:, 4]
        ori_data = np.concatenate((np.zeros((len(ori_data), 2)), ori_data.reshape(-1, 1)), axis=1)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Draw workspace lines
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])

        baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", basePosition=[0, -0.2, 0], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        # self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm1.urdf",
        #                          basePosition=[-0.08, 0, 0.02], useFixedBase=True,
        #                          flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(self.urdf_path + "img_1.png")
        p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5,
                         angularDamping=0.5)
        # p.changeDynamics(self.arm_id, 7, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        # p.changeDynamics(self.arm_id, 8, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId,
                            rgbaColor=[np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), np.random.uniform(0.9, 1),
                                       1])

        ################### recover urdf boxes based on lw_data ###################
        boxes = []
        xyz_list = []
        new_pos_data = []
        new_ori_data = []
        for i in range(len(index_flag[0])):
            boxes.append(URDF.load('../urdf/box_generator/box_%d.urdf' % index_flag[0, i]))
            xyz_list.append(boxes[i].links[0].visuals[0].geometry.box.size)
        # for i in range(50):
        #     boxes.append(URDF.load('../urdf/box_generator/box_%d.urdf' % i))
        #     xyz_list.append(boxes[i].links[0].visuals[0].geometry.box.size)
        lego_idx = []
        for i in range(len(lw_data)):
            for j in range(len(xyz_list)):
                if (np.abs(xyz_list[j][0] - lw_data[i][0]) < 0.0001 and np.abs(
                        xyz_list[j][1] - lw_data[i][1]) < 0.0001) or \
                        np.abs(xyz_list[j][1] - lw_data[i][0]) < 0.0001 and np.abs(
                    xyz_list[j][0] - lw_data[i][1]) < 0.0001:
                    print(f'this is matching urdf{j}')
                    print(pos_data[i])
                    print(xyz_list[j])
                    print(ori_data[i])
                    pos_data[i, 2] += 0.006
                    lego_idx.append(
                        p.loadURDF(self.urdf_path + f"box_generator/box_{int(index_flag[0, j])}.urdf",
                                   basePosition=pos_data[i],
                                   baseOrientation=p.getQuaternionFromEuler(ori_data[i]), useFixedBase=False,
                                   flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
        ################### recover urdf boxes based on lw_data ###################

        return self.get_obs('images', None)

    def reset(self):

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects()
        self.obj_idx = []
        if self.real_operate == False:
            self.xyz_list, _, _, self.all_index, self.transform_flag = items_sort.get_data_virtual(self.area_num,
                                                                                                   self.ratio_num,
                                                                                                   self.num_list,
                                                                                                   self.boxes_index)
            restrict = np.max(self.xyz_list)
            gripper_height = 0.012
            last_pos = np.array([[0, 0, 1]])

            ############## collect ori and pos to calculate the error of detection ##############
            collect_ori = []
            collect_pos = []
            ############## collect ori and pos to calculate the error of detection ##############

            for i in range(len(self.all_index)):
                for j in range(len(self.all_index[i])):
                    #         pass
                    # for i in range(len(self.grasp_order)):
                    #     for j in range(self.num_list[self.grasp_order[i]]):

                    rdm_pos = np.array([random.uniform(self.x_low_obs, self.x_high_obs),
                                        random.uniform(self.y_low_obs, self.y_high_obs), 0.0])
                    ori = [0, 0, random.uniform(0, np.pi)]
                    # ori = [0, 0, 0]
                    collect_ori.append(ori)
                    check_list = np.zeros(last_pos.shape[0])

                    while 0 in check_list:
                        rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs),
                                   random.uniform(self.y_low_obs, self.y_high_obs), 0.0]
                        for z in range(last_pos.shape[0]):
                            if np.linalg.norm(last_pos[z] - rdm_pos) < restrict + gripper_height:
                                check_list[z] = 0
                            else:
                                check_list[z] = 1
                    collect_pos.append(rdm_pos)

                    last_pos = np.append(last_pos, [rdm_pos], axis=0)

            collect_ori = np.asarray(collect_ori)
            collect_pos = np.asarray(collect_pos)
            self.check_ori = collect_ori[:, 2]
            self.check_pos = collect_pos[:, :2]
        for i in range(60):
            p.stepSimulation()
        # return self.get_obs('images', None)
        return self.check_pos, self.check_ori, self.xyz_list[:, :2], self.transform_flag

    def change_config(self):  # this is main function!!!!!!!!!

        p.resetSimulation()
        p.setGravity(0, 0, -10)

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])

        baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", basePosition=[0, -0.2, 0], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        # self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm1.urdf",
        #                          basePosition=[-0.08, 0, 0.02], useFixedBase=True,
        #                          flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(self.urdf_path + "img_1.png")
        p.changeDynamics(baseid, -1, lateralFriction=self.lateral_friction, frictionAnchor=True)
        # p.changeDynamics(self.arm_id, 7, lateralFriction=self.lateral_friction, frictionAnchor=True)
        # p.changeDynamics(self.arm_id, 8, lateralFriction=self.lateral_friction, frictionAnchor=True)
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId,
                            rgbaColor=[np.random.uniform(0.9, 1), np.random.uniform(0.9, 1), np.random.uniform(0.9, 1),
                                       1])

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects()
        if self.real_operate == False:
            self.xyz_list, _, _, self.all_index, self.transform_flag = items_sort.get_data_virtual(self.area_num,
                                                                                                   self.ratio_num,
                                                                                                   self.num_list,
                                                                                                   self.boxes_index)

        # print(f'this is standard trim xyz list\n {self.xyz_list}')
        # print(f'this is standard trim index list\n {self.all_index}')

        def calculate_items(item_num, item_xyz):

            min_xy = np.ones(2) * 100
            best_item_config = []
            item_iteration = 100
            item_odd_flag = False
            all_item_x = 100
            all_item_y = 100

            fac = []  # 定义一个列表存放因子
            for i in range(1, item_num + 1):
                if item_num % i == 0:
                    fac.append(i)
                    continue

            if item_num % 2 != 0 and len(fac) == 2:  # its odd! we should generate the factor again!
                item_num += 1
                item_odd_flag = True
                fac = []  # 定义一个列表存放因子
                for i in range(1, item_num + 1):
                    if item_num % i == 0:
                        fac.append(i)
                        continue

            item_sequence = np.random.choice(len(item_xyz), len(item_xyz), replace=False)
            if item_odd_flag == True:
                item_sequence = np.append(item_sequence, item_sequence[-1])

            for j in range(len(fac)):
                item_num_row = int(fac[j])
                item_num_column = int(item_num / item_num_row)
                item_sequence = item_sequence.reshape(item_num_row, item_num_column)
                item_min_x = 0
                item_min_y = 0

                for r in range(item_num_row):
                    new_row = item_xyz[item_sequence[r, :]]
                    item_min_x = item_min_x + np.max(new_row, axis=0)[0]

                for c in range(item_num_column):
                    new_column = item_xyz[item_sequence[:, c]]
                    item_min_y = item_min_y + np.max(new_column, axis=0)[1]

                item_min_x = item_min_x + (item_num_row - 1) * self.gap_item
                item_min_y = item_min_y + (item_num_column - 1) * self.gap_item

                if item_min_x + item_min_y < all_item_x + all_item_y:
                    best_item_config = [item_num_row, item_num_column]
                    all_item_x = item_min_x
                    all_item_y = item_min_y
                    min_xy = np.array([all_item_x, all_item_y])

            return min_xy, best_item_config, item_odd_flag

        def calculate_block():  # first: calculate, second: reorder!

            min_result = []
            best_config = []
            item_odd_list = []
            for i in range(len(self.all_index)):
                item_index = self.all_index[i]
                item_xyz = self.xyz_list[item_index, :]
                item_num = len(item_index)
                xy, config, odd = calculate_items(item_num, item_xyz)
                # print(f'this is min xy {xy}')
                min_result.append(list(xy))
                # print(f'this is the best item config\n {config}')
                best_config.append(list(config))
                item_odd_list.append(odd)
            min_result = np.asarray(min_result).reshape(-1, 2)
            best_config = np.asarray(best_config).reshape(-1, 2)
            item_odd_list = np.asarray(item_odd_list)
            # print(best_config)

            # 安排总的摆放
            iteration = 100
            all_num = best_config.shape[0]
            all_x = 100
            all_y = 100
            odd_flag = False

            fac = []  # 定义一个列表存放因子
            for i in range(1, all_num + 1):
                if all_num % i == 0:
                    fac.append(i)
                    continue

            if all_num % 2 != 0 and len(fac) == 2:  # its odd! we should generate the factor again!
                all_num += 1
                odd_flag = True
                fac = []  # 定义一个列表存放因子
                for i in range(1, all_num + 1):
                    if all_num % i == 0:
                        fac.append(i)
                        continue

            for i in range(iteration):
                sequence = np.random.choice(best_config.shape[0], size=len(self.all_index), replace=False)
                if odd_flag == True:
                    sequence = np.append(sequence, sequence[-1])
                else:
                    pass
                zero_or_90 = np.random.choice(np.array([0, 90]))

                for j in range(len(fac)):

                    min_xy = np.copy(min_result)
                    # print(f'this is the min_xy before rotation\n {min_xy}')

                    num_row = int(fac[j])
                    num_column = int(all_num / num_row)
                    sequence = sequence.reshape(num_row, num_column)
                    min_x = 0
                    min_y = 0
                    rotate_flag = np.full((num_row, num_column), False, dtype=bool)
                    # print(f'this is {sequence}')

                    for r in range(num_row):
                        for c in range(num_column):
                            new_row = min_xy[sequence[r][c]]
                            zero_or_90 = np.random.choice(np.array([0, 90]))
                            if zero_or_90 == 90:
                                rotate_flag[r][c] = True
                                temp = new_row[0]
                                new_row[0] = new_row[1]
                                new_row[1] = temp

                        # insert 'whether to rotate' here
                    for r in range(num_row):
                        new_row = min_xy[sequence[r, :]]
                        min_x = min_x + np.max(new_row, axis=0)[0]

                    for c in range(num_column):
                        new_column = min_xy[sequence[:, c]]
                        min_y = min_y + np.max(new_column, axis=0)[1]

                    if min_x + min_y < all_x + all_y:
                        best_all_config = sequence
                        all_x = min_x
                        all_y = min_y
                        best_rotate_flag = rotate_flag
                        best_min_xy = np.copy(min_xy)
            # print(f'in iteration{i}, the min all_x and all_y are {all_x} {all_y}')
            # print('this is best all sequence', best_all_config)

            return reorder_block(best_config, best_all_config, best_rotate_flag, best_min_xy, odd_flag, item_odd_list)

        def reorder_item(best_config, start_pos, index_block, item_index, item_xyz, rotate_flag, item_odd_list):

            # initiate the pos and ori
            # we don't analysis these imported oris
            # we directly define the ori is 0 or 90 degree, depending on the algorithm.
            item_row = best_config[index_block][0]
            item_column = best_config[index_block][1]
            item_odd_flag = item_odd_list[index_block]
            if item_odd_flag == True:
                item_pos = np.zeros([len(item_index) + 1, 3])
                item_ori = np.zeros([len(item_index) + 1, 3])
                item_xyz = np.append(item_xyz, item_xyz[-1]).reshape(-1, 2)
                index_temp = np.arange(item_pos.shape[0] - 1)
                index_temp = np.append(index_temp, index_temp[-1]).reshape(item_row, item_column)
            else:
                item_pos = np.zeros([len(item_index), 3])
                item_ori = np.zeros([len(item_index), 3])
                index_temp = np.arange(item_pos.shape[0]).reshape(item_row, item_column)

            # the initial position of the first items

            if rotate_flag == True:

                temp = np.copy(item_xyz[:, 0])
                item_xyz[:, 0] = item_xyz[:, 1]
                item_xyz[:, 1] = temp
                item_ori[:, 2] = np.pi / 2
                # print(item_ori)
                temp = item_row
                item_row = item_column
                item_column = temp
                index_temp = index_temp.transpose()
            else:
                item_ori[:, 2] = 0

            # start_pos[0] = start_pos[0] + np.max(item_xyz, axis=0)[0] / 2
            # start_pos[1] = start_pos[1] + np.max(item_xyz, axis=0)[1] / 2
            #
            #
            # for j in range(item_row):
            #     for k in range(item_column):
            #         ################### check whether to transform for each item in each block!################
            #         if self.transform_flag[item_index[index_temp[j][k]]] == 1:
            #             print(f'the index {item_index[index_temp[j][k]]} should be rotated because of transformation')
            #             item_ori[index_temp[j][k], 2] -= np.pi / 2
            #         ################### check whether to transform for each item in each block!################
            #         x_2x2 = start_pos[0] + (item_xyz[index_temp[j][k]][0]) * j + self.gap_item * j
            #         y_2x2 = start_pos[1] + (item_xyz[index_temp[j][k]][1]) * k + self.gap_item * k
            #         item_pos[index_temp[j][k]][0] = x_2x2
            #         item_pos[index_temp[j][k]][1] = y_2x2

            start_item_x = np.array([start_pos[0]])
            start_item_y = np.array([start_pos[1]])
            previous_start_item_x = start_item_x
            previous_start_item_y = start_item_y

            for m in range(item_row):
                new_row = item_xyz[index_temp[m, :]]
                start_item_x = np.append(start_item_x,
                                         (previous_start_item_x + np.max(new_row, axis=0)[0] + self.gap_item))
                previous_start_item_x = (previous_start_item_x + np.max(new_row, axis=0)[0] + self.gap_item)
            start_item_x = np.delete(start_item_x, -1)

            for n in range(item_column):
                new_column = item_xyz[index_temp[:, n]]
                start_item_y = np.append(start_item_y,
                                         (previous_start_item_y + np.max(new_column, axis=0)[1] + self.gap_item))
                previous_start_item_y = (previous_start_item_y + np.max(new_column, axis=0)[1] + self.gap_item)
            start_item_y = np.delete(start_item_y, -1)

            x_pos, y_pos = np.copy(start_pos)[0], np.copy(start_pos)[1]

            for j in range(item_row):
                for k in range(item_column):
                    if item_odd_flag == True and j == item_row - 1 and k == item_column - 1:
                        break
                    ################### check whether to transform for each item in each block!################
                    if self.transform_flag[item_index[index_temp[j][k]]] == 1:
                        # print(f'the index {item_index[index_temp[j][k]]} should be rotated because of transformation')
                        item_ori[index_temp[j][k], 2] -= np.pi / 2
                    ################### check whether to transform for each item in each block!################
                    x_pos = start_item_x[j] + (item_xyz[index_temp[j][k]][0]) / 2
                    y_pos = start_item_y[k] + (item_xyz[index_temp[j][k]][1]) / 2
                    item_pos[index_temp[j][k]][0] = x_pos
                    item_pos[index_temp[j][k]][1] = y_pos
            if item_odd_flag == True:
                item_pos = np.delete(item_pos, -1, axis=0)
                item_ori = np.delete(item_ori, -1, axis=0)
            else:
                pass
            # print('this is the shape of item pos', item_pos.shape)
            return item_ori, item_pos

        def reorder_block(best_config, best_all_config, best_rotate_flag, min_xy, odd_flag, item_odd_list):

            # print(f'the best configuration of all items is\n {best_all_config}')
            # print(f'the best configuration of each kind of items is\n {best_config}')
            # print(f'the rotate of each block of items is\n {best_rotate_flag}')
            # print(f'this is the min_xy of each kind of items after rotation\n {min_xy}')

            num_all_row = best_all_config.shape[0]
            num_all_column = best_all_config.shape[1]

            start_x = [0]
            start_y = [0]
            previous_start_x = 0
            previous_start_y = 0

            for m in range(num_all_row):
                new_row = min_xy[best_all_config[m, :]]
                # print(new_row)
                # print(np.max(new_row, axis=0)[0])
                start_x.append((previous_start_x + np.max(new_row, axis=0)[0] + self.gap_block))
                previous_start_x = (previous_start_x + np.max(new_row, axis=0)[0] + self.gap_block)
            start_x = np.delete(start_x, -1)
            # print(f'this is start_x {start_x}')

            for n in range(num_all_column):
                new_column = min_xy[best_all_config[:, n]]
                # print(new_column)
                # print(np.max(new_column, axis=0)[1])
                start_y.append((previous_start_y + np.max(new_column, axis=0)[1] + self.gap_block))
                previous_start_y = (previous_start_y + np.max(new_column, axis=0)[1] + self.gap_block)
            start_y = np.delete(start_y, -1)
            # print(f'this is start_y {start_y}')d

            # determine the start position per item
            item_pos = np.zeros([len(self.xyz_list), 3])
            item_ori = np.zeros([len(self.xyz_list), 3])
            # print(self.xyz_list[self.all_index[0]])
            # print(self.all_index)
            for m in range(num_all_row):
                for n in range(num_all_column):
                    if odd_flag == True and m == num_all_row - 1 and n == num_all_column - 1:
                        break  # this is the redundancy block
                    item_index = self.all_index[best_all_config[m][n]]  # determine the index of blocks

                    # print('try', item_index)
                    item_xyz = self.xyz_list[item_index, :]
                    # print('try', item_xyz)
                    start_pos = np.asarray([start_x[m], start_y[n]])
                    index_block = best_all_config[m][n]
                    rotate_flag = best_rotate_flag[m][n]

                    ori, pos = reorder_item(best_config, start_pos, index_block, item_index, item_xyz, rotate_flag,
                                            item_odd_list)
                    # print('tryori', ori)
                    # print('trypos', pos)
                    item_pos[item_index] = pos
                    item_ori[item_index] = ori

            return item_pos, item_ori  # pos_list, ori_list

        # determine the center of the tidy configuration
        self.items_pos_list, self.items_ori_list = calculate_block()
        x_low = np.min(self.items_pos_list, axis=0)[0]
        x_high = np.max(self.items_pos_list, axis=0)[0]
        y_low = np.min(self.items_pos_list, axis=0)[1]
        y_high = np.max(self.items_pos_list, axis=0)[1]
        center = np.array([(x_low + x_high) / 2, (y_low + y_high) / 2, 0])
        x_length = abs(x_high - x_low)
        y_length = abs(y_high - y_low)
        # print(x_low, x_high, y_low, y_high)
        if self.random_offset == True:
            self.total_offset = np.array([random.uniform(self.x_low_obs + x_length / 2, self.x_high_obs - x_length / 2),
                                          random.uniform(self.y_low_obs + y_length / 2, self.y_high_obs - y_length / 2),
                                          0.0])
        else:
            pass
        self.items_pos_list = self.items_pos_list + self.total_offset
        self.manipulator_after = np.concatenate((self.items_pos_list, self.items_ori_list), axis=1)

        # return self.get_obs('images', None)
        return self.items_pos_list[:, :2], self.items_ori_list[:, 2], self.xyz_list[:, :2], self.transform_flag


if __name__ == '__main__':

    command = 'recover'
    before_after = 'after'

    evaluations = 10000
    range_low = 4
    range_high = 5
    total_urdf = 50

    area_num = 3
    ratio_num = 2

    target_path = '../data/learning_data_502/'
    images_log_path = target_path + 'images_%s/' % before_after
    os.makedirs(images_log_path, exist_ok=True)

    if command == 'recover':

        env = Arm(is_render=True)
        for i in range(range_low, range_high):
            data = np.loadtxt(target_path + 'labels_%s/num_%d.txt' % (before_after, i))
            index_flag = np.loadtxt(target_path + 'index_flag/num_%s_flag.txt' % i).reshape(-1, i * 2)

            if len(data.shape) == 1:
                data = data.reshape(1, len(data))

            lego_num = i

            for j in range(len(data)):
                env.get_parameters(lego_num=lego_num)
                print(f'this is data {j}')

                image = env.label2image(data[j], index_flag[j])
                image = image[..., :3]
                # print('this is shape of image', image.shape)
                # image = np.transpose(image, (2, 0, 1))
                # temp = image[:, :, 2]
                # image[:, :, 2] = image[:, :, 0]
                # image[:, :, 0] = temp
                cv2.namedWindow('zzz', 0)
                cv2.imshow("zzz", image)
                cv2.waitKey()
                cv2.destroyAllWindows()

                cv2.imwrite(images_log_path + '%d_%d.png' % (i, j), image)

    if command == 'knolling':

        target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/learning_data_5categories/'
        before_path = target_path + 'labels_before/'
        after_path = target_path + 'labels_after/'
        index_flag_path = target_path + 'index_flag/'
        os.makedirs(before_path, exist_ok=True)
        os.makedirs(after_path, exist_ok=True)
        os.makedirs(index_flag_path, exist_ok=True)

        env = Arm(is_render=False)
        for num in range(range_low, range_high):

            data_before = []
            data_after = []
            index_flag = []
            for j in tqdm(range(evaluations)):

                # if j == 1397:
                #     print('here!')

                lego_num = num
                boxes_index = np.random.choice(50, lego_num)

                total_offset = [0.016, -0.17 + 0.016, 0]
                gap_item = 0.01
                gap_block = 0.02
                random_offset = False
                real_operate = False
                obs_order = 'sim_image_obj'
                check_detection_loss = False
                obs_img_from = 'env'
                use_yolo_pos = False

                env.get_parameters(lego_num=lego_num, area_num=area_num, ratio_num=ratio_num, boxes_index=boxes_index,
                                   total_offset=total_offset,
                                   gap_item=gap_item, gap_block=gap_block,
                                   real_operate=real_operate, obs_order=obs_order,
                                   random_offset=random_offset, check_detection_loss=check_detection_loss,
                                   obs_img_from=obs_img_from, use_yolo_pos=use_yolo_pos)

                pos_before, ori_before, xy_before, transform_before = env.reset()
                data_before.append(
                    np.concatenate((pos_before, xy_before, ori_before.reshape(-1, 1)), axis=1).reshape(-1, ))
                pos_after, ori_after, xy_after, transform_after = env.change_config()
                data_after.append(np.concatenate((pos_after, xy_after, ori_after.reshape(-1, 1)), axis=1).reshape(-1))

                index_flag.append(np.concatenate((boxes_index, transform_after)))

                # print(data_before[j].shape[0])
                # print(data_after[j].shape[0])
                if data_before[j].shape[0] != 5 * num:
                    print('before error')
                    print(j, data_before[j].shape[0])
                    print(data_before[j].reshape(-1, 5))
                    exit()
                if data_after[j].shape[0] != 5 * num:
                    print('after error')
                    print(j, data_after[j].shape[0])
                    exit()

            data_before = np.asarray(data_before)
            data_after = np.asarray(data_after)
            index_flag = np.asarray(index_flag)
            np.savetxt(before_path + 'num_%s.txt' % num, data_before)
            np.savetxt(after_path + 'num_%s.txt' % num, data_after)
            np.savetxt(index_flag_path + 'num_%s_flag.txt' % num, index_flag)
