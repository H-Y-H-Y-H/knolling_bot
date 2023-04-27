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

torch.manual_seed(42)
np.random.seed(202)
random.seed(202)


class Arm:

    def __init__(self, is_render=True):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = '../urdf/'
        self.pybullet_path = pd.getDataPath()
        self.is_render = is_render
        if self.is_render:
            p.connect(p.GUI, options="--width=1280 --height=720")
            # p.connect(p.GUI)
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

        self.correct = np.array([[0.016, 0.016, 0.012],
                                [0.024, 0.016, 0.012],
                                [0.032, 0.016, 0.012],
                                [0.01518, 0.09144, 0.01524]])

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

        if random.uniform(0,1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[random.randint(1,3), random.randint(1,2), 5])
        else:
            p.configureDebugVisualizer(lightPosition=[random.randint(1,3), random.randint(-2, -1), 5])
        p.configureDebugVisualizer(lightPosition=[random.randint(1, 3), random.randint(1, 2), 5],
                                   shadowMapResolution=8192, shadowMapIntensity=np.random.randint(5, 8) / 10)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())

    def get_parameters(self, lego_num,
                       total_offset=None, grasp_order=None,
                       gap_item=0.03, gap_block=0.02,
                       real_operate=False, obs_order='1',
                       random_offset = False, check_detection_loss=None, obs_img_from=None, use_yolo_pos=True):

        # self.lego_num = lego_num
        self.total_offset = total_offset
        self.grasp_order = grasp_order
        self.gap_item = gap_item
        self.gap_block = gap_block
        self.real_operate = real_operate
        self.obs_order = obs_order
        self.random_offset = random_offset
        self.num_list = lego_num
        self.check_detection_loss = check_detection_loss
        self.obs_img_from = obs_img_from
        self.use_yolo_pos = use_yolo_pos


    def get_obs(self, order, evaluation):

        def get_joints_obs():

            pass

        def get_images():

            (width, length, image, _, _) = p.getCameraImage(width=640,
                                                            height=480,
                                                            viewMatrix=self.view_matrix,
                                                            projectionMatrix=self.projection_matrix,
                                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
            return image

        def get_sim_image_obs():

            (width, length, image, _, _) = p.getCameraImage(width=640,
                                                            height=480,
                                                            viewMatrix=self.view_matrix,
                                                            projectionMatrix=self.projection_matrix,
                                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

            img = image[:, :,:3]
            img_path = 'Test_images/image_sim'
            cv2.imwrite(img_path + '.png', img)

            ############### order the ground truth depend on x, y in the world coordinate system ###############
            new_xyz_list = self.xyz_list

            # collect the cur pos and cur ori in pybullet as the ground truth
            new_pos_before, new_ori_before = [], []
            for i in range(len(self.obj_idx)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.obj_idx[i])[0][:2])
                new_ori_before.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1])[2])
            new_pos_before = np.asarray(new_pos_before)
            new_ori_before = np.asarray(new_ori_before)

            ground_truth_pose = data_preprocess(new_pos_before, new_xyz_list[:, :2], new_ori_before)
            # structure of ground_truth_pose: yolo-pose label!!!!!!!
            order_ground_truth = np.lexsort((ground_truth_pose[:, 2], ground_truth_pose[:, 1]))

            ground_truth_pose_test = np.copy(ground_truth_pose[order_ground_truth, :])
            for i in range(len(order_ground_truth) - 1):
                if np.abs(ground_truth_pose_test[i, 1] - ground_truth_pose_test[i+1, 1]) < 0.003:
                    if ground_truth_pose_test[i, 2] < ground_truth_pose_test[i+1, 2]:
                        # ground_truth_pose[order_ground_truth[i]], ground_truth_pose[order_ground_truth[i+1]] = ground_truth_pose[order_ground_truth[i+1]], ground_truth_pose[order_ground_truth[i]]
                        order_ground_truth[i], order_ground_truth[i+1] = order_ground_truth[i+1], order_ground_truth[i]
                        print('truth change the order!')
                    else:
                        pass
            print('this is the ground truth order', order_ground_truth)
            print('this is the ground truth before changing the order\n', ground_truth_pose)
            new_xyz_list = new_xyz_list[order_ground_truth, :]
            ground_truth_pose = ground_truth_pose[order_ground_truth, :]
            ############### order the ground truth depend on x, y in the world coordinate system ###############



            ################### the results of object detection has changed the order!!!! ####################
            # structure of results: x, y, length, width, ori
            results = yolov8_predict(img_path=img_path,
                                     real_flag=self.real_operate,
                                     target=ground_truth_pose)
            print('this is the result of yolo-pose\n', results)
            ################### the results of object detection has changed the order!!!! ####################


            # #################### check the error of resnet and yolo #####################
            # if self.check_detection_loss == True:
            #     if self.obs_img_from == 'env':
            #         env_loss = check_resnet_yolo(obs_img_from, results, target_compare)
            #     elif self.obs_img_from == 'dataset':
            #         dataset_loss = check_resnet_yolo(obs_img_from, results, target_compare)
            #     else:
            #         pass
            # #################### check the error of resnet and yolo #####################

            # arange the sequence based on categories of cubes
            z = 0
            roll = 0
            pitch = 0
            index = []
            correct = []
            for i in range(len(self.grasp_order)):
                correct.append(self.xyz_list[self.all_index[i][0]])
            correct = np.asarray(correct)
            for i in range(len(correct)):
                for j in range(len(results)):
                    if np.linalg.norm(correct[i][0] - results[j][2]) < 0.004:
                        index.append(j)
            manipulator_before = []
            for i in index:
                manipulator_before.append([results[i][0], results[i][1], z, roll, pitch, results[i][4]])
            manipulator_before = np.asarray(manipulator_before)
            new_xyz_list = self.xyz_list
            print('this is manipulator before after the detection \n', manipulator_before)

            if self.obs_order == 'sim_image_obj_evaluate':
                return manipulator_before, new_xyz_list, env_loss
            else:
                return manipulator_before, new_xyz_list

        def get_lego_obs():

            # sequence: pos before, ori before
            new_pos_before, new_ori_before = [], []
            for i in range(len(self.obj_idx)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
                new_ori_before.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_idx[i])[1]))

            manipulator_before = np.concatenate((new_pos_before, new_ori_before), axis=1)
            new_xyz_list = self.xyz_list

            return manipulator_before, new_xyz_list

        def get_real_image_obs():

            pipeline = rs.pipeline()
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))

            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                exit(0)

            # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            if device_product_line == 'L500':
                config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            else:
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
            # Start streaming
            pipeline.start(config)

            for _ in range(100):
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                # depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                color_colormap_dim = color_image.shape
                resized_color_image = color_image

                img_path = 'Test_images/image_real'
                cv2.imwrite(img_path + '.png', resized_color_image)
                cv2.waitKey(1)

            # structure: x,y,length,width,yaw
            results = yolov8_predict(img_path=img_path,
                                     real_flag=self.real_operate,
                                     target=None)
            print('this is the result of yolo-pose', results)

            # arange the sequence based on categories of cubes
            all_index = []
            new_xyz_list = []
            kind = []
            new_results = []
            ori_index = []
            z = 0
            roll = 0
            pitch = 0
            num = 0
            for i in range(len(self.correct)):
                kind_index = []
                for j in range(len(results)):
                    # if np.linalg.norm(self.correct[i][:2] - results[j][3:5]) < 0.003:
                    if np.linalg.norm(self.correct[i][0] - results[j][2]) < 0.004:
                        kind_index.append(num)
                        new_xyz_list.append(self.correct[i])
                        num += 1
                        if i in kind:
                            pass
                        else:
                            kind.append(i)
                        ori_index.append(j)
                        new_results.append(results[j])
                    else:
                        pass
                        print('detect failed!!!')
                if len(kind_index) != 0:
                    all_index.append(kind_index)
            new_xyz_list = np.asarray(new_xyz_list)
            ori_index = np.asarray(ori_index)
            new_results = np.asarray(new_results)
            print(new_results)
            print(all_index)

            manipulator_before = []
            for i in range(len(all_index)):
                for j in range(len(all_index[i])):
                    manipulator_before.append(
                        [new_results[all_index[i][j]][0], new_results[all_index[i][j]][1], z, roll, pitch,
                         new_results[all_index[i][j], 4]])
            manipulator_before = np.asarray(manipulator_before)

            print('this is the result of dectection before changing the sequence\n', results)
            print('this is manipulator before after the detection \n', manipulator_before)

            return manipulator_before, new_xyz_list

        if order == 'sim_obj':
            manipulator_before, new_xyz_list = get_lego_obs()
            return manipulator_before, new_xyz_list
        elif order == 'sim_image_obj':
            manipulator_before, new_xyz_list = get_sim_image_obs()
            return manipulator_before, new_xyz_list
        elif order == 'joint':
            get_joints_obs()
        elif order == 'images':
            image = get_images()
            return image
        elif order == 'real_image_obj':
            manipulator_before, new_xyz_list = get_real_image_obs()
            return manipulator_before, new_xyz_list
        elif order == 'sim_image_obj_evaluate':
            manipulator_before, new_xyz_list, error = get_sim_image_obs()
            return manipulator_before, new_xyz_list, error

    def reset(self):

        # p.resetSimulation()
        # p.setGravity(0, 0, -9.8)
        #
        # # Draw workspace lines
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
        #     lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
        #     lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
        #     lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
        #     lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        #
        # baseid = p.loadURDF(self.urdf_path + "plane_1.urdf", basePosition=[0, -0.2, 0], useFixedBase=1,
        #                     flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        # # self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm1.urdf",
        # #                          basePosition=[-0.08, 0, 0.02], useFixedBase=True,
        # #                          flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        #
        # textureId = p.loadTexture(self.urdf_path + "img_1.png")
        # p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5, angularDamping=0.5)
        # # p.changeDynamics(self.arm_id, 7, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        # # p.changeDynamics(self.arm_id, 8, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        # p.changeVisualShape(baseid, -1, textureUniqueId=textureId,rgbaColor=[np.random.uniform(0.9,1), np.random.uniform(0.9,1),np.random.uniform(0.9,1), 1])

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects()
        self.obj_idx = []
        if self.real_operate == False:
            self.xyz_list, _, _, self.all_index = items_sort.get_data_virtual(self.grasp_order, self.num_list)
            restrict = np.max(self.xyz_list)
            gripper_height = 0.012
            last_pos = np.array([[0, 0, 1]])

            ############## collect ori and pos to calculate the error of detection ##############
            collect_ori = []
            collect_pos = []
            ############## collect ori and pos to calculate the error of detection ##############

            for i in range(len(self.grasp_order)):
                for j in range(self.num_list[self.grasp_order[i]]):

                    rdm_pos = np.array([random.uniform(self.x_low_obs, self.x_high_obs),
                                        random.uniform(self.y_low_obs, self.y_high_obs), 0.0])
                    ori = [0, 0, random.uniform(0, math.pi)]
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
                    # self.obj_idx.append(
                    #     p.loadURDF(self.urdf_path + f"item_{self.grasp_order[i]}_lego/{j}.urdf",
                    #                basePosition=rdm_pos,
                    #                baseOrientation=p.getQuaternionFromEuler(ori), useFixedBase=False,
                    #                flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))

                    # r = np.random.uniform(0, 0.9)
                    # g = np.random.uniform(0, 0.9)
                    # b = np.random.uniform(0, 0.9)
                    # p.changeVisualShape(self.obj_idx[i], -1, rgbaColor=(r, g, b, 1))

            collect_ori = np.asarray(collect_ori)
            collect_pos = np.asarray(collect_pos)
            # check the error of the ResNet
            self.check_ori = collect_ori[:, 2]
            self.check_pos = collect_pos[:, :2]
            # check the error of the ResNet
            print('this is random ori when reset the environment', collect_ori)
            print('this is random pos when reset the environment', collect_pos)

        # print(self.obj_idx)
        # for i in range(len(self.obj_idx)):
        #     p.changeDynamics(self.obj_idx[i], -1, restitution=30)
        #     r = np.random.uniform(0, 0.9)
        #     g = np.random.uniform(0, 0.9)
        #     b = np.random.uniform(0, 0.9)
        #     p.changeVisualShape(self.obj_idx[i], -1, rgbaColor=(r, g, b, 1))

        # set the initial pos of the arm
        # ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=[0, 0, 0.06],
        #                                           maxNumIterations=200,
        #                                           targetOrientation=p.getQuaternionFromEuler([0, math.pi / 2, 0]))
        # for motor_index in range(self.num_motor):
        #     p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
        #                             targetPosition=ik_angles0[motor_index], maxVelocity=7)
        for i in range(60):
            p.stepSimulation()

        # ####################### try the corner pos and ori to calibrate the camera ####################
        # test_pos = []
        # for i in range(4):
        #     test_pos.append(p.getBasePositionAndOrientation(self.obj_idx[i])[0])
        # test_pos = np.asarray(test_pos)
        # print('this is test of cube pos\n', test_pos)
        # ####################### try the corner pos and ori to calibrate the camera ####################

        # return self.get_obs('images', None)
        return self.check_pos, self.check_ori, self.xyz_list[:, :2]

    def change_config(self):  # this is main function!!!!!!!!!

        # p.resetSimulation()
        # p.setGravity(0, 0, -10)
        #
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
        #     lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
        #     lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
        #     lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
        #     lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        #
        # baseid = p.loadURDF(self.urdf_path + "plane_1.urdf", basePosition=[0, -0.2, 0], useFixedBase=1,
        #                     flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        # # self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm1.urdf",
        # #                          basePosition=[-0.08, 0, 0.02], useFixedBase=True,
        # #                          flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        #
        # textureId = p.loadTexture(self.urdf_path + "img_1.png")
        # p.changeDynamics(baseid, -1, lateralFriction=self.lateral_friction, frictionAnchor=True)
        # # p.changeDynamics(self.arm_id, 7, lateralFriction=self.lateral_friction, frictionAnchor=True)
        # # p.changeDynamics(self.arm_id, 8, lateralFriction=self.lateral_friction, frictionAnchor=True)
        # p.changeVisualShape(baseid, -1, textureUniqueId=textureId,rgbaColor=[np.random.uniform(0.9,1), np.random.uniform(0.9,1),np.random.uniform(0.9,1), 1])

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects()
        if self.real_operate == False:
            self.xyz_list, _, _, self.all_index = items_sort.get_data_virtual(self.grasp_order, self.num_list)
        print(f'this is standard trim xyz list\n {self.xyz_list}')
        print(f'this is standard trim index list\n {self.all_index}')

        def calculate_items(item_num, item_xyz):

            min_xy = np.ones(2) * 100
            best_item_config = []

            if item_num % 2 == 0:
                fac = []  # 定义一个列表存放因子
                for i in range(1, item_num + 1):
                    if item_num % i == 0:
                        fac.append(i)
                        continue
                for i in range(len(fac)):
                    num_row = int(fac[i])
                    # print(num_row)
                    num_column = int(item_num / num_row)
                    # print(num_column)
                    len_x = num_row * item_xyz[0][0] + (num_row - 1) * self.gap_item
                    len_y = num_column * item_xyz[0][1] + (num_column - 1) * self.gap_item

                    if np.sum(np.array([len_x, len_y])) < np.sum(min_xy):
                        # print('for 2x2, this is the shorter distance')
                        min_xy = np.array([len_x, len_y])
                        best_item_config = [num_row, num_column]
            else:
                len_x1 = item_xyz[0][0] * item_num + (item_num - 1) * self.gap_item
                len_y1 = item_xyz[0][1]
                len_x2 = item_xyz[0][0]
                len_y2 = item_xyz[0][1] * item_num + (item_num - 1) * self.gap_item
                if np.sum(np.array([len_x1, len_y1])) < np.sum(np.array([len_x2, len_y2])):
                    min_xy = np.array([len_x1, len_y1])
                    best_item_config = [item_num, 1]
                else:
                    min_xy = np.array([len_x2, len_y2])
                    best_item_config = [1, item_num]
            best_item_config = np.asarray(best_item_config)

            return min_xy, best_item_config

        def calculate_block(): # first: calculate, second: reorder!

            min_result = []
            best_config = []
            for i in range(len(self.all_index)):
                item_index = self.all_index[i]
                item_xyz = self.xyz_list[item_index, :]
                item_num = len(item_index)
                xy, config = calculate_items(item_num, item_xyz)
                print(f'this is min xy {xy}')
                min_result.append(list(xy))
                print(f'this is the best item config {config}')
                best_config.append(list(config))
            min_result = np.asarray(min_result).reshape(-1, 2)
            min_xy = np.copy(min_result)
            best_config = np.asarray(best_config).reshape(-1, 2)
            print(best_config)

            # 安排总的摆放
            iteration = 500
            all_num = best_config.shape[0]
            all_x = 100
            all_y = 100

            if all_num % 2 == 0:
                fac = []  # 定义一个列表存放因子
                for i in range(1, all_num + 1):
                    if all_num % i == 0:
                        fac.append(i)
                        continue
            else:
                fac = [1, all_num]

            for i in range(iteration):

                sequence = np.random.choice(best_config.shape[0], size=len(self.all_index), replace=False)
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
            print(f'in iteration{i}, the min all_x and all_y are {all_x} {all_y}')
            print(all_x + all_y)

            return reorder_block(best_config, best_all_config, best_rotate_flag, best_min_xy)

        def reorder_item(best_config, start_pos, index_block, item_index, item_xyz, index_flag):

            # initiate the pos and ori
            # we don't analysis these imported oris
            # we directly define the ori is 0 or 90 degree, depending on the algorithm.
            item_pos = np.zeros([len(item_index), 3])
            item_ori = np.zeros([len(item_index), 3])
            # print(item_pos)
            num_2x2_row = best_config[index_block][0]
            num_2x2_column = best_config[index_block][1]
            index_2x2 = np.arange(item_pos.shape[0]).reshape(num_2x2_row, num_2x2_column)

            # the initial position of the first items

            if index_flag == True:

                temp = np.copy(item_xyz[:, 0])
                item_xyz[:, 0] = item_xyz[:, 1]
                item_xyz[:, 1] = temp
                item_ori[:, 2] = math.pi / 2
                # print(item_ori)
                temp = num_2x2_row
                num_2x2_row = num_2x2_column
                num_2x2_column = temp
                index_2x2 = index_2x2.transpose()
            else:

                item_ori[:, 2] = 0
                # print(item_ori)

            start_pos[0] = start_pos[0] + item_xyz[0][0] / 2
            start_pos[1] = start_pos[1] + item_xyz[0][1] / 2
            # print(f'this is try start {start_pos}')

            for j in range(num_2x2_row):
                for k in range(num_2x2_column):
                    x_2x2 = start_pos[0] + (item_xyz[index_2x2[j][k]][0]) * j + self.gap_item * j
                    y_2x2 = start_pos[1] + (item_xyz[index_2x2[j][k]][1]) * k + self.gap_item * k
                    item_pos[index_2x2[j][k]][0] = x_2x2
                    item_pos[index_2x2[j][k]][1] = y_2x2
            # print(item_pos)

            return item_ori, item_pos

        def reorder_block(best_config, best_all_config, best_rotate_flag, min_xy):

            print(f'the best configuration of all items is\n {best_all_config}')
            print(f'the best configuration of each kind of items is\n {best_config}')
            print(f'the rotate of each block of items is\n {best_rotate_flag}')
            print(f'this is the min_xy of each kind of items after rotation\n {min_xy}')

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
                    item_index = self.all_index[
                        best_all_config[m][n]]  # determine the index of blocks
                    # print('try', item_index)
                    item_xyz = self.xyz_list[item_index, :]
                    # print('try', item_xyz)
                    start_pos = np.asarray([start_x[m], start_y[n]])
                    index_block = best_all_config[m][n]
                    index_flag = best_rotate_flag[m][n]

                    ori, pos = reorder_item(best_config, start_pos, index_block, item_index, item_xyz,
                                                    index_flag)
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
        print(x_low, x_high, y_low, y_high)
        if self.random_offset == True:
            self.total_offset = np.array([random.uniform(self.x_low_obs + x_length / 2, self.x_high_obs - x_length / 2),
                                          random.uniform(self.y_low_obs + y_length / 2, self.y_high_obs - y_length / 2), 0.0])
        else:
            pass
        self.items_pos_list = self.items_pos_list + self.total_offset - center
        self.manipulator_after = np.concatenate((self.items_pos_list, self.items_ori_list), axis=1)
        print('this is manipulation after', self.manipulator_after)

        # # import urdf and assign the trim pos & ori
        # items_names = globals()
        # self.obj_idx = []
        # if self.real_operate == False:
        #     for i in range(len(self.grasp_order)):
        #         items_names[f'index_{self.grasp_order[i]}'] = self.all_index[i]
        #         items_names[f'num_{self.grasp_order[i]}'] = len(items_names[f'index_{self.grasp_order[i]}'])
        #         items_names[f'pos_{self.grasp_order[i]}'] = self.items_pos_list[items_names[f'index_{self.grasp_order[i]}'], :]
        #         items_names[f'ori_{self.grasp_order[i]}'] = self.items_ori_list[items_names[f'index_{self.grasp_order[i]}'], :]
        #         for j in range(self.num_list[self.grasp_order[i]]):
        #             self.obj_idx.append(p.loadURDF(self.urdf_path + f"item_{self.grasp_order[i]}_lego/{j}.urdf",
        #                                            basePosition=items_names[f'pos_{self.grasp_order[i]}'][j],
        #                                            baseOrientation=p.getQuaternionFromEuler(items_names[f'ori_{self.grasp_order[i]}'][j]),
        #                                            useFixedBase=True,
        #                                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
        #
        # for i in range(len(self.obj_idx)):
        #     p.changeDynamics(self.obj_idx[i], -1, restitution=30)
        #     r = np.random.uniform(0, 0.9)
        #     g = np.random.uniform(0, 0.9)
        #     b = np.random.uniform(0, 0.9)
        #     p.changeVisualShape(self.obj_idx[i], -1, rgbaColor=(r, g, b, 1))
        # return self.get_obs('images', None)
        return self.items_pos_list[:, :2], self.items_ori_list[:, 2], self.xyz_list[:, :2]


if __name__ == '__main__':

    command = 'knolling'

    if command == 'knolling':

        lego_num = np.array([2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        # item_0_lego: 16x16
        # item_1_lego: 20x16
        # item_2_lego: 20x20
        # item_3_lego: 24x16
        # item_4_lego: 24x20
        # item_5_lego: 24x24
        # item_6_lego: 28x16
        # item_7_lego: 28x20
        # item_8_lego: 28x24
        # item_9_lego: 32x16
        # item_10_lego: 32x20
        # item_11_lego: 32x24
        total_offset = [0.15, 0.1, 0]
        grasp_order = np.arange(len(lego_num))
        index = np.where(lego_num == 0)
        grasp_order = np.delete(grasp_order, index)
        gap_item = 0.01
        gap_block = 0.02
        random_offset = False
        real_operate = False
        obs_order = 'sim_image_obj'
        check_detection_loss = False
        obs_img_from = 'env'
        use_yolo_pos = False


        env = Arm(is_render=True)
        env.get_parameters(lego_num=lego_num,
                           total_offset=total_offset, grasp_order=grasp_order,
                           gap_item=gap_item, gap_block=gap_block,
                           real_operate=real_operate, obs_order=obs_order,
                           random_offset=random_offset, check_detection_loss=check_detection_loss,
                           obs_img_from=obs_img_from, use_yolo_pos=use_yolo_pos)

        target_path = '/home/zhizhuo/ADDdisk/Create Machine Lab/knolling_dataset/learning_data/'
        before_path = target_path + 'labels_reset/'
        after_path = target_path + 'labels_knolling/'
        os.makedirs(before_path, exist_ok=True)
        os.makedirs(after_path, exist_ok=True)

        evaluations = 1
        data_before = []
        data_after = []
        for i in range(evaluations):
            pos_before, ori_before, xy_before = env.reset()
            data_before.append(np.concatenate((pos_before, xy_before, ori_before.reshape(-1, 1)), axis=1).reshape(-1,))
            pos_after, ori_after, xy_after = env.change_config()
            data_after.append(np.concatenate((pos_after, xy_after, ori_after.reshape(-1, 1)), axis=1).reshape(-1))
        np.savetxt(before_path + '1.txt', data_before)
        np.savetxt(after_path + '1.txt', data_after)
            # image_chaotic = env.reset()
            # print(image_chaotic.shape)
            # temp = np.copy(image_chaotic[:, :, 0])
            # image_chaotic[:, :, 0] = np.copy(image_chaotic[:, :, 2])
            # image_chaotic[:, :, 2] = temp
            # # print(f'this is {image_chaotic}')
            # image_trim = env.change_config()
            # temp = np.copy(image_trim[:, :, 0])
            # image_trim[:, :, 0] = np.copy(image_trim[:, :, 2])
            # image_trim[:, :, 2] = temp
            # # print(image_trim.shape)
            #
            # new_img = np.concatenate((image_chaotic, image_trim), axis=1)
            # # print(new_img)
            #
            # cv2.line(new_img, (int(new_img.shape[1] / 2), 0), (int(new_img.shape[1] / 2), new_img.shape[0]), (0, 0, 0),
            #          20)
            # cv2.imshow("Comparison between Chaotic Configuration and Trim Configuration", new_img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()