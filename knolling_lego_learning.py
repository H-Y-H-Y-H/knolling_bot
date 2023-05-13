#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'     :::::     .:::::::''::::.
#            .:::'       :::::  .:::::::::'  ':::::.
#           .::'        :::::.:::::::::'      '::::::.
#          .::'         ::::::::::::::'         ``:::::
#      ...:::           ::::::::::::'              `::::.
#     ```` ':.          ':::::::::'                  ::::::..
#                        '.:::::'                    ':'``````:.
#                     美女保佑 永无BUG

import numpy as np
import pyrealsense2 as rs
from items_real_learning import Sort_objects
import pybullet_data as pd
import math
from turdf import *
import socket
import cv2
from cam_obs_learning import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Polygon
from knolling_configuration import configuration_zzz

torch.manual_seed(42)
np.random.seed(202)
random.seed(202)


class Arm:

    def __init__(self, is_render=True):

        self.kImageSize = {'width': 480, 'height': 480}
        self.urdf_path = './urdf/'
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

    def get_parameters(self, lego_num=None, area_num=None, ratio_num=None, boxes_index=None,
                       total_offset=None, evaluations=None,
                       gap_item=0.03, gap_block=0.02,
                       real_operate=False, obs_order='1',
                       random_offset = False, check_detection_loss=None, obs_img_from=None, use_lego_urdf=True,
                       item_odd_prevent=None, block_odd_prevent=None, upper_left_max = None, forced_rotate_box=None):

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
        self.use_lego_urdf = use_lego_urdf
        self.boxes_index = boxes_index
        self.evaluations = evaluations
        self.configuration = None
        self.item_odd_prevent = item_odd_prevent
        self.block_odd_prevent = block_odd_prevent
        self.upper_left_max = upper_left_max
        self.forced_rotate_box = forced_rotate_box

        self.correct = np.array([[0.016, 0.016, 0.012],
                                 [0.020, 0.016, 0.012],
                                 [0.020, 0.020, 0.012],
                                 [0.024, 0.016, 0.012],
                                 [0.024, 0.020, 0.012],
                                 [0.024, 0.024, 0.012],
                                 [0.028, 0.016, 0.012],
                                 [0.028, 0.020, 0.012],
                                 [0.028, 0.024, 0.012],
                                 [0.032, 0.016, 0.012],
                                 [0.032, 0.020, 0.012],
                                 [0.032, 0.024, 0.012]])

    def get_obs(self, order):

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
            # cv2.imwrite(img_path + '.png', img)

            ############### order the ground truth depend on x, y in the world coordinate system ###############
            new_xyz_list = self.xyz_list

            # collect the cur pos and cur ori in pybullet as the ground truth
            new_pos_before, new_ori_before = [], []
            for i in range(len(self.lego_idx)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.lego_idx[i])[0][:2])
                new_ori_before.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.lego_idx[i])[1])[2])
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
            # print('this is the ground truth before changing the order\n', ground_truth_pose)
            new_xyz_list = new_xyz_list[order_ground_truth, :]
            ground_truth_pose = ground_truth_pose[order_ground_truth, :]
            ############### order the ground truth depend on x, y in the world coordinate system ###############



            ################### the results of object detection has changed the order!!!! ####################
            # structure of results: x, y, length, width, ori
            if self.real_operate == False:
                results, env_loss = yolov8_predict(img_path=img_path, img=img,
                                                             real_flag=self.real_operate,
                                                             target=ground_truth_pose)
            else:
                results = yolov8_predict(img_path=img_path, img=img,
                                          real_flag=self.real_operate,
                                          target=ground_truth_pose)
            print('this is the result of yolo-pose\n', results)
            ################### the results of object detection has changed the order!!!! ####################

            z = 0
            roll = 0
            pitch = 0
            index = []
            # width_index = []
            print('this is self.xyz', self.xyz_list)
            for i in range(len(self.xyz_list)):
                for j in range(len(results)):
                    if (np.abs(self.xyz_list[i, 0] - results[j, 2]) < 0.002 and np.abs(self.xyz_list[i, 1] - results[j, 3]) < 0.002) or \
                            (np.abs(self.xyz_list[i, 1] - results[j, 2]) < 0.002 and np.abs(self.xyz_list[i, 0] - results[j, 3]) < 0.002):
                        if j not in index:
                            index.append(j)
                        # if i not in width_index:
                        #     width_index.append(i)
                        else:
                            pass


            # # arange the sequence based on categories of cubes
            # index = []
            # correct = []
            # for i in range(len(self.grasp_order)):
            #     correct.append(self.xyz_list[self.all_index[i][0]])
            # correct = np.asarray(correct)
            # for i in range(len(correct)):
            #     for j in range(len(results)):
            #         if np.linalg.norm(correct[i][0] - results[j][2]) < 0.004:
            #             index.append(j)

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
            for i in range(len(self.lego_idx)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.lego_idx[i])[0])
                new_ori_before.append(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.lego_idx[i])[1]))

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
                # cv2.imwrite(img_path + '.png', resized_color_image)
                # cv2.waitKey(1)

            # structure: x,y,length,width,yaw
            results = yolov8_predict(img_path=img_path, img=resized_color_image,
                                     real_flag=self.real_operate,
                                     target=None)
            print('this is the result of yolo-pose', results)

            z = 0
            roll = 0
            pitch = 0
            index = []
            print('this is self.xyz', self.xyz_list)
            for i in range(len(self.xyz_list)):
                for j in range(len(results)):
                    if (np.abs(self.xyz_list[i, 0] - results[j, 2]) < 0.0015 and np.abs(
                            self.xyz_list[i, 1] - results[j, 3]) < 0.0015) or \
                            (np.abs(self.xyz_list[i, 1] - results[j, 2]) < 0.0015 and np.abs(
                                self.xyz_list[i, 0] - results[j, 3]) < 0.0015):
                        if j not in index:
                            index.append(j)
                        else:
                            pass

            manipulator_before = []
            for i in index:
                manipulator_before.append([results[i][0], results[i][1], z, roll, pitch, results[i][4]])
            manipulator_before = np.asarray(manipulator_before)
            new_xyz_list = self.xyz_list
            print('this is manipulator before after the detection \n', manipulator_before)

            # # arange the sequence based on categories of cubes
            # all_index = []
            # new_xyz_list = []
            # kind = []
            # new_results = []
            # ori_index = []
            # z = 0
            # roll = 0
            # pitch = 0
            # num = 0
            # for i in range(len(self.correct)):
            #     kind_index = []
            #     for j in range(len(results)):
            #         # if np.linalg.norm(self.correct[i][:2] - results[j][3:5]) < 0.003:
            #         if np.linalg.norm(self.correct[i][0] - results[j][2]) < 0.004:
            #             kind_index.append(num)
            #             new_xyz_list.append(self.correct[i])
            #             num += 1
            #             if i in kind:
            #                 pass
            #             else:
            #                 kind.append(i)
            #             ori_index.append(j)
            #             new_results.append(results[j])
            #         else:
            #             pass
            #             print('detect failed!!!')
            #     if len(kind_index) != 0:
            #         all_index.append(kind_index)
            # new_xyz_list = np.asarray(new_xyz_list)
            # ori_index = np.asarray(ori_index)
            # new_results = np.asarray(new_results)
            # print(new_results)
            # print(all_index)
            #
            # manipulator_before = []
            # for i in range(len(all_index)):
            #     for j in range(len(all_index[i])):
            #         manipulator_before.append(
            #             [new_results[all_index[i][j]][0], new_results[all_index[i][j]][1], z, roll, pitch,
            #              new_results[all_index[i][j], 4]])
            # manipulator_before = np.asarray(manipulator_before)
            #
            # print('this is the result of dectection before changing the sequence\n', results)
            # print('this is manipulator before after the detection \n', manipulator_before)

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

        baseid = p.loadURDF(self.urdf_path + "plane_1.urdf", basePosition=[0, -0.2, 0], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        self.arm_id = p.loadURDF(os.path.join(self.urdf_path, "robot_arm928/robot_arm1.urdf"),
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(self.urdf_path + "img_1.png")
        p.changeDynamics(baseid, -1, lateralFriction=1, spinningFriction=1, rollingFriction=0.002, linearDamping=0.5, angularDamping=0.5)
        p.changeDynamics(self.arm_id, 7, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.arm_id, 8, lateralFriction=1, spinningFriction=1, rollingFriction=0, linearDamping=0, angularDamping=0)
        # p.changeVisualShape(baseid, -1, textureUniqueId=textureId,rgbaColor=[np.random.uniform(0.9,1), np.random.uniform(0.9,1),np.random.uniform(0.9,1), 1])
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId)

        # set the initial pos of the arm
        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=[0, 0, 0.06],
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler([0, math.pi / 2, 0]))
        for motor_index in range(self.num_motor):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=7)
        for i in range(100):
            p.stepSimulation()

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects(self.evaluations)
        self.lego_idx = []
        if self.real_operate == False:
            self.xyz_list, _, _, self.all_index, self.transform_flag, self.urdf_index, self.lego_urdf_index = items_sort.get_data_virtual(self.area_num, self.ratio_num, self.num_list, self.boxes_index, self.use_lego_urdf)
            restrict = np.max(self.xyz_list)
            gripper_height = 0.012
            last_pos = np.array([[0, 0, 1]])

            ############## collect ori and pos to calculate the error of detection ##############
            collect_ori = []
            collect_pos = []
            ############## collect ori and pos to calculate the error of detection ##############

            if self.use_lego_urdf == True:
                # for i in range(len(self.xyz_list)):
                #     for j in range(len(self.num_list)):
                #         if np.abs(self.xyz_list[i][0] - self.correct[j][0]) < 0.001 and np.abs(self.xyz_list[i][1] - self.correct[j][1]) < 0.001:
                #             pass
                for i in range(len(self.lego_urdf_index)):
                    for j in range(len(self.lego_urdf_index[i])):
                # for i in range(length):
                #     for j in range(self.num_list[lego_index[i]]):
                        rdm_pos = np.array([random.uniform(self.x_low_obs, self.x_high_obs),
                                            random.uniform(self.y_low_obs, self.y_high_obs), 0])
                        ori = [0, 0, random.uniform(0, math.pi)]
                        # ori = [0, 0, np.pi / 2]
                        # ori = [0, 0, 0]
                        collect_ori.append(ori)
                        check_list = np.zeros(last_pos.shape[0])

                        while 0 in check_list:
                            rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs),
                                       random.uniform(self.y_low_obs, self.y_high_obs), 0]
                            for z in range(last_pos.shape[0]):
                                if np.linalg.norm(last_pos[z] - rdm_pos) < restrict + gripper_height:
                                    check_list[z] = 0
                                else:
                                    check_list[z] = 1
                        collect_pos.append(rdm_pos)
                        last_pos = np.append(last_pos, [rdm_pos], axis=0)

                        self.lego_idx.append(p.loadURDF(self.urdf_path + "item_%d_lego/%d.urdf" % (self.lego_urdf_index[i][j], j),
                                           basePosition=rdm_pos,
                                           baseOrientation=p.getQuaternionFromEuler(ori), useFixedBase=False,
                                           flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                for m in range(len(self.lego_idx)):
                    r = np.random.uniform(0, 0.9)
                    g = np.random.uniform(0, 0.9)
                    b = np.random.uniform(0, 0.9)
                    p.changeVisualShape(self.lego_idx[m], -1, rgbaColor=(r, g, b, 1))
            else:
                length = len(self.urdf_index)

                for i in range(length):
                    rdm_pos = np.array([random.uniform(self.x_low_obs, self.x_high_obs),
                                        random.uniform(self.y_low_obs, self.y_high_obs), 0.006])
                    ori = [0, 0, random.uniform(0, math.pi)]
                    # ori = [0, 0, np.pi / 2]

                    ################### after generate the neat configuration, we should recover the lw based on that in the urdf files!
                    if self.transform_flag[i] == 1:
                        self.xyz_list[i, [0, 1]] = self.xyz_list[i, [1, 0]]
                        # if ori[2] > np.pi:
                        #     ori[2] -= np.pi / 2
                        # else:
                        #     ori[2] += np.pi / 2
                        # we dont' need to change the ori here, because the ori is definitely random
                        # the real ori provided to arm is genereatd by yolo
                    ################### after generate the neat configuration, we should recover the lw based on that in the urdf files!
                    # ori = [0, 0, 0]
                    collect_ori.append(ori)
                    check_list = np.zeros(last_pos.shape[0])

                    while 0 in check_list:
                        rdm_pos = [random.uniform(self.x_low_obs, self.x_high_obs),
                                   random.uniform(self.y_low_obs, self.y_high_obs), 0.006]
                        for z in range(last_pos.shape[0]):
                            if np.linalg.norm(last_pos[z] - rdm_pos) < restrict + gripper_height:
                                check_list[z] = 0
                            else:
                                check_list[z] = 1
                    collect_pos.append(rdm_pos)
                    last_pos = np.append(last_pos, [rdm_pos], axis=0)
                    self.lego_idx.append(
                        p.loadURDF(self.urdf_path + f"box_generator/box_{self.urdf_index[i]}.urdf",
                                   basePosition=rdm_pos,
                                   baseOrientation=p.getQuaternionFromEuler(ori), useFixedBase=False,
                                   flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                    r = np.random.uniform(0, 0.9)
                    g = np.random.uniform(0, 0.9)
                    b = np.random.uniform(0, 0.9)
                    p.changeVisualShape(self.lego_idx[i], -1, rgbaColor=(r, g, b, 1))

            collect_ori = np.asarray(collect_ori)
            collect_pos = np.asarray(collect_pos)
            # check the error of the ResNet
            self.check_ori = collect_ori[:, 2]
            self.check_pos = collect_pos[:, :2]
            # check the error of the ResNet
            print('this is random ori when reset the environment', collect_ori)
            print('this is random pos when reset the environment', collect_pos)
        else:
            # these data has defined in function change_config, we don't need to define them twice!!!
            # self.xyz_list, pos_before, ori_before, self.all_index, self.kind = items_sort.get_data_real()
            num_lego = 0
            sim_pos = np.copy(self.pos_before)
            sim_pos[2] += 0.006
            for i in range(len(self.xyz_list)):
                self.lego_idx.append(
                    p.loadURDF(self.urdf_path + f"knolling_box/knolling_box_{self.urdf_index[i]}.urdf",
                               basePosition=sim_pos[i],
                               baseOrientation=p.getQuaternionFromEuler(self.ori_before[i]), useFixedBase=False,
                               flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
                r = np.random.uniform(0, 0.9)
                g = np.random.uniform(0, 0.9)
                b = np.random.uniform(0, 0.9)
                p.changeVisualShape(self.lego_idx[num_lego], -1, rgbaColor=(r, g, b, 1))
                num_lego += 1

        data_before = np.concatenate((self.pos_before[:, :2], self.xyz_list[:, :2], self.ori_before[:, 2].reshape(-1, 1)), axis=1)
        np.savetxt('./learning_data_demo/cfg_0/labels_before/label_%d.txt' % self.evaluations, data_before, fmt='%.03f')

        return self.get_obs('images')
        # return self.pos_before, self.ori_before, self.xyz_list

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

        baseid = p.loadURDF(self.urdf_path + "plane_1.urdf", basePosition=[0, -0.2, 0], useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm1.urdf",
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        textureId = p.loadTexture(self.urdf_path + "img_1.png")
        p.changeDynamics(baseid, -1, lateralFriction=self.lateral_friction, frictionAnchor=True)
        p.changeDynamics(self.arm_id, 7, lateralFriction=self.lateral_friction, frictionAnchor=True)
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.lateral_friction, frictionAnchor=True)
        p.changeVisualShape(baseid, -1, textureUniqueId=textureId)

        # get the standard xyz and corresponding index from files in the computer
        items_sort = Sort_objects(self.evaluations)
        if self.real_operate == False:
            self.xyz_list, _, _, self.all_index, self.transform_flag, self.urdf_index, self.lego_urdf_index = items_sort.get_data_virtual(self.area_num, self.ratio_num, self.num_list, self.boxes_index, self.use_lego_urdf)
        else:
            self.xyz_list, self.pos_before, self.ori_before, self.all_index, self.transform_flag, self.urdf_index = items_sort.get_data_real(self.area_num, self.ratio_num, self.num_list)
        print(f'this is standard trim xyz list\n {self.xyz_list}')
        print(f'this is standard trim index list\n {self.all_index}')


        calculate_reorder = configuration_zzz(self.xyz_list, self.all_index, self.gap_item, self.gap_block,
                                              self.transform_flag, self.configuration,
                                              self.item_odd_prevent, self.block_odd_prevent, self.upper_left_max,
                                              self.forced_rotate_box)

        # determine the center of the tidy configuration
        self.items_pos_list, self.items_ori_list = calculate_reorder.calculate_block()
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
        # self.items_pos_list = self.items_pos_list + self.total_offset - center
        self.items_pos_list = self.items_pos_list + self.total_offset
        items_ori_list_arm = np.copy(self.items_ori_list)
        ################### after generate the neat configuration, we should recover the lw based on that in the urdf files!
        for i in range(len(self.transform_flag)):
            if self.transform_flag[i] == 1:
                print(f'ori changed in index{i}!')
                if self.items_ori_list[i, 2] > np.pi:
                    items_ori_list_arm[i, 2] = self.items_ori_list[i, 2] - np.pi / 2
                else:
                    items_ori_list_arm[i, 2] = self.items_ori_list[i, 2] + np.pi / 2
        ################### after generate the neat configuration, we should recover the lw based on that in the urdf files!
        self.manipulator_after = np.concatenate((self.items_pos_list, items_ori_list_arm), axis=1)
        print('this is manipulation after', self.manipulator_after)


        self.lego_idx = []
        for i in range(len(self.urdf_index)):
            if self.real_operate == False:
                if self.use_lego_urdf == True:
                    pass
                else:
                    self.lego_idx.append(p.loadURDF(self.urdf_path + f"box_generator/box_{self.urdf_index[i]}.urdf",
                                       basePosition=self.items_pos_list[i],
                                       baseOrientation=p.getQuaternionFromEuler(self.items_ori_list[i]), useFixedBase=False,
                                       flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))
            else:
                sim_pos = np.copy(self.items_pos_list)
                sim_pos[:, 2] += 0.006
                self.lego_idx.append(p.loadURDF(self.urdf_path + f"knolling_box/knolling_box_{self.urdf_index[i]}.urdf",
                                   basePosition=sim_pos[i],
                                   baseOrientation=p.getQuaternionFromEuler(self.items_ori_list[i]), useFixedBase=False,
                                   flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT))

            r = np.random.uniform(0, 0.9)
            g = np.random.uniform(0, 0.9)
            b = np.random.uniform(0, 0.9)
            p.changeVisualShape(self.lego_idx[i], -1, rgbaColor=(r, g, b, 1))

        # if the urdf is lego, all ori after knolling should be 0, not pi / 2
        self.items_ori_list[:, 2] = 0
        data_after = np.concatenate((self.items_pos_list[:, :2], self.xyz_list[:, :2], self.items_ori_list[:, 2].reshape(-1, 1)), axis=1)
        np.savetxt('./learning_data_demo/cfg_0/labels_after/label_%d.txt' % self.evaluations, data_after, fmt='%.03f')

        return self.get_obs('images')

    def planning(self, order, conn, real_height, sim_height):

        def get_start_end():  # generating all trajectories of all items in normal condition
            arm_z = 0
            roll = 0
            pitch = 0
            if self.obs_order == 'sim_image_obj_evaluate':
                manipulator_before, new_xyz_list, error = self.get_obs(self.obs_order)
                return error
            else:
                manipulator_before, new_xyz_list = self.get_obs(self.obs_order)

            # sequence pos_before, ori_before, pos_after, ori_after
            start_end = []
            for i in range(len(self.lego_idx)):
                start_end.append([manipulator_before[i][0], manipulator_before[i][1], arm_z, roll, pitch, manipulator_before[i][5],
                                  self.manipulator_after[i][0], self.manipulator_after[i][1], arm_z, roll, pitch, self.manipulator_after[i][5]])
            start_end = np.asarray((start_end))
            print('get start and end')

            return start_end, new_xyz_list

        def move(cur_pos, cur_ori, tar_pos, tar_ori):

            # add the offset manually
            if self.real_operate == True:
                # # automatically add z and x bias
                d = np.array([0, 0.3])
                d_y = np.array((0, 0.17, 0.21, 0.30))
                d_y = d
                z_bias = np.array([-0.005, 0.004])
                x_bias = np.array([-0.002, 0.00])# yolo error is +2mm along x axis!
                y_bias = np.array([0, -0.004, -0.001, 0.004])
                y_bias = np.array([0.002, 0.006])
                # z_parameters = np.polyfit(d, z_bias, 3)
                z_parameters = np.polyfit(d, z_bias, 1)
                x_parameters = np.polyfit(d, x_bias, 1)
                y_parameters = np.polyfit(d_y, y_bias, 1)
                new_z_formula = np.poly1d(z_parameters)
                new_x_formula = np.poly1d(x_parameters)
                new_y_formula = np.poly1d(y_parameters)

                distance = tar_pos[0]
                distance_y = tar_pos[0]
                tar_pos[2] = tar_pos[2] + new_z_formula(distance)
                print('this is z add', new_z_formula(distance))
                tar_pos[0] = tar_pos[0] + new_x_formula(distance)
                print('this is x add', new_x_formula(distance))
                if tar_pos[1] > 0:
                    tar_pos[1] += new_y_formula(distance_y) * np.clip((6 * (tar_pos[1] + 0.01)), 0, 1)
                else:
                    tar_pos[1] -= new_y_formula(distance_y) * np.clip((6 * (tar_pos[1] - 0.01)), 0, 1)
                print('this is tar pos after manual', tar_pos)

            if tar_ori[2] > 3.1416 / 2:
                tar_ori[2] = tar_ori[2] - np.pi
                print('tar ori is too large')
            elif tar_ori[2] < -3.1416 / 2:
                tar_ori[2] = tar_ori[2] + np.pi
                print('tar ori is too small')
            # print('this is tar ori', tar_ori)

            #################### use feedback control ###################
            if abs(cur_pos[0] - tar_pos[0]) < 0.001 and abs(cur_pos[1] - tar_pos[1]) < 0.001:
                # vertical, choose a small slice
                move_slice = 0.004
            else:
                # horizontal, choose a large slice
                if self.real_operate == True:
                    move_slice = 0.008
                else:
                    move_slice = 0.004

            # ###### zzz set time sleep ######
            # if cur_pos[2] - tar_pos[2] > 0.02:
            #     print(cur_pos)
            #     print(tar_pos)
            #     print('this is time sleep')
            #     time.sleep(1)

            if self.real_operate == True:
                tar_pos = tar_pos + np.array([0, 0, real_height])
                target_pos = np.copy(tar_pos)
                target_ori = np.copy(tar_ori)
                # target_pos[2] = Cartesian_offset_nn(np.array([tar_pos])).reshape(-1, )[2] # remove nn offset temporary

                if np.abs(target_pos[2] - cur_pos[2]) > 0.01 \
                        and np.abs(target_pos[0] - cur_pos[0]) < 0.01 \
                        and np.abs(target_pos[1] - cur_pos[1]) < 0.01:
                    print('we dont need feedback control')
                    mark_ratio = 0.8
                    seg_time = 0
                else:
                    mark_ratio = 0.99
                    seg_time = 0

                while True:
                    plot_cmd = []
                    # plot_real = []
                    sim_xyz = []
                    sim_ori = []
                    real_xyz = []

                    # divide the whole trajectory into several segment
                    seg_time += 1
                    seg_pos = mark_ratio * (target_pos - cur_pos) + cur_pos
                    seg_ori = mark_ratio * (target_ori - cur_ori) + cur_ori
                    distance = np.linalg.norm(seg_pos - cur_pos)
                    num_step = np.ceil(distance / move_slice)
                    step_pos = (seg_pos - cur_pos) / num_step
                    step_ori = (seg_ori - cur_ori) / num_step

                    while True:
                        tar_pos = cur_pos + step_pos
                        tar_ori = cur_ori + step_ori
                        sim_xyz.append(tar_pos)
                        sim_ori.append(tar_ori)
                        # print(tar_ori)

                        ik_angles_sim = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos,
                                                                     maxNumIterations=200,
                                                                     targetOrientation=p.getQuaternionFromEuler(
                                                                         tar_ori))

                        for motor_index in range(self.num_motor):
                            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                    targetPosition=ik_angles_sim[motor_index], maxVelocity=25)
                        for i in range(30):
                            p.stepSimulation()

                        angle_sim = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles_sim[0:5])), dtype=np.float32)
                        plot_cmd.append(angle_sim)

                        break_flag = abs(seg_pos[0] - tar_pos[0]) < 0.001 and abs(
                            seg_pos[1] - tar_pos[1]) < 0.001 and abs(seg_pos[2] - tar_pos[2]) < 0.001 and \
                                     abs(seg_ori[0] - tar_ori[0]) < 0.001 and abs(
                            seg_ori[1] - tar_ori[1]) < 0.001 and abs(seg_ori[2] - tar_ori[2]) < 0.001
                        if break_flag:
                            break

                        # update cur_pos and cur_ori in several step of each segment
                        cur_pos = tar_pos
                        cur_ori = tar_ori

                    sim_xyz = np.asarray(sim_xyz)

                    plot_step = np.arange(num_step)
                    plot_cmd = np.asarray(plot_cmd)
                    # print('this is the shape of cmd', plot_cmd.shape)
                    # print('this is the shape of xyz', sim_xyz.shape)
                    # print('this is the motor pos sent', plot_cmd[-1])
                    conn.sendall(plot_cmd.tobytes())
                    # print('waiting the manipulator')
                    angles_real = conn.recv(8192)
                    # print('received')
                    angles_real = np.frombuffer(angles_real, dtype=np.float32)
                    angles_real = angles_real.reshape(-1, 6)

                    if seg_time > 0:
                        seg_flag = False
                        print('segment fail, try to tune!')
                        ik_angles_sim = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=target_pos,
                                                                     maxNumIterations=200,
                                                                     targetOrientation=p.getQuaternionFromEuler(
                                                                         target_ori))

                        for motor_index in range(self.num_motor):
                            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                    targetPosition=ik_angles_sim[motor_index], maxVelocity=7.5)
                        for i in range(30):
                            p.stepSimulation()

                        angle_sim = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles_sim[0:5])), dtype=np.float32)
                        final_cmd = np.append(angle_sim, 0).reshape(1, -1)
                        final_cmd = np.asarray(final_cmd, dtype=np.float32)
                        conn.sendall(final_cmd.tobytes())

                        # get the pos after tune!
                        final_angles_real = conn.recv(4096)
                        # print('received')
                        final_angles_real = np.frombuffer(final_angles_real, dtype=np.float32).reshape(-1, 6)
                        print('this is final after moving', final_angles_real)

                        ik_angles_real = np.asarray(cmd2rad(real_tarpos2cmd(final_angles_real)), dtype=np.float32)
                        for motor_index in range(self.num_motor):
                            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                    targetPosition=ik_angles_real[motor_index], maxVelocity=25)
                        for i in range(30):
                            p.stepSimulation()
                        real_xyz = np.append(real_xyz, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)
                        cur_pos = real_xyz[-1]
                        # print(real_xyz)
                        break
                    else:
                        print('this is the shape of angles real', angles_real.shape)
                        for i in range(len(angles_real)):
                            ik_angles_real = np.asarray(cmd2rad(real_tarpos2cmd(angles_real[i])), dtype=np.float32)
                            for motor_index in range(self.num_motor):
                                p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                        targetPosition=ik_angles_real[motor_index], maxVelocity=25)
                            for i in range(30):
                                p.stepSimulation()
                            real_xyz = np.append(real_xyz, np.asarray(p.getLinkState(self.arm_id, 9)[0])).reshape(-1, 3)
                        cur_pos = real_xyz[-1]
                        break

            else:
                tar_pos = tar_pos + np.array([0, 0, sim_height])
                target_pos = np.copy(tar_pos)
                target_ori = np.copy(tar_ori)

                distance = np.linalg.norm(tar_pos - cur_pos)
                num_step = np.ceil(distance / move_slice)
                step_pos = (target_pos - cur_pos) / num_step
                step_ori = (target_ori - cur_ori) / num_step

                print('this is sim tar pos', tar_pos)
                while True:
                    tar_pos = cur_pos + step_pos
                    tar_ori = cur_ori + step_ori
                    ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=tar_pos,
                                                              maxNumIterations=200,
                                                              targetOrientation=p.getQuaternionFromEuler(tar_ori))
                    for motor_index in range(self.num_motor):
                        p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                                targetPosition=ik_angles0[motor_index], maxVelocity=25)
                    for i in range(30):
                        p.stepSimulation()
                        time.sleep(1 / 960)
                    if abs(target_pos[0] - tar_pos[0]) < 0.001 and abs(target_pos[1] - tar_pos[1]) < 0.001 and abs(
                            target_pos[2] - tar_pos[2]) < 0.001 and \
                            abs(target_ori[0] - tar_ori[0]) < 0.001 and abs(target_ori[1] - tar_ori[1]) < 0.001 and abs(
                        target_ori[2] - tar_ori[2]) < 0.001:
                        break
                    cur_pos = tar_pos
                    cur_ori = tar_ori

            return cur_pos

        def gripper(gap, obj_width):
            obj_width += 0.006
            obj_width_range = np.array([0.021, 0.026, 0.032, 0.039, 0.045, 0.052, 0.057])
            motor_pos_range = np.array([2000, 2100, 2200, 2300, 2400, 2500, 2600])
            formula_parameters = np.polyfit(obj_width_range, motor_pos_range, 3)
            motor_pos = np.poly1d(formula_parameters)

            if self.real_operate == True:
                if gap > 0.0265:  # close
                    pos_real = np.asarray([[gap, 1600]], dtype=np.float32)
                elif gap <= 0.0265:  # open
                    pos_real = np.asarray([[gap, motor_pos(obj_width)]], dtype=np.float32)
                print('gripper', pos_real)
                conn.sendall(pos_real.tobytes())
                # print(f'this is the cmd pos {pos_real}')
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gap, force=10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gap, force=10)

                real_pos = conn.recv(4096)
                # test_real_pos = np.frombuffer(real_pos, dtype=np.float32)
                real_pos = np.frombuffer(real_pos, dtype=np.float32)
                # print('this is test float from buffer', test_real_pos)

            else:
                p.setJointMotorControl2(self.arm_id, 7, p.POSITION_CONTROL, targetPosition=gap, force=10)
                p.setJointMotorControl2(self.arm_id, 8, p.POSITION_CONTROL, targetPosition=gap, force=10)
            for i in range(30):
                p.stepSimulation()
                time.sleep(1 / 120)

        def clean_desk():

            if self.real_operate == False:
                gripper_width = 0.024
                gripper_height = 0.034
            else:
                gripper_width = 0.018
                gripper_height = 0.04
            restrict_gripper_diagonal = np.sqrt(gripper_width ** 2 + gripper_height ** 2)
            barricade_pos = []
            barricade_index = []
            manipulator_before, new_xyz_list = self.get_obs(self.obs_order)
            print('this is test obs xyz', new_xyz_list)
            for i in range(len(manipulator_before)):
                for j in range(len(self.manipulator_after)):
                    restrict_item_i = np.sqrt((new_xyz_list[i][0]) ** 2 + (new_xyz_list[i][1]) ** 2)
                    restrict_item_j = np.sqrt((new_xyz_list[j][0]) ** 2 + (new_xyz_list[j][1]) ** 2)
                    if np.linalg.norm(self.manipulator_after[j][:3] - manipulator_before[i][:3]) < restrict_item_i / 2 + restrict_item_j / 2:
                        if i not in barricade_index:
                            print('We need to sweep the desktop to provide an enough space')
                            barricade_pos.append(manipulator_before[i][:3])
                            barricade_index.append(i)
            barricade_pos = np.asarray(barricade_pos)

            x_high = np.max(self.manipulator_after[:, 0])
            x_low = np.min(self.manipulator_after[:, 0])
            y_high = np.max(self.manipulator_after[:, 1])
            y_low = np.min(self.manipulator_after[:, 1])
            p.addUserDebugLine(lineFromXYZ=[x_low, y_low, 0], lineToXYZ=[x_high, y_low, 0])
            p.addUserDebugLine(lineFromXYZ=[x_low, y_low, 0], lineToXYZ=[x_low, y_high, 0])
            p.addUserDebugLine(lineFromXYZ=[x_high, y_high, 0], lineToXYZ=[x_high, y_low, 0])
            p.addUserDebugLine(lineFromXYZ=[x_high, y_high, 0], lineToXYZ=[x_low, y_high, 0])

            while len(barricade_index) > 0:

                # pos
                offset_low = np.array([0, 0, 0.005])
                offset_high = np.array([0, 0, 0.035])
                # ori
                rest_ori = np.array([0, 1.57, 0])
                # axis and direction
                if y_high - y_low > x_high - x_low:
                    offset_rectangle = np.array([0, 0, math.pi / 2])
                    axis = 'x_axis'
                    if (x_high + x_low) / 2 > (self.x_high_obs + self.x_low_obs) / 2:
                        direction = 'negative'
                        offset_horizontal = np.array([np.max(self.xyz_list) - 0.001, 0, 0])
                    else:
                        direction = 'positive'
                        offset_horizontal = np.array([-(np.max(self.xyz_list) - 0.001), 0, 0])
                else:
                    offset_rectangle = np.array([0, 0, 0])
                    axis = 'y_axis'
                    if (y_high + y_low) / 2 > (self.y_high_obs + self.y_low_obs) / 2:
                        direction = 'negative'
                        offset_horizontal = np.array([0, np.max(self.xyz_list) - 0.001, 0])
                    else:
                        direction = 'positive'
                        offset_horizontal = np.array([0, -(np.max(self.xyz_list) - 0.001), 0])

                trajectory_pos_list = []
                trajectory_ori_list = []
                print(barricade_index)
                for i in range(len(barricade_index)):
                    diag = np.sqrt((new_xyz_list[barricade_index[i]][0]) ** 2 + (new_xyz_list[barricade_index[i]][1]) ** 2)
                    if axis == 'x_axis':
                        if direction == 'positive':
                            print('x,p')
                            offset_horizontal = np.array([-(diag / 2 + gripper_height / 2), 0, 0])
                            terminate = np.array([x_high, barricade_pos[i][1], barricade_pos[i][2]])
                        elif direction == 'negative':
                            print('x,n')
                            offset_horizontal = np.array([diag / 2 + gripper_height / 2, 0, 0])
                            terminate = np.array([x_low, barricade_pos[i][1], barricade_pos[i][2]])
                    elif axis == 'y_axis':
                        if direction == 'positive':
                            print('y,p')
                            offset_horizontal = np.array([0, -(diag / 2 + gripper_height / 2), 0])
                            terminate = np.array([barricade_pos[i][0], y_high, barricade_pos[i][2]])
                        elif direction == 'negative':
                            print('y,n')
                            offset_horizontal = np.array([0, diag / 2 + gripper_height / 2, 0])
                            terminate = np.array([barricade_pos[i][0], y_low, barricade_pos[i][2]])

                    trajectory_pos_list.append([0.03159, 0])
                    trajectory_pos_list.append(barricade_pos[i] + offset_high + offset_horizontal)
                    trajectory_pos_list.append(barricade_pos[i] + offset_low + offset_horizontal)
                    trajectory_pos_list.append(offset_low - offset_horizontal + terminate)
                    trajectory_pos_list.append(offset_high - offset_horizontal + terminate)

                    trajectory_ori_list.append(rest_ori + offset_rectangle)
                    trajectory_ori_list.append(rest_ori + offset_rectangle)
                    trajectory_ori_list.append(rest_ori + offset_rectangle)
                    trajectory_ori_list.append(rest_ori + offset_rectangle)
                    trajectory_ori_list.append(rest_ori + offset_rectangle)

                # reset the manipulator to read the image
                trajectory_pos_list.append([0, 0, 0.08])
                trajectory_ori_list.append([0, math.pi / 2, 0])

                last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                for j in range(len(trajectory_pos_list)):

                    if len(trajectory_pos_list[j]) == 3:
                        last_pos = move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])
                    elif len(trajectory_pos_list[j]) == 1:
                        gripper(trajectory_pos_list[j][0])

                break_flag = False
                barricade_pos = []
                barricade_index = []
                manipulator_before, new_xyz_list = self.get_obs(self.obs_order)
                for i in range(len(manipulator_before)):
                    for j in range(len(self.manipulator_after)):
                        # 这里会因为漏检的bug而报错！
                        restrict_item_i = np.sqrt((new_xyz_list[i][0]) ** 2 + (new_xyz_list[i][1]) ** 2)
                        restrict_item_j = np.sqrt((new_xyz_list[j][0]) ** 2 + (new_xyz_list[j][1]) ** 2)
                        if np.linalg.norm(self.manipulator_after[j][:3] - manipulator_before[i][:3]) < restrict_item_i / 2 + restrict_item_j / 2:
                            if i not in barricade_index:
                                print('We still need to sweep the desktop to provide an enough space')
                                barricade_pos.append(manipulator_before[i][:3])
                                barricade_index.append(i)
                                break_flag = True
                                break
                    if break_flag == True:
                        break
                barricade_pos = np.asarray(barricade_pos)
            else:
                print('nothing to sweep')
                pass
            print('Sweeping desktop end')

            return manipulator_before, new_xyz_list

        def clean_item(manipulator_before, new_xyz_list):

            if self.real_operate == False:
                gripper_width = 0.024
                gripper_height = 0.034
            else:
                gripper_width = 0.018
                gripper_height = 0.04

            restrict_gripper_diagonal = np.sqrt(gripper_width ** 2 + gripper_height ** 2)
            gripper_lego_gap = 0.006
            crowded_pos = []
            crowded_ori = []
            crowded_index = []

            # these two variables have been defined in clean_desk function, we don't need to define them twice!!!!!
            # manipulator_before, new_xyz_list = self.get_obs(self.obs_order, _)
            # print(manipulator_before)

            # define the cube which is crowded
            for i in range(len(manipulator_before)):
                for j in range(len(manipulator_before)):
                    restrict_item_i = np.sqrt((new_xyz_list[i][0]) ** 2 + (new_xyz_list[i][1]) ** 2)
                    restrict_item_j = np.sqrt((new_xyz_list[j][0]) ** 2 + (new_xyz_list[j][1]) ** 2)
                    if 0.0001 < np.linalg.norm(manipulator_before[j][:3] - manipulator_before[i][:3]) < restrict_item_i / 2 + restrict_item_j / 2 + 0.001:
                        if i not in crowded_index and j not in crowded_index:
                            print('We need to separate the items surrounding it to provide an enough space')
                            crowded_pos.append(manipulator_before[i][:3])
                            crowded_ori.append(manipulator_before[i][3:6])
                            crowded_pos.append(manipulator_before[j][:3])
                            crowded_ori.append(manipulator_before[j][3:6])
                            crowded_index.append(i)
                            crowded_index.append(j)
                        if i in crowded_index and j not in crowded_index:
                            print('We need to separate the items surrounding it to provide an enough space')
                            crowded_pos.append(manipulator_before[j][:3])
                            crowded_ori.append(manipulator_before[j][3:6])
                            crowded_index.append(j)
            crowded_pos = np.asarray(crowded_pos)

            while len(crowded_index) > 0:
                # pos
                offset_low = np.array([0, 0, 0.005])
                offset_high = np.array([0, 0, 0.035])
                # ori
                rest_ori = np.array([0, 1.57, 0])

                trajectory_pos_list = []
                trajectory_ori_list = []
                for i in range(len(crowded_index)):
                    break_flag = False
                    once_flag = False

                    length_lego = new_xyz_list[crowded_index[i]][0]
                    width_lego = new_xyz_list[crowded_index[i]][1]
                    theta = manipulator_before[crowded_index[i]][5]

                    matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                       [np.sin(theta), np.cos(theta)]])
                    target_point = np.array([[(length_lego + gripper_height + gripper_lego_gap) / 2, (width_lego + gripper_width + gripper_lego_gap) / 2],
                                            [-(length_lego + gripper_height + gripper_lego_gap) / 2, (width_lego + gripper_width + gripper_lego_gap) / 2],
                                            [-(length_lego + gripper_height + gripper_lego_gap) / 2, -(width_lego + gripper_width + gripper_lego_gap) / 2],
                                            [(length_lego + gripper_height + gripper_lego_gap) / 2, -(width_lego + gripper_width + gripper_lego_gap) / 2]])
                    target_point_rotate = (matrix.dot(target_point.T)).T
                    print('this is target point rotate\n', target_point_rotate)
                    sequence_point = np.concatenate((target_point_rotate, np.zeros((4, 1))), axis=1)

                    t = 0
                    for j in range(len(sequence_point)):
                        vertex_break_flag = False
                        for k in range(len(manipulator_before)):
                            # exclude itself
                            if np.linalg.norm(crowded_pos[i] - manipulator_before[k][:3]) < 0.001:
                                continue
                            restrict_item_k = np.sqrt((new_xyz_list[k][0]) ** 2 + (new_xyz_list[k][1]) ** 2)
                            if 0.001 < np.linalg.norm(sequence_point[0] + crowded_pos[i] - manipulator_before[k][:3]) < restrict_item_k/2 + restrict_gripper_diagonal/2 + 0.001:
                                print(np.linalg.norm(sequence_point[0] + crowded_pos[i] - manipulator_before[k][:3]))
                                p.addUserDebugPoints([sequence_point[0] + crowded_pos[i]], [[0.1, 0, 0]], pointSize=5)
                                p.addUserDebugPoints([manipulator_before[k][:3]], [[0, 1, 0]], pointSize=5)
                                print("this vertex doesn't work")
                                vertex_break_flag = True
                                break
                        if vertex_break_flag == False:
                            print("this vertex is ok")
                            print(break_flag)
                            once_flag = True
                            break
                        else:
                            # should change the vertex and try again
                            sequence_point = np.roll(sequence_point, -1, axis=0)
                            print(sequence_point)
                            t += 1
                        if t == len(sequence_point):
                            # all vertex of this cube fail, should change the cube
                            break_flag = True

                    # problem, change another crowded cube

                    if break_flag == True:
                        if i == len(crowded_index) - 1:
                            print('cannot find any proper vertices to insert, we should unpack the heap!!!')
                            x_high = np.max(self.manipulator_after[:, 0])
                            x_low = np.min(self.manipulator_after[:, 0])
                            y_high = np.max(self.manipulator_after[:, 1])
                            y_low = np.min(self.manipulator_after[:, 1])
                            crowded_x_high = np.max(crowded_pos[:, 0])
                            crowded_x_low = np.min(crowded_pos[:, 0])
                            crowded_y_high = np.max(crowded_pos[:, 1])
                            crowded_y_low = np.min(crowded_pos[:, 1])

                            trajectory_pos_list.append([0.03159, 0])
                            trajectory_pos_list.append([(x_high + x_low) / 2, (y_high + y_low) / 2, offset_high[2]])
                            trajectory_pos_list.append([(x_high + x_low) / 2, (y_high + y_low) / 2, offset_low[2]])
                            trajectory_pos_list.append([(crowded_x_high + crowded_x_low) / 2, (crowded_y_high + crowded_y_low) / 2, offset_low[2]])
                            trajectory_pos_list.append([(crowded_x_high + crowded_x_low) / 2, (crowded_y_high + crowded_y_low) / 2, offset_high[2]])
                            trajectory_pos_list.append([0, 0, 0.08])

                            trajectory_ori_list.append(rest_ori)
                            trajectory_ori_list.append(rest_ori)
                            trajectory_ori_list.append(rest_ori)
                            trajectory_ori_list.append(rest_ori)
                            trajectory_ori_list.append(rest_ori)
                            trajectory_ori_list.append(rest_ori)
                        else:
                            pass
                    else:
                        trajectory_pos_list.append([0.03159, 0])
                        print('this is crowded pos', crowded_pos[i])
                        print('this is sequence point', sequence_point)
                        trajectory_pos_list.append(crowded_pos[i] + offset_high + sequence_point[0])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[0])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[1])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[2])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[3])
                        trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[0])
                        trajectory_pos_list.append(crowded_pos[i] + offset_high + sequence_point[0])
                        # reset the manipulator to read the image
                        trajectory_pos_list.append([0, 0, 0.06])

                        trajectory_ori_list.append(rest_ori)
                        trajectory_ori_list.append(rest_ori + crowded_ori[i])
                        trajectory_ori_list.append(rest_ori + crowded_ori[i])
                        trajectory_ori_list.append(rest_ori + crowded_ori[i])
                        trajectory_ori_list.append(rest_ori + crowded_ori[i])
                        trajectory_ori_list.append(rest_ori + crowded_ori[i])
                        trajectory_ori_list.append(rest_ori + crowded_ori[i])
                        trajectory_ori_list.append(rest_ori + crowded_ori[i])
                        # reset the manipulator to read the image
                        trajectory_ori_list.append([0, math.pi / 2, 0])

                    # only once!
                    if once_flag == True:
                        break
                last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                trajectory_pos_list = np.asarray(trajectory_pos_list)
                trajectory_ori_list = np.asarray(trajectory_ori_list)

                ######################### add the debug lines for visualization ####################
                line_id = []
                four_points = trajectory_pos_list[2:6]
                line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[0], lineToXYZ=four_points[1]))
                line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[1], lineToXYZ=four_points[2]))
                line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[2], lineToXYZ=four_points[3]))
                line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[3], lineToXYZ=four_points[0]))
                ######################### add the debug line for visualization ####################

                for j in range(len(trajectory_pos_list)):
                    if len(trajectory_pos_list[j]) == 3:
                        last_pos = move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])
                    elif len(trajectory_pos_list[j]) == 1:
                        gripper(trajectory_pos_list[j][0])

                ######################### remove the debug lines after moving ######################
                for i in line_id:
                    p.removeUserDebugItem(i)
                ######################### remove the debug lines after moving ######################

                # check the environment again and update the pos and ori of cubes
                crowded_pos = []
                crowded_ori = []
                crowded_index = []
                manipulator_before, new_xyz_list = self.get_obs(self.obs_order)
                for i in range(len(manipulator_before)):
                    for j in range(len(manipulator_before)):
                        restrict_item_i = np.sqrt((new_xyz_list[i][0]) ** 2 + (new_xyz_list[i][1]) ** 2)
                        restrict_item_j = np.sqrt((new_xyz_list[j][0]) ** 2 + (new_xyz_list[j][1]) ** 2)
                        if 0.0001 < np.linalg.norm(manipulator_before[j][:3] - manipulator_before[i][:3]) < restrict_item_i / 2 + restrict_item_j / 2:
                            if i not in crowded_index and j not in crowded_index:
                                print('We need to separate the items surrounding it to provide an enough space')
                                crowded_pos.append(manipulator_before[i][:3])
                                crowded_ori.append(manipulator_before[i][3:6])
                                crowded_pos.append(manipulator_before[j][:3])
                                crowded_ori.append(manipulator_before[j][3:6])
                                crowded_index.append(i)
                                crowded_index.append(j)
                            if i in crowded_index and j not in crowded_index:
                                print('We need to separate the items surrounding it to provide an enough space')
                                crowded_pos.append(manipulator_before[j][:3])
                                crowded_ori.append(manipulator_before[j][3:6])
                                crowded_index.append(j)
                crowded_pos = np.asarray(crowded_pos)
            else:
                print('nothing around the item')
                pass
            print('separating end')

        def knolling():

            if self.obs_order == 'sim_image_obj_evaluate':
                env_loss = get_start_end()
                return env_loss
            else:
                start_end, new_xyz_list_knolling = get_start_end()


            rest_pos = np.array([0, 0, 0.05])
            rest_ori = np.array([0, 1.57, 0])
            offset_low = np.array([0, 0, 0.002])
            offset_high = np.array([0, 0, 0.035])
            offset_highest = np.array([0, 0, 0.05])

            grasp_width = np.min(new_xyz_list_knolling[:, :2], axis=1)

            for i in range(len(self.lego_idx)):

                trajectory_pos_list = [[0.01, grasp_width[i]], # open!
                                       offset_high + start_end[i][:3],
                                       offset_low + start_end[i][:3],
                                       [0.0273, grasp_width[i]], # close
                                       offset_high + start_end[i][:3],
                                       offset_high + start_end[i][6:9],
                                       offset_low + start_end[i][6:9],
                                       [0.01, grasp_width[i]],
                                       offset_high + start_end[i][6:9]]

                trajectory_ori_list = [rest_ori + start_end[i][3:6],
                                       rest_ori + start_end[i][3:6],
                                       rest_ori + start_end[i][3:6],
                                       [0.0273, grasp_width[i]],
                                       rest_ori + start_end[i][3:6],
                                       rest_ori + start_end[i][9:12],
                                       rest_ori + start_end[i][9:12],
                                       [0.01, grasp_width[i]],
                                       rest_ori + start_end[i][9:12]]
                if i == 0:
                    last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                    last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                else:
                    pass

                for j in range(len(trajectory_pos_list)):

                    if len(trajectory_pos_list[j]) == 3:
                        print('ready to move', trajectory_pos_list[j])
                        # print('ready to move cur ori', last_ori)
                        last_pos = move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                        last_ori = np.copy(trajectory_ori_list[j])
                        # print('this is last ori after moving', last_ori)

                    elif len(trajectory_pos_list[j]) == 1:
                        gripper(trajectory_pos_list[j][0])

            # back to the reset pos and ori
            last_pos = move(last_pos, last_ori, rest_pos, rest_ori)
            last_ori = np.copy(rest_ori)
            ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=rest_pos,
                                                      maxNumIterations=200,
                                                      targetOrientation=p.getQuaternionFromEuler(rest_ori))
            for motor_index in range(self.num_motor):
                p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                        targetPosition=ik_angles0[motor_index], maxVelocity=7)
            for i in range(30):
                p.stepSimulation()
                # self.images = self.get_image()
                # time.sleep(1 / 48)

        def check_accuracy_sim(): # need improvement


            manipulator_before, new_xyz_list = self.get_obs(self.obs_order) # the sequence along 2,3,4
            manipulator_knolling = manipulator_before[:, :2]
            xyz_knolling = new_xyz_list
            # don't change the order of xyz in sim!!!!!!!!!!!!!

            order_knolling = np.lexsort((manipulator_before[:, 1], manipulator_before[:, 0]))
            manipulator_knolling_test = np.copy(manipulator_knolling[order_knolling, :])
            for i in range(len(order_knolling) - 1):
                if np.abs(manipulator_knolling_test[i, 0] - manipulator_knolling_test[i + 1, 0]) < 0.003:
                    if manipulator_knolling_test[i, 1] < manipulator_knolling_test[i + 1, 1]:
                        order_knolling[i], order_knolling[i + 1] = order_knolling[i + 1], \
                                                                           order_knolling[i]
                        print('knolling change the order!')
                    else:
                        pass
            print('this is the ground truth order', order_knolling)
            # print('this is the ground truth before changing the order\n', manipulator_knolling)
            manipulator_knolling = manipulator_knolling[order_knolling, :]


            new_pos_before, new_ori_before = [], []
            for i in range(len(self.lego_idx)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.lego_idx[i])[0][:2])
            new_pos_before = np.asarray(new_pos_before)
            manipulator_ground_truth = new_pos_before
            xyz_ground_truth = self.xyz_list
            # don't change the order of xyz in sim!!!!!!!!!!!!!

            order_ground_truth = np.lexsort((manipulator_ground_truth[:, 1], manipulator_ground_truth[:, 0]))
            manipulator_ground_truth_test = np.copy(manipulator_ground_truth[order_ground_truth, :])
            for i in range(len(order_ground_truth) - 1):
                if np.abs(manipulator_ground_truth_test[i, 0] - manipulator_ground_truth_test[i + 1, 0]) < 0.003:
                    if manipulator_ground_truth_test[i, 1] < manipulator_ground_truth_test[i + 1, 1]:
                        order_ground_truth[i], order_ground_truth[i + 1] = order_ground_truth[i + 1], \
                                                                   order_ground_truth[i]
                        print('truth change the order!')
                    else:
                        pass
            print('this is the ground truth order', order_knolling)
            # print('this is the ground truth before changing the order\n', manipulator_knolling)
            manipulator_ground_truth = manipulator_ground_truth[order_ground_truth, :]

            print('this is manipulator ground truth while checking \n', manipulator_ground_truth)
            print('this is manipulator after knolling while checking \n', manipulator_knolling)
            print('this is xyz ground truth while checking \n', xyz_ground_truth)
            print('this is xyz after knolling while checking \n', xyz_knolling)
            for i in range(len(manipulator_ground_truth)):
                if np.linalg.norm(manipulator_ground_truth[i] - manipulator_knolling[i]) < 0.005 and \
                    np.linalg.norm(xyz_ground_truth[i] - xyz_knolling[i]) < 0.005:
                    print('find it!')
                else:
                    print('error!')

        def check_accuracy_real():
            manipulator_before, new_xyz_list = self.get_obs(self.obs_order)
            manipulator_knolling = manipulator_before[:, :2]
            xyz_knolling = new_xyz_list
            # don't change the order of xyz in sim!!!!!!!!!!!!!

            order_knolling = np.lexsort((manipulator_before[:, 1], manipulator_before[:, 0]))
            manipulator_knolling_test = np.copy(manipulator_knolling[order_knolling, :])
            for i in range(len(order_knolling) - 1):
                if np.abs(manipulator_knolling_test[i, 0] - manipulator_knolling_test[i + 1, 0]) < 0.003:
                    if manipulator_knolling_test[i, 1] < manipulator_knolling_test[i + 1, 1]:
                        order_knolling[i], order_knolling[i + 1] = order_knolling[i + 1], \
                                                                   order_knolling[i]
                        print('knolling change the order!')
                    else:
                        pass
            print('this is the ground truth order', order_knolling)
            # print('this is the ground truth before changing the order\n', manipulator_knolling)
            manipulator_knolling = manipulator_knolling[order_knolling, :]

            new_pos_before, new_ori_before = [], []
            for i in range(len(self.lego_idx)):
                new_pos_before.append(p.getBasePositionAndOrientation(self.lego_idx[i])[0][:2])
            new_pos_before = np.asarray(new_pos_before)
            manipulator_ground_truth = new_pos_before
            xyz_ground_truth = self.xyz_list
            # don't change the order of xyz in sim!!!!!!!!!!!!!

            order_ground_truth = np.lexsort((manipulator_ground_truth[:, 1], manipulator_ground_truth[:, 0]))
            manipulator_ground_truth_test = np.copy(manipulator_ground_truth[order_ground_truth, :])
            for i in range(len(order_ground_truth) - 1):
                if np.abs(manipulator_ground_truth_test[i, 0] - manipulator_ground_truth_test[i + 1, 0]) < 0.003:
                    if manipulator_ground_truth_test[i, 1] < manipulator_ground_truth_test[i + 1, 1]:
                        order_ground_truth[i], order_ground_truth[i + 1] = order_ground_truth[i + 1], \
                                                                           order_ground_truth[i]
                        print('truth change the order!')
                    else:
                        pass
            print('this is the ground truth order', order_knolling)
            # print('this is the ground truth before changing the order\n', manipulator_knolling)
            manipulator_ground_truth = manipulator_ground_truth[order_ground_truth, :]

            print('this is manipulator ground truth while checking \n', manipulator_ground_truth)
            print('this is manipulator after knolling while checking \n', manipulator_knolling)
            print('this is xyz ground truth while checking \n', xyz_ground_truth)
            print('this is xyz after knolling while checking \n', xyz_knolling)
            for i in range(len(manipulator_ground_truth)):
                if np.linalg.norm(manipulator_ground_truth[i] - manipulator_knolling[i]) < 0.005 and \
                        np.linalg.norm(xyz_ground_truth[i] - xyz_knolling[i]) < 0.005:
                    print('find it!')
                else:
                    print('error!')

            print('this is all distance between messy and neat in real world')

        if order == 1:
            manipulator_before_desk, new_xyz_list_desk = clean_desk()
            clean_item(manipulator_before_desk, new_xyz_list_desk)
        elif order == 3:
            if self.obs_order == 'sim_image_obj_evaluate':
                error = knolling()
                return error
            else:
                knolling()
        elif order == 4:
            if self.real_operate == True:
                check_accuracy_real()
            else:
                check_accuracy_sim()
        elif order == 5:
            error = get_start_end()
            return error

    def step(self):

        if self.real_operate == True:

            with open(file="Cartisian_data/cmd.txt", mode="w", encoding="utf-8") as f:
                f.truncate(0)
            with open(file="Cartisian_data/real.txt", mode="w", encoding="utf-8") as f:
                f.truncate(0)
            with open(file="Cartisian_data/step.txt", mode="w", encoding="utf-8") as f:
                f.truncate(0)

            HOST = "192.168.0.186"  # Standard loopback interface address (localhost)
            PORT = 8880  # Port to listen on (non-privileged ports are > 1023)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST, PORT))
            # It should be an integer from 1 to 65535, as 0 is reserved. Some systems may require superuser privileges if the port number is less than 8192.
            # associate the socket with a specific network interface
            s.listen()
            print(f"Waiting for connection...\n")
            conn, addr = s.accept()
            print(conn)
            print(f"Connected by {addr}")
            table_surface_height = 0.032
            sim_table_surface_height = -0.01
            num_motor = 5
            # ! reset the pos in both real and sim
            reset_pos = [0.005, 0, 0.1]
            ik_angles = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=reset_pos,
                                                     maxNumIterations=300,
                                                     targetOrientation=p.getQuaternionFromEuler([0, 1.57, 0]))
            reset_real = np.asarray(real_cmd2tarpos(rad2cmd(ik_angles[0:5])), dtype=np.float32)
            conn.sendall(reset_real.tobytes())

            for i in range(num_motor):
                p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=ik_angles[i],
                                        maxVelocity=3)
            for _ in range(30):
                p.stepSimulation()
                # time.sleep(1 / 24)
        else:
            conn = None
            table_surface_height = 0.032
            sim_table_surface_height = -0.01

        #######################################################################################
        # 1: clean_desk + clean_item, 3: knolling, 4: check_accuracy of knolling, 5: get_camera
        # self.planning(1, conn, table_surface_height, sim_table_surface_height, evaluation)
        # error = self.planning(5, conn, table_surface_height, sim_table_surface_height, evaluation)
        error = self.planning(3, conn, table_surface_height, sim_table_surface_height)
        # self.planning(4, conn, table_surface_height, sim_table_surface_height, evaluation)
        #######################################################################################

        if self.obs_order == 'sim_image_obj_evaluate':
            return error

        if self.real_operate == True:
            end = np.array([0], dtype=np.float32)
            conn.sendall(end.tobytes())
        print(f'evaluation 0 over!!!!!')

if __name__ == '__main__':

    command = 'knolling'

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)
    # model = Net().to(device)
    # model.load_state_dict(torch.load("Test_and_Calibration/nn_calibration/model_pt_xyz/combine_all_free_001005_far_free_001005_flank_free_001005_useful.pt"))
    # # print(model)
    # model.eval()

    # Visualize two images (before and after knolling)
    if command == 'image':
        num_2x2 = 2
        num_2x3 = 4
        num_2x4 = 3
        num_pencil = 0
        # total_offset = [0.1, -0.1, 0]
        total_offset = [0.016, -0.17 + 0.016, 0]
        gap_item = 0.015
        gap_block = 0.02
        random_offset = True
        real_operate = False
        obs_order = 'images'

        lego_num = np.array([2, 2, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0])
        grasp_order = np.arange(len(lego_num))
        index = np.where(lego_num == 0)
        grasp_order = np.delete(grasp_order, index)

        env = Arm(is_render=True, urdf_path='./urdf/')
        env.get_parameters(lego_num=lego_num,
                           total_offset=total_offset, grasp_order=grasp_order,
                           gap_item=gap_item, gap_block=gap_block,
                           real_operate=real_operate, obs_order=obs_order, random_offset=random_offset)
        # env.get_parameters(num_2x2=num_2x2, num_2x3=num_2x3, num_2x4=num_2x4,
        #                    total_offset=total_offset, grasp_order=grasp_order,
        #                    gap_item=gap_item, gap_block=gap_block, from_virtual=from_virtual)

        image_chaotic = env.reset()
        temp = np.copy(image_chaotic[:, :, 0])
        image_chaotic[:, :, 0] = np.copy(image_chaotic[:, :, 2])
        image_chaotic[:, :, 2] = temp
        # print(f'this is {image_chaotic}')
        image_trim = env.change_config()
        temp = np.copy(image_trim[:, :, 0])
        image_trim[:, :, 0] = np.copy(image_trim[:, :, 2])
        image_trim[:, :, 2] = temp
        # print(image_trim.shape)

        new_img = np.concatenate((image_chaotic, image_trim), axis=1)
        # print(new_img)

        cv2.line(new_img, (int(new_img.shape[1] / 2), 0), (int(new_img.shape[1] / 2), new_img.shape[0]), (0, 0, 0), 20)
        cv2.imshow("Comparison between Chaotic Configuration and Trim Configuration", new_img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if command == 'knolling':

        lego_num = 8
        area_num = 3
        ratio_num = 1
        boxes_index = np.random.choice(30, lego_num)
        # total_offset = [0.15, 0.1, 0]
        total_offset = [0.016, -0.17 + 0.016, 0]
        gap_item = 0.015
        gap_block = 0.015
        random_offset = False
        real_operate = True
        obs_order = 'real_image_obj'
        check_detection_loss = False
        obs_img_from = 'env'
        use_lego_urdf = True

        item_odd_prevent = True
        block_odd_prevent = True
        upper_left_max = True
        forced_rotate_box = False

        lego_num = np.array([2, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0])
        grasp_order = np.arange(len(lego_num))
        index = np.where(lego_num == 0)
        grasp_order = np.delete(grasp_order, index)

        env = Arm(is_render=True)
        evaluations = 0
        env.get_parameters(lego_num=lego_num, area_num=area_num, ratio_num=ratio_num, boxes_index=boxes_index,
                           total_offset=total_offset, evaluations=evaluations,
                           gap_item=gap_item, gap_block=gap_block,
                           real_operate=real_operate, obs_order=obs_order,
                           random_offset=random_offset, check_detection_loss=check_detection_loss,
                           obs_img_from=obs_img_from, use_lego_urdf=use_lego_urdf,
                           item_odd_prevent=item_odd_prevent, block_odd_prevent = block_odd_prevent,
                           upper_left_max = upper_left_max, forced_rotate_box=forced_rotate_box)


        # for i in range(evaluations):
        image_trim = env.change_config()
        _ = env.reset()
        env.step()

    if command == 'evaluate_object_detection':

        evaluations = 15
        error_min = 100
        evaluation_min = 0
        error_list = []
        env = Arm(is_render=True)
        for i in range(evaluations):
            # num_2x2 = np.random.randint(1, 6)
            # num_2x3 = np.random.randint(1, 6)
            # num_2x4 = np.random.randint(1, 6)
            # total_offset = [0.15, 0, 0]
            # grasp_order = [1, 0, 2]
            # gap_item = 0.015
            # gap_block = 0.02
            # random_offset = True
            # real_operate = False
            # obs_order = 'sim_image_obj_evaluate'
            # check_detection_loss = False
            # obs_img_from = 'env'
            #
            # lego_num = np.array([2, 2, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0])
            #
            # env = Arm(is_render=True)
            # env.get_parameters(lego_num=lego_num,
            #                    total_offset=total_offset, grasp_order=grasp_order,
            #                    gap_item=gap_item, gap_block=gap_block,
            #                    real_operate=real_operate, obs_order=obs_order,
            #                    random_offset=random_offset, check_detection_loss=check_detection_loss,
            #                    obs_img_from=obs_img_from)
            # image_trim = env.change_config()

            lego_num = np.random.randint(4, 10)
            area_num = 4
            ratio_num = 1
            boxes_index = np.random.choice(30, lego_num)
            # total_offset = [0.15, 0.1, 0]
            total_offset = [0.016, -0.17 + 0.016, 0]
            gap_item = 0.015
            gap_block = 0.02
            random_offset = False
            real_operate = False
            obs_order = 'sim_image_obj_evaluate'
            check_detection_loss = False
            obs_img_from = 'env'
            use_yolo_pos = False

            lego_num = np.array([2, 2, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0])
            grasp_order = np.arange(len(lego_num))
            index = np.where(lego_num == 0)
            grasp_order = np.delete(grasp_order, index)

            env.get_parameters(lego_num=lego_num, area_num=area_num, ratio_num=ratio_num, boxes_index=boxes_index,
                               total_offset=total_offset,
                               gap_item=gap_item, gap_block=gap_block,
                               real_operate=real_operate, obs_order=obs_order,
                               random_offset=random_offset, check_detection_loss=check_detection_loss,
                               obs_img_from=obs_img_from, use_yolo_pos=use_yolo_pos)

            _ = env.reset()
            error = env.step(i)
            error_list.append(error)
            if error_min > error:
                print(error)
                evaluation_min = i
                # print(evaluation_max)
                error_min = error
        error_list = np.asarray(error_list)
        np.savetxt('Test_images/movie_yolo_resnet/error_list', error_list)
        print(evaluation_min)
        print(error_min)
